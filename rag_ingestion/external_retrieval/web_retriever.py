import logging
import ipaddress
import socket
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
import requests
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser
from urllib3 import connection as urllib3_connection
from urllib3 import connectionpool as urllib3_connectionpool
from urllib3 import poolmanager as urllib3_poolmanager
from urllib3.util.connection import create_connection
from requests.adapters import HTTPAdapter

from rag_ingestion.crawler.robots import fetch_robots_txt_sync, is_allowed
from rag_ingestion.extractor.html_cleaner import HTMLCleaner
from rag_ingestion.llm_resilience import remaining_request_budget

logger = logging.getLogger(__name__)
MAX_REDIRECTS = 3


def _validated_destination_ip(url: str) -> Optional[str]:
    """Resolve and validate the address that the socket will actually use."""
    if not _is_public_http_url(url):
        return None
    parsed = urlsplit(url)
    hostname = parsed.hostname.rstrip(".").casefold()
    try:
        try:
            addresses = {ipaddress.ip_address(hostname)}
        except ValueError:
            addresses = {
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(
                    hostname,
                    parsed.port or (443 if parsed.scheme.lower() == "https" else 80),
                    type=socket.SOCK_STREAM,
                )
            }
        if not addresses or not all(address.is_global for address in addresses):
            return None
        return str(sorted(addresses, key=lambda address: (address.version, int(address)))[0])
    except (OSError, TypeError, ValueError):
        return None


def _pinned_session(destination_ip: str) -> requests.Session:
    """Create a requests session whose socket connects to one validated IP.

    The URL hostname remains in urllib3's connection object, preserving Host,
    TLS SNI, and certificate hostname verification. Only the TCP destination
    is pinned. Environment proxies are disabled because a proxy would perform
    its own DNS resolution and break that guarantee.
    """
    class PinnedSocketMixin:
        def _new_conn(self):
            return create_connection(
                (destination_ip, self.port),
                self.timeout,
                source_address=self.source_address,
                socket_options=self.socket_options,
            )

    class PinnedHTTPConnection(PinnedSocketMixin, urllib3_connection.HTTPConnection):
        pass

    class PinnedHTTPSConnection(PinnedSocketMixin, urllib3_connection.HTTPSConnection):
        pass

    http_pool = type(
        "PinnedHTTPConnectionPool",
        (urllib3_connectionpool.HTTPConnectionPool,),
        {"ConnectionCls": PinnedHTTPConnection},
    )
    https_pool = type(
        "PinnedHTTPSConnectionPool",
        (urllib3_connectionpool.HTTPSConnectionPool,),
        {"ConnectionCls": PinnedHTTPSConnection},
    )

    class PinnedHTTPAdapter(HTTPAdapter):
        def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
            self.poolmanager = urllib3_poolmanager.PoolManager(
                num_pools=connections,
                maxsize=maxsize,
                block=block,
                **pool_kwargs,
            )
            self.poolmanager.pool_classes_by_scheme = {
                "http": http_pool,
                "https": https_pool,
            }

    session = requests.Session()
    session.trust_env = False
    adapter = PinnedHTTPAdapter()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _get_pinned(url: str, destination_ip: str, **kwargs):
    session = _pinned_session(destination_ip)
    try:
        return session.get(url, **kwargs), session
    except Exception:
        session.close()
        raise


def _fetch_robots_txt_pinned(base_url: str, timeout: float, user_agent: str):
    parsed = urlsplit(base_url)
    current_url = urljoin(f"{parsed.scheme}://{parsed.netloc}", "/robots.txt")
    for redirect_count in range(MAX_REDIRECTS + 1):
        destination_ip = _validated_destination_ip(current_url)
        if destination_ip is None:
            return None
        response = session = None
        try:
            response, session = _get_pinned(
                current_url,
                destination_ip,
                headers={"User-Agent": user_agent},
                timeout=timeout,
                allow_redirects=False,
            )
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                if not location or redirect_count == MAX_REDIRECTS:
                    return None
                current_url = urljoin(current_url, location)
                continue
            if response.status_code != 200:
                return None
            parser = RobotFileParser()
            parser.set_url(current_url)
            parser.parse(response.text.splitlines())
            return parser
        except requests.exceptions.RequestException:
            return None
        finally:
            if response is not None:
                response.close()
            if session is not None:
                session.close()
    return None


def _is_public_http_url(url: str) -> bool:
    """Reject non-public destinations before making any outbound request."""
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return False
        if parsed.username is not None or parsed.password is not None:
            return False
        hostname = parsed.hostname.rstrip(".").casefold()
        if (
            hostname in {"localhost", "metadata"}
            or hostname.endswith((".localhost", ".local", ".internal"))
        ):
            return False
        try:
            addresses = {ipaddress.ip_address(hostname)}
        except ValueError:
            addresses = {
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(
                    hostname,
                    parsed.port or (443 if parsed.scheme.lower() == "https" else 80),
                    type=socket.SOCK_STREAM,
                )
            }
        return bool(addresses) and all(address.is_global for address in addresses)
    except (OSError, TypeError, ValueError):
        return False

class WebRetrievalResult(BaseModel):
    url: str
    final_url: Optional[str] = None
    status_code: Optional[int] = None
    content_type: Optional[str] = None
    text: Optional[str] = None
    retrieved_date: datetime
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}

class WebSourceRetriever:
    """Component to retrieve and extract text from web pages.
    Respects robots.txt, enforces size limits, and reuses the HTMLCleaner.
    """
    search_capable = False
    fetch_only = True
    returns_full_text = True
    def __init__(self, timeout: int = 15, max_size: int = 5 * 1024 * 1024, user_agent: str = "AntigravityIDE/1.0 (WebRetriever)"):
        self.timeout = timeout
        self.max_size = max_size
        self.user_agent = user_agent
        self.html_cleaner = HTMLCleaner()

    def request_timeout(self) -> float:
        """Return this retriever's timeout bounded by the active request deadline."""
        configured = float(self.timeout)
        remaining = remaining_request_budget()
        if remaining is None:
            return configured
        return max(0.001, min(configured, remaining))

    def retrieve(self, url: str) -> WebRetrievalResult:
        result = WebRetrievalResult(url=url, retrieved_date=datetime.utcnow())

        if not isinstance(url, str) or not _is_public_http_url(url):
            result.error = "URL is invalid or does not resolve exclusively to public addresses"
            return result

        try:
            headers = {"User-Agent": self.user_agent}
            current_url = url
            for redirect_count in range(MAX_REDIRECTS + 1):
                if remaining_request_budget() is not None and remaining_request_budget() <= 0:
                    result.error = "Request deadline exceeded"
                    return result
                if not _is_public_http_url(current_url):
                    result.error = "Redirect target is not a public HTTP(S) URL"
                    return result
                rp = _fetch_robots_txt_pinned(
                    current_url,
                    timeout=self.request_timeout(),
                    user_agent=self.user_agent,
                )
                if not is_allowed(rp, self.user_agent, current_url):
                    result.error = "Crawling disallowed by robots.txt"
                    return result

                destination_ip = _validated_destination_ip(current_url)
                if destination_ip is None:
                    result.error = "URL is invalid or does not resolve exclusively to public addresses"
                    return result
                response = session = None
                try:
                    response, session = _get_pinned(
                        current_url,
                        destination_ip,
                        headers=headers,
                        timeout=self.request_timeout(),
                        stream=True,
                        allow_redirects=False,
                    )
                    result.status_code = response.status_code
                    result.final_url = response.url
                    result.content_type = response.headers.get("Content-Type", "")
                    if not _is_public_http_url(result.final_url):
                        result.error = "Response URL is not a public HTTP(S) URL"
                        return result

                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("Location")
                        if not location:
                            result.error = "Redirect response has no Location header"
                            return result
                        if redirect_count == MAX_REDIRECTS:
                            result.error = "Too many redirects"
                            return result
                        current_url = urljoin(current_url, location)
                        continue

                    response.raise_for_status()
                    if not result.content_type or "text/html" not in result.content_type.lower():
                        result.error = f"Non-HTML content type: {result.content_type}"
                        return result

                    content_bytes = bytearray()
                    for chunk in response.iter_content(chunk_size=8192):
                        content_bytes.extend(chunk)
                        if len(content_bytes) > self.max_size:
                            result.error = f"Response size exceeds limit of {self.max_size} bytes"
                            return result
                    if not content_bytes:
                        result.error = "Empty response"
                        return result

                    source_metadata = {"url": result.final_url}
                    doc = self.html_cleaner.extract_from_content(
                        bytes(content_bytes), source_metadata
                    )
                    if doc and doc.text:
                        result.text = doc.text
                    else:
                        result.error = "No meaningful article text could be extracted."
                    return result
                finally:
                    if response is not None:
                        response.close()
                    if session is not None:
                        session.close()

            result.error = "Too many redirects"

        except requests.exceptions.Timeout:
            result.error = "Request timed out"
        except requests.exceptions.ConnectionError:
            result.error = "Connection failed"
        except requests.exceptions.HTTPError as e:
            result.error = f"HTTP error: {e}"
        except requests.exceptions.RequestException as e:
            result.error = f"Request failed: {e}"
        except Exception as e:
            result.error = f"Unexpected error: {e}"

        return result
