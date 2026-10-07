import logging
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
import requests
from urllib.parse import urlsplit

from rag_ingestion.crawler.robots import fetch_robots_txt_sync, is_allowed
from rag_ingestion.extractor.html_cleaner import HTMLCleaner

logger = logging.getLogger(__name__)

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
    def __init__(self, timeout: int = 15, max_size: int = 5 * 1024 * 1024, user_agent: str = "AntigravityIDE/1.0 (WebRetriever)"):
        self.timeout = timeout
        self.max_size = max_size
        self.user_agent = user_agent
        self.html_cleaner = HTMLCleaner()

    def retrieve(self, url: str) -> WebRetrievalResult:
        result = WebRetrievalResult(url=url, retrieved_date=datetime.utcnow())

        parsed = urlsplit(url) if isinstance(url, str) else None
        if not parsed or parsed.scheme.lower() not in {"http", "https"}:
            result.error = (
                "Unsupported URL scheme: " + url
                if isinstance(url, str) and "@" not in url
                else "Unsupported URL scheme"
            )
            return result
        if parsed.username is not None or parsed.password is not None:
            result.error = "Credential-bearing URLs are not permitted"
            return result

        try:
            # Respect robots.txt
            rp = fetch_robots_txt_sync(url, timeout=self.timeout)
            if not is_allowed(rp, self.user_agent, url):
                result.error = "Crawling disallowed by robots.txt"
                return result

            headers = {"User-Agent": self.user_agent}

            # Using stream=True to enforce size limit
            response = requests.get(url, headers=headers, timeout=self.timeout, stream=True)
            result.status_code = response.status_code
            result.final_url = response.url
            result.content_type = response.headers.get("Content-Type", "")

            response.raise_for_status()

            if not result.content_type or "text/html" not in result.content_type.lower():
                result.error = f"Non-HTML content type: {result.content_type}"
                return result

            content_bytes = b""
            for chunk in response.iter_content(chunk_size=8192):
                content_bytes += chunk
                if len(content_bytes) > self.max_size:
                    result.error = f"Response size exceeds limit of {self.max_size} bytes"
                    return result

            if not content_bytes:
                result.error = "Empty response"
                return result

            source_metadata = {"url": result.final_url}
            doc = self.html_cleaner.extract_from_content(content_bytes, source_metadata)

            if doc and doc.text:
                result.text = doc.text
            else:
                result.error = "No meaningful article text could be extracted."

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
