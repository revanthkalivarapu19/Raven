"""Credential-safe connectivity diagnostics for Raven's configured providers.

Run from the Raven virtual environment with:
    .venv\\Scripts\\python.exe benchmarks\\provider_connectivity.py

Reports sanitized proxy metadata, basic HTTPS checks, and one request through
each configured adapter with Requests environment inheritance enabled/disabled.
No credential values, authenticated URLs, or exception messages are printed.
"""

from __future__ import annotations

import json
import os
import socket
import ssl
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
import requests

load_dotenv(ROOT / ".env")

from rag_ingestion.external_retrieval import fact_check_source, gdelt_source, news_api_source
from rag_ingestion.external_retrieval.fact_check_source import FactCheckSource
from rag_ingestion.external_retrieval.gdelt_source import GDELTSource
from rag_ingestion.external_retrieval.news_api_source import NewsAPISource


PROXY_NAMES = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy",
    "all_proxy", "NO_PROXY", "no_proxy",
)
PROBES = (
    "https://www.google.com/generate_204",
    "https://www.cloudflare.com/cdn-cgi/trace",
    "https://api.github.com/rate_limit",
)
PROVIDER_QUERIES = {
    "news_api": "current news",
    "fact_check": "smoking cancer",
    "gdelt": "New Delhi air quality",
}


def _host_port(value: str) -> tuple[str | None, int | None]:
    try:
        parsed = urlsplit(value if "://" in value else f"//{value}")
        return parsed.hostname, parsed.port
    except ValueError:
        return None, None


def _tcp(host: str | None, port: int | None, timeout: float = 2.0) -> bool | None:
    if not host or not port:
        return None
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _dns(host: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        addresses = sorted({row[4][0] for row in socket.getaddrinfo(host, None)})
        return {"result": "resolved", "address_count": len(addresses),
                "runtime_ms": round((time.perf_counter() - started) * 1000)}
    except socket.gaierror as exc:
        return {"result": "DNS_error", "error_type": type(exc).__name__,
                "runtime_ms": round((time.perf_counter() - started) * 1000)}


def _tls(host: str, port: int = 443, timeout: float = 4.0) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        raw = socket.create_connection((host, port), timeout=timeout)
        with raw, ssl.create_default_context().wrap_socket(raw, server_hostname=host):
            return {"result": "success",
                    "runtime_ms": round((time.perf_counter() - started) * 1000)}
    except socket.gaierror as exc:
        result, error_type = "DNS_error", type(exc).__name__
    except ssl.SSLError as exc:
        result, error_type = "tls_error", type(exc).__name__
    except (TimeoutError, socket.timeout) as exc:
        result, error_type = "timeout", type(exc).__name__
    except OSError as exc:
        result, error_type = "connection_error", type(exc).__name__
    return {"result": result, "error_type": error_type,
            "runtime_ms": round((time.perf_counter() - started) * 1000)}


def _category(exc: BaseException, http_status: int | None = None) -> str:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    names = {type(item).__name__.casefold() for item in chain}
    if any("gaierror" in name or "nameresolution" in name for name in names):
        return "DNS_error"
    if any("ssl" in name or "tls" in name for name in names):
        return "tls_error"
    if any("timeout" in name for name in names):
        return "timeout"
    if http_status == 429:
        return "rate_limited"
    if http_status in {401, 403}:
        return "invalid_credentials"
    if any("jsondecode" in name or "decodeerror" in name for name in names):
        return "parse_error"
    if http_status is not None and http_status >= 400:
        return "http_error"
    if any("connectionerror" in name or "proxyerror" in name for name in names):
        return "connection_error"
    if isinstance(exc, (ValueError, TypeError)):
        return "invalid_response"
    return "error"


def _http_probe(url: str, trust_env: bool) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {"url_host": urlsplit(url).hostname,
                              "trust_env": trust_env}
    try:
        with requests.Session() as session:
            session.trust_env = trust_env
            response = session.get(url, timeout=(4, 7))
            result.update(http_status=response.status_code,
                          category="success" if response.ok else _category(
                              requests.HTTPError(response=response), response.status_code))
    except Exception as exc:
        result.update(http_status=None, category=_category(exc),
                      error_type=type(exc).__name__)
    result["runtime_ms"] = round((time.perf_counter() - started) * 1000)
    return result


def _run_adapter(name: str, trust_env: bool) -> dict[str, Any]:
    query = PROVIDER_QUERIES[name]
    adapter: Any = {
        "news_api": NewsAPISource(timeout=8),
        "fact_check": FactCheckSource(timeout=8),
        "gdelt": GDELTSource(timeout=8, max_retries=0),
    }[name]
    original_get = requests.get
    provider_module = {
        "news_api": news_api_source,
        "fact_check": fact_check_source,
        "gdelt": gdelt_source,
    }[name]
    original_proxy_policy = provider_module.provider_proxy_overrides
    # Measure raw environment inheritance, independent of Raven's protective
    # request-local override added after the diagnosis was established.
    provider_module.provider_proxy_overrides = lambda _url: None
    session = requests.Session()
    session.trust_env = trust_env
    if name == "gdelt":
        adapter.session.trust_env = trust_env
    else:
        # These adapters use requests.get; this routes the diagnostic request
        # through an explicitly configured Session without changing production.
        requests.get = session.get
    started = time.perf_counter()
    status = None
    try:
        items = adapter.search(query=query, top_k=1)
        status = getattr(adapter, "last_http_status", None)
        category = "success" if items else "empty_results"
        count = len(items)
        error_type = None
    except Exception as exc:  # exception text can contain authenticated URLs
        status = getattr(adapter, "last_http_status", None)
        category = _category(exc, status)
        count = 0
        error_type = type(exc).__name__
        cause = exc.__cause__ or exc.__context__
        if cause is not None:
            error_type = type(cause).__name__
            if status is None:
                status = getattr(getattr(cause, "response", None), "status_code", None)
    finally:
        requests.get = original_get
        provider_module.provider_proxy_overrides = original_proxy_policy
        session.close()
    return {
        "provider": name,
        "enabled": {
            "news_api": bool(os.getenv("NEWS_API_KEY", "").strip()),
            "fact_check": bool(os.getenv("GOOGLE_FACTCHECK_API_KEY", "").strip()),
            "gdelt": os.getenv("ENABLE_GDELT", "false").strip().lower()
            in {"1", "true", "yes", "on"},
        }[name],
        "endpoint_host": urlsplit(adapter.base_url).hostname,
        "trust_env": trust_env,
        "timeout_seconds": adapter.request_timeout(),
        "dns": _dns(urlsplit(adapter.base_url).hostname or ""),
        "tcp_443": _tcp(urlsplit(adapter.base_url).hostname, 443),
        "tls": _tls(urlsplit(adapter.base_url).hostname or ""),
        "http_status": status,
        "runtime_ms": round((time.perf_counter() - started) * 1000),
        "candidate_count": count,
        "failure_category": category,
        "error_type": error_type,
    }


def main() -> None:
    proxy_report = []
    for name in PROXY_NAMES:
        value = os.getenv(name)
        host, port = _host_port(value) if value else (None, None)
        proxy_report.append({
            "name": name,
            "exists": value is not None,
            "host": host,
            "port": port,
            "localhost": host in {"localhost", "127.0.0.1", "::1"},
            "tcp_reachable": _tcp(host, port) if host else None,
        })

    configured = {
        "NEWS_API_KEY": bool(os.getenv("NEWS_API_KEY", "").strip()),
        "GOOGLE_FACTCHECK_API_KEY": bool(
            os.getenv("GOOGLE_FACTCHECK_API_KEY", "").strip()),
        "ENABLE_GDELT": os.getenv("ENABLE_GDELT", "false").strip().lower()
        in {"1", "true", "yes", "on"},
    }
    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "provider_modules": {
            "news_api": news_api_source.__file__,
            "fact_check": fact_check_source.__file__,
            "gdelt": gdelt_source.__file__,
        },
        "configuration": configured,
        "searxng_enabled": bool(os.getenv("SEARXNG_BASE_URL", "").strip()),
        "proxies": proxy_report,
        "basic_https": {
            urlsplit(url).hostname: {
                "dns": _dns(urlsplit(url).hostname or ""),
                "tcp_443": _tcp(urlsplit(url).hostname, 443),
                "tls": _tls(urlsplit(url).hostname or ""),
                "http_trust_env_true": _http_probe(url, True),
                "http_trust_env_false": _http_probe(url, False),
            }
            for url in PROBES
        },
        "provider_checks": [
            _run_adapter(name, trust_env)
            for name in PROVIDER_QUERIES
            for trust_env in (True, False)
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
