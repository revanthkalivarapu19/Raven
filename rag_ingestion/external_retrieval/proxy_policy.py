"""Request-local handling for stale, unreachable loopback proxies."""

from __future__ import annotations

import socket
import logging
from urllib.parse import urlsplit

import requests

logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _proxy_endpoint(proxy_url: str) -> tuple[str | None, int | None]:
    try:
        parsed = urlsplit(proxy_url if "://" in proxy_url else f"//{proxy_url}")
        host = parsed.hostname
        if not host:
            return None, None
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 1080 if parsed.scheme.startswith("socks") else 80
        return host, port
    except ValueError:
        return None, None


def _tcp_reachable(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def provider_proxy_overrides(url: str) -> dict[str, str] | None:
    """Bypass only dead loopback proxies; preserve all other proxy behavior.

    Requests inherits process proxy variables by default. In development and
    managed environments those can contain a stale localhost endpoint. If
    every proxy applicable to this URL is loopback and unreachable, empty
    request-level overrides route this provider call directly while retaining
    Requests' other environment behavior, including CA bundle and netrc use.
    A reachable local proxy or any non-local proxy is always preserved.
    """
    inherited = requests.utils.get_environ_proxies(url)
    active = [inherited.get(key) for key in ("https", "http", "all") if inherited.get(key)]
    if not active:
        return None

    endpoints = [_proxy_endpoint(value) for value in active]
    if any(host not in _LOOPBACK_HOSTS or port is None for host, port in endpoints):
        return None
    if any(_tcp_reachable(host, port) for host, port in endpoints if host and port):
        return None

    safe_endpoint = next((item for item in endpoints if item[0] and item[1]), None)
    if safe_endpoint:
        logger.warning(
            "provider_http proxy_bypass=unreachable_loopback "
            "proxy_host=%s proxy_port=%d request_host=%s",
            safe_endpoint[0], safe_endpoint[1], urlsplit(url).hostname,
        )
    return {"http": "", "https": "", "all": ""}
