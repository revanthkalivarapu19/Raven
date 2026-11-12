"""Optional SearXNG web-discovery provider.

Returned titles and snippets are discovery metadata only. The claim pipeline
must fetch source pages and validate their text before fusion.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

import requests

from .base_source import BaseExternalSource
from .source_schema import ExternalEvidence

logger = logging.getLogger(__name__)
# Optional integration: Raven must never assume a local service is running.
DEFAULT_SEARXNG_BASE_URL = ""


class SearXNGSource(BaseExternalSource):
    """Query a configurable SearXNG instance without API credentials."""

    name = "searxng"
    supported_domains = frozenset({"medical", "political", "science", "finance", "general"})
    search_capable = True
    fetch_only = False
    returns_full_text = False
    news_capable = False
    supports_temporal_filtering = True
    requires_authentication = False
    diagnostic_prefix = "searxng"

    def __init__(self, base_url: str | None = None, timeout: float = 5.0) -> None:
        configured = (
            os.getenv("SEARXNG_BASE_URL", DEFAULT_SEARXNG_BASE_URL)
            if base_url is None else base_url
        )
        self.base_url = configured.strip().rstrip("/")
        self.timeout = timeout
        self.enabled = bool(self.base_url)

    @classmethod
    def is_enabled(cls) -> bool:
        return bool(os.getenv("SEARXNG_BASE_URL", DEFAULT_SEARXNG_BASE_URL).strip())

    @staticmethod
    def _published_date(value: object) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None

    @staticmethod
    def _publisher(row: dict[str, Any], hostname: str) -> str:
        parsed = row.get("parsed_url")
        if isinstance(parsed, list) and len(parsed) > 1 and isinstance(parsed[1], str) and parsed[1].strip():
            return parsed[1].strip().lower().removeprefix("www.")
        return hostname.lower().removeprefix("www.")

    def search(
        self,
        query: str,
        domain: Optional[str] = None,
        top_k: int = 5,
        language: Optional[str] = None,
        country: Optional[str] = None,
    ) -> List[ExternalEvidence]:
        if not self.enabled:
            logger.info("searxng enabled=false outcome=disabled")
            return []
        if not query or not query.strip():
            raise ValueError("Query must be non-empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        parsed_base = urlsplit(self.base_url)
        if parsed_base.scheme not in {"http", "https"} or not parsed_base.hostname or parsed_base.username:
            logger.info("searxng outcome=config_error http_status=not_available duration_seconds=0.000 result_count=0")
            raise RuntimeError("SEARXNG_BASE_URL is invalid")

        endpoint = f"{self.base_url}/search"
        params: Dict[str, Any] = {"q": query.strip(), "format": "json"}
        if language and language.strip():
            params["language"] = language.strip()
        started = time.monotonic()
        status = "not_available"
        try:
            response = requests.get(
                endpoint,
                params=params,
                timeout=min(self.timeout, self.request_timeout()),
            )
            status = response.status_code
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout as exc:
            logger.info(
                "searxng enabled=true outcome=timeout http_status=%s duration_seconds=%.3f result_count=0",
                status, time.monotonic() - started,
            )
            raise RuntimeError("SearXNG request timed out") from exc
        except requests.exceptions.HTTPError as exc:
            logger.info(
                "searxng enabled=true outcome=http_error http_status=%s duration_seconds=%.3f result_count=0",
                status, time.monotonic() - started,
            )
            raise RuntimeError(f"SearXNG returned HTTP {status}") from exc
        except requests.exceptions.RequestException as exc:
            logger.info(
                "searxng enabled=true outcome=unavailable http_status=%s duration_seconds=%.3f result_count=0",
                status, time.monotonic() - started,
            )
            raise RuntimeError("SearXNG request failed") from exc
        except ValueError as exc:
            logger.info(
                "searxng enabled=true outcome=malformed_response http_status=%s duration_seconds=%.3f result_count=0",
                status, time.monotonic() - started,
            )
            raise RuntimeError("SearXNG returned malformed JSON") from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("results", []), list):
            logger.info(
                "searxng enabled=true outcome=malformed_response http_status=%s duration_seconds=%.3f result_count=0",
                status, time.monotonic() - started,
            )
            raise RuntimeError("SearXNG returned an invalid result list")

        results: List[ExternalEvidence] = []
        for row in payload.get("results", [])[: min(top_k, 5)]:
            if not isinstance(row, dict):
                continue
            url = row.get("url")
            title = row.get("title")
            snippet = row.get("content") or row.get("snippet")
            if not all(isinstance(value, str) and value.strip() for value in (url, title, snippet)):
                continue
            parsed_url = urlsplit(url.strip())
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname or parsed_url.username:
                continue
            publisher = self._publisher(row, parsed_url.hostname)
            published_at = row.get("publishedDate") or row.get("published_date")
            evidence_id = "searxng_" + hashlib.sha256(url.strip().encode("utf-8")).hexdigest()
            results.append(ExternalEvidence(
                evidence_id=evidence_id,
                text=snippet.strip(),
                source=publisher,
                title=title.strip(),
                url=url.strip(),
                domain=domain or "general",
                retrieved_date=datetime.utcnow(),
                publication_date=self._published_date(published_at),
                language=language,
                source_type="searxng_search",
                metadata={
                    "provider": "SearXNG",
                    "publisher": publisher,
                    "publisher_site": publisher,
                    "text_limitation": "Search snippet is discovery metadata; fetch and validate the source page before fusion.",
                    "discovery_only": True,
                },
            ))

        logger.info(
            "searxng enabled=true outcome=%s http_status=%s duration_seconds=%.3f result_count=%d",
            "success" if results else "zero_results",
            status,
            time.monotonic() - started,
            len(results),
        )
        return results
