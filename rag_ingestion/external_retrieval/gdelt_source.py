import hashlib
import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import List, Optional, Dict, Any
import requests

from .base_source import BaseExternalSource
from .source_schema import ExternalEvidence

logger = logging.getLogger(__name__)

class GDELTSource(BaseExternalSource):
    """Concrete external source for the GDELT Project."""

    def __init__(
        self,
        timespan: str = "7d",
        timeout: int = 30,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds cannot be negative")
        self.name = "gdelt"
        self.timespan = timespan
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.base_url = "https://api.gdeltproject.org/api/v2/doc/doc"

    @staticmethod
    def _retry_after_seconds(value: Optional[str]) -> Optional[float]:
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None

    def search(
        self,
        query: str,
        domain: Optional[str] = None,
        top_k: int = 5,
        language: Optional[str] = None,
        country: Optional[str] = None,
    ) -> List[ExternalEvidence]:
        if not query or not query.strip():
            raise ValueError("Query must be non-empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        query_parts = [query]
        if language and language.strip():
            query_parts.append(f"sourceLang:{language.strip()}")
        if country and country.strip():
            query_parts.append(f"sourceCountry:{country.strip().upper()}")

        params = {
            "query": " ".join(query_parts),
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(top_k),
            "timespan": self.timespan
        }

        headers = {
            "User-Agent": "AntigravityIDE/1.0 (test_script)"
        }
        response = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.get(
                    self.base_url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.exceptions.Timeout as exc:
                raise RuntimeError(f"GDELT API request timed out: {exc}") from exc
            except requests.exceptions.RequestException as exc:
                raise RuntimeError(f"GDELT API request failed: {exc}") from exc

            if response.status_code != 429:
                break

            if attempt >= self.max_retries:
                try:
                    response.raise_for_status()
                except requests.exceptions.RequestException as exc:
                    raise RuntimeError(f"GDELT API request failed: {exc}") from exc
                raise RuntimeError("GDELT API request failed: HTTP 429 Too Many Requests")

            retry_after = self._retry_after_seconds(response.headers.get("Retry-After"))
            delay = retry_after
            if delay is None:
                delay = self.retry_backoff_seconds * (2 ** attempt)
            logger.warning(
                "GDELT rate limited the request; retrying attempt %d/%d after %.2f seconds",
                attempt + 1,
                self.max_retries,
                delay,
            )
            time.sleep(delay)

        try:
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(f"GDELT API request timed out: {exc}") from exc
        except ValueError as exc:
            headers = getattr(response, "headers", {})
            content_type = headers.get("Content-Type", "unknown")
            if not isinstance(content_type, str):
                content_type = "unknown"
            content = getattr(response, "content", None)
            if isinstance(content, (bytes, bytearray)):
                response_length = f"{len(content)} bytes"
            elif isinstance(content, str):
                response_length = f"{len(content.encode('utf-8'))} bytes"
            else:
                response_length = "unknown length"
            raise RuntimeError(
                "Invalid JSON response from GDELT "
                f"(HTTP {response.status_code}, Content-Type '{content_type}', "
                f"response length {response_length}): {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"GDELT API request failed: {exc}") from exc

        if not isinstance(data, dict):
            raise RuntimeError("Unexpected response structure: JSON is not a dictionary")

        articles = data.get("articles", [])
        if not isinstance(articles, list):
            raise RuntimeError("Unexpected response structure: 'articles' is not a list")

        results = []
        for idx, art in enumerate(articles[:top_k]):
            url = art.get("url", "")

            if url:
                evidence_id = f"gdelt_{hashlib.sha256(url.encode('utf-8')).hexdigest()}"
            else:
                fallback = f"{art.get('title', '')}_{art.get('domain', '')}_{idx}"
                evidence_id = f"gdelt_{hashlib.sha256(fallback.encode('utf-8')).hexdigest()}"

            title = art.get("title", "")
            source_domain = art.get("domain", "")

            # GDELT mode=artlist does not return the article body
            text = f"{title}" if title else "No content available."

            pub_date = None
            seendate = art.get("seendate")
            if seendate:
                try:
                    pub_date = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ")
                except ValueError:
                    pass

            lang = art.get("language")
            if not lang:
                lang = None

            metadata: Dict[str, Any] = {
                "text_limitation": "GDELT article list API provides title and metadata, not the full article body."
            }
            if source_domain:
                metadata["source_domain"] = source_domain
            if "sourcecountry" in art:
                metadata["sourcecountry"] = art["sourcecountry"]
            if seendate:
                metadata["seendate"] = seendate
            if lang:
                metadata["language"] = lang

            ev = ExternalEvidence(
                evidence_id=evidence_id,
                text=text,
                source=source_domain if source_domain else "GDELT",
                title=title,
                url=url,
                domain=domain if domain else "general",
                retrieved_date=datetime.utcnow(),
                metadata=metadata,
                publication_date=pub_date,
                language=lang,
                source_type="gdelt",
                authority_score=None
            )
            results.append(ev)

        return results
    supported_domains = frozenset({"medical", "political", "science", "finance", "general"})
