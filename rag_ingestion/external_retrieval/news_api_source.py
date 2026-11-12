from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from .base_source import BaseExternalSource
from .proxy_policy import provider_proxy_overrides
from .source_schema import ExternalEvidence

logger = logging.getLogger(__name__)


class NewsAPISource(BaseExternalSource):
    search_capable = True
    returns_full_text = False
    news_capable = True
    supports_temporal_filtering = True
    requires_authentication = True
    """External news discovery provider using the NewsAPI /v2/everything endpoint."""

    supported_domains = frozenset(
        {"medical", "political", "science", "finance", "general"}
    )

    def __init__(self, timeout: int = 4) -> None:
        self.name = "news_api"
        self.timeout = min(float(timeout), 4.0)
        self.base_url = "https://newsapi.org/v2/everything"
        self.last_http_status = None

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

        api_key = os.environ.get("NEWS_API_KEY")
        if not api_key:
            raise RuntimeError("NEWS_API_KEY is not configured.")

        params: Dict[str, Any] = {
            "q": query.strip(),
            "pageSize": min(top_k, 100),
            "sortBy": "relevancy",
        }

        params["language"] = language.strip() if language and language.strip() else "en"
        # Apply a bounded date range only when the grounded query itself names
        # a month and year. The original atomic claim remains the source of the
        # date; no date is inferred or added here.
        month_match = re.search(
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})\b",
            query, flags=re.IGNORECASE,
        )
        if month_match:
            try:
                month_start = datetime.strptime(
                    f"{month_match.group(1)} {month_match.group(2)}", "%B %Y"
                ).date()
                today = datetime.now(timezone.utc).date()
                if month_start <= today:
                    params["from"] = month_start.isoformat()
                    params["to"] = today.isoformat()
            except ValueError:
                pass

        headers = {
            "X-Api-Key": api_key,
            "User-Agent": "Raven/1.0",
        }

        try:
            response = requests.get(
                self.base_url,
                params=params,
                headers=headers,
                timeout=self.request_timeout(),
                proxies=provider_proxy_overrides(self.base_url),
            )
            self.last_http_status = response.status_code
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.Timeout as exc:
            raise RuntimeError("NewsAPI request timed out") from exc

        except requests.exceptions.HTTPError as exc:
            status_code = response.status_code

            if status_code in (401, 403):
                raise RuntimeError(
                    f"NewsAPI authentication error: HTTP {status_code}"
                ) from exc

            if status_code == 429:
                raise RuntimeError(
                    "NewsAPI rate limit exceeded"
                ) from exc

            raise RuntimeError(
                f"NewsAPI request failed with HTTP {status_code}"
            ) from exc

        except requests.exceptions.RequestException as exc:
            raise RuntimeError("NewsAPI request failed") from exc

        except ValueError as exc:
            raise RuntimeError(
                f"Invalid JSON response from NewsAPI: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(
                "Unexpected response structure: JSON is not a dictionary"
            )

        if data.get("status") != "ok":
            code = data.get("code", "unknown")
            message = data.get("message", "Unknown NewsAPI error")
            raise RuntimeError(
                f"NewsAPI error {code}: {message}"
            )

        articles = data.get("articles", [])
        if not isinstance(articles, list):
            raise RuntimeError(
                "Unexpected response structure: 'articles' is not a list"
            )

        results: List[ExternalEvidence] = []

        for index, article in enumerate(articles[:top_k]):
            if not isinstance(article, dict):
                continue

            url = article.get("url") or ""
            title = article.get("title") or ""
            description = article.get("description") or ""
            content = article.get("content") or ""

            if not url:
                continue

            source_info = article.get("source") or {}
            source_name = source_info.get("name") or "NewsAPI"
            source_id = source_info.get("id")

            text_parts = []

            if description:
                text_parts.append(description)

            if content and content != description:
                text_parts.append(content)

            text = " | ".join(text_parts).strip()

            if not text:
                text = title or "No textual summary available."

            evidence_id = (
                "newsapi_"
                + hashlib.sha256(url.encode("utf-8")).hexdigest()
            )

            published_at = article.get("publishedAt")
            publication_date = None

            if published_at:
                try:
                    publication_date = datetime.fromisoformat(
                        published_at.replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except ValueError:
                    pass

            metadata: Dict[str, Any] = {
                "provider": "NewsAPI",
                "text_limitation": (
                    "NewsAPI provides article metadata and may provide "
                    "truncated article content; the source URL can be "
                    "retrieved separately for full-page extraction."
                ),
            }

            if source_id:
                metadata["source_id"] = source_id

            if source_name:
                metadata["publisher"] = source_name

            if published_at:
                metadata["published_at"] = published_at

            results.append(
                ExternalEvidence(
                    evidence_id=evidence_id,
                    text=text,
                    source=source_name,
                    title=title,
                    url=url,
                    domain=domain or "general",
                    retrieved_date=datetime.utcnow(),
                    metadata=metadata,
                    publication_date=publication_date,
                    language=language,
                    source_type="news_api",
                    authority_score=0.95,
                )
            )

        return results
