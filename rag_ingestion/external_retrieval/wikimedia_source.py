# src/external_retrieval/wikimedia_source.py
"""External evidence provider for the Wikimedia MediaWiki API.

Wikimedia content is retrieved as general encyclopedic evidence only.  This
provider does not assess truth, authority, or claim verdicts.
"""

from __future__ import annotations

import html
import hashlib
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

from .base_source import BaseExternalSource
from .source_schema import ExternalEvidence

logger = logging.getLogger(__name__)

# Conservative score on the existing 0-10 authority scale. Wikimedia is a
# useful reference source, but it is crowd-sourced and not proof by itself.
ENCYCLOPEDIC_AUTHORITY_SCORE = 5.5


class WikimediaSource(BaseExternalSource):
    """Concrete external source for anonymous Wikimedia search and page data."""

    def __init__(self, timeout: int = 30) -> None:
        self.name = "wikimedia"
        self.timeout = timeout
        self.base_url = "https://en.wikipedia.org/w/api.php"

    @staticmethod
    def _string(value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    def _clean_text(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            return ""
        value = html.unescape(value)
        value = re.sub(r"<[^>]+>", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _page_id(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return None

    @staticmethod
    def _fallback_id(title: str, index: int) -> str:
        material = f"{title}|{index}".encode("utf-8")
        return hashlib.sha1(material).hexdigest()[:16]

    def _request_json(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = requests.get(
                self.base_url,
                params=params,
                headers={"User-Agent": "MERIDIAN/1.0 (evidence retrieval)"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(f"Wikimedia API request timed out: {exc}") from exc
        except requests.exceptions.HTTPError as exc:
            status_code = getattr(response, "status_code", None)
            if status_code == 429:
                raise RuntimeError("Wikimedia API rate limit exceeded") from exc
            raise RuntimeError(f"Wikimedia API request failed: {exc}") from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Wikimedia API request failed: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"Invalid JSON response from Wikimedia: {exc}") from exc

        if not isinstance(data, dict):
            raise RuntimeError("Unexpected response structure from Wikimedia")
        return data

    @staticmethod
    def _page_url(page: Dict[str, Any], title: str) -> str:
        for key in ("fullurl", "canonicalurl"):
            value = page.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'), safe='()/:') }"

    @staticmethod
    def _revision_metadata(page: Dict[str, Any]) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        revisions = page.get("revisions")
        if isinstance(revisions, list) and revisions and isinstance(revisions[0], dict):
            revision = revisions[0]
            for key, metadata_key in (("revid", "revision_id"), ("timestamp", "timestamp")):
                if revision.get(key) not in (None, ""):
                    metadata[metadata_key] = revision[key]
        return metadata

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

        search_data = self._request_json({
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": top_k,
        })
        query_data = search_data.get("query")
        search_results = query_data.get("search") if isinstance(query_data, dict) else None
        if search_results is None:
            return []
        if not isinstance(search_results, list):
            raise RuntimeError("Unexpected response structure: Wikimedia query.search is not a list")
        search_results = [item for item in search_results[:top_k] if isinstance(item, dict)]
        if not search_results:
            return []

        page_ids = [str(page_id) for item in search_results if (page_id := self._page_id(item.get("pageid"))) is not None]
        pages: Dict[str, Dict[str, Any]] = {}
        if page_ids:
            content_data = self._request_json({
                "action": "query",
                "pageids": "|".join(page_ids),
                "prop": "extracts|info|revisions",
                "explaintext": 1,
                "inprop": "url",
                "rvprop": "ids|timestamp",
                "rvlimit": 1,
                "format": "json",
            })
            content_query = content_data.get("query")
            content_pages = content_query.get("pages") if isinstance(content_query, dict) else None
            if content_pages is None:
                content_pages = {}
            if not isinstance(content_pages, dict):
                raise RuntimeError("Unexpected response structure: Wikimedia query.pages is not an object")
            pages = {str(key): value for key, value in content_pages.items() if isinstance(value, dict)}

        retrieved_date = datetime.utcnow()
        results: List[ExternalEvidence] = []
        for index, search_item in enumerate(search_results):
            page_id = self._page_id(search_item.get("pageid"))
            page = pages.get(str(page_id), {}) if page_id is not None else {}
            title = self._string(page.get("title")) or self._string(search_item.get("title")) or "Untitled Wikimedia page"
            extract = self._clean_text(page.get("extract"))
            snippet = self._clean_text(search_item.get("snippet"))
            text = extract or snippet
            if not text:
                logger.warning("Skipping Wikimedia result %d: no page extract or search snippet", index)
                continue

            page_id_text = str(page_id) if page_id is not None else self._fallback_id(title, index)
            metadata: Dict[str, Any] = {
                "provider": "Wikimedia",
                "pageid": page_id if page_id is not None else page_id_text,
                "title": title,
                "snippet": snippet,
                "source_project": "English Wikipedia",
                "text_limitation": "Wikimedia is crowd-sourced encyclopedic content and is not proof of a claim by itself.",
            }
            for key in ("ns", "wordcount", "length"):
                if search_item.get(key) not in (None, ""):
                    metadata[key] = search_item[key]
            metadata.update(self._revision_metadata(page))
            url = self._page_url(page, title)
            metadata["canonical_url"] = url
            results.append(
                ExternalEvidence(
                    evidence_id=f"wikimedia_{page_id_text}",
                    text=text,
                    source="Wikimedia",
                    title=title,
                    url=url,
                    domain="general",
                    retrieved_date=retrieved_date,
                    metadata=metadata,
                    publication_date=None,
                    language=language or "en",
                    source_type="encyclopedic",
                    authority_score=ENCYCLOPEDIC_AUTHORITY_SCORE,
                )
            )
        return results
    supported_domains = frozenset({"medical", "political", "science", "finance", "general"})
