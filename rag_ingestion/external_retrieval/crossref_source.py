# src/external_retrieval/crossref_source.py
"""External evidence provider for the Crossref Works REST API.

Crossref primarily supplies scholarly bibliographic metadata.  This provider
preserves that distinction: its output is evidence about a publication record,
not an interpretation or adjudication of the claim being searched.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from .base_source import BaseExternalSource
from .source_schema import ExternalEvidence

logger = logging.getLogger(__name__)

# Scholarly publication metadata is authoritative as a source record, but it
# does not establish the truth of a claim.  This remains on the project's
# existing 0-10 authority scale.
SCHOLARLY_METADATA_AUTHORITY_SCORE = 8.0


class CrossrefSource(BaseExternalSource):
    """Concrete external source for the Crossref Works API."""

    def __init__(self, timeout: int = 30) -> None:
        self.name = "crossref"
        self.timeout = timeout
        self.base_url = "https://api.crossref.org/works"

    @staticmethod
    def _first_string(value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    def _first_list_string(value: Any) -> str:
        if isinstance(value, list) and value and isinstance(value[0], str):
            return value[0].strip()
        return ""

    @staticmethod
    def _date(record: Dict[str, Any]) -> Optional[datetime]:
        """Return the best available Crossref date as a naive UTC datetime."""
        for field in ("published", "published-print", "published-online", "issued", "created", "updated"):
            value = record.get(field)
            date_parts = value.get("date-parts") if isinstance(value, dict) else None
            if not isinstance(date_parts, list) or not date_parts or not isinstance(date_parts[0], list):
                continue
            parts = date_parts[0]
            if not parts or not isinstance(parts[0], int):
                continue
            try:
                year = parts[0]
                month = parts[1] if len(parts) > 1 and isinstance(parts[1], int) else 1
                day = parts[2] if len(parts) > 2 and isinstance(parts[2], int) else 1
                return datetime(year, month, day)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _doi(record: Dict[str, Any]) -> str:
        return CrossrefSource._first_string(record.get("DOI")).lower()

    @staticmethod
    def _work_id(record: Dict[str, Any], index: int) -> str:
        """Use Crossref's work id, with a deterministic record fallback."""
        work_id = CrossrefSource._first_string(record.get("id"))
        if work_id:
            return work_id
        url = CrossrefSource._first_string(record.get("URL"))
        if url:
            return url.rstrip("/").rsplit("/", 1)[-1]
        material = "|".join(
            [CrossrefSource._first_list_string(record.get("title")), str(index)]
        )
        return hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _authors(record: Dict[str, Any]) -> List[Dict[str, str]]:
        authors = record.get("author")
        if not isinstance(authors, list):
            return []
        result = []
        for author in authors:
            if not isinstance(author, dict):
                continue
            item = {
                key: author[key].strip()
                for key in ("given", "family", "sequence", "ORCID")
                if isinstance(author.get(key), str) and author[key].strip()
            }
            if item:
                result.append(item)
        return result

    @staticmethod
    def _clean_abstract(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            return ""
        # Crossref abstracts are often JATS/XML fragments. Keep the returned
        # content readable without interpreting or changing its meaning.
        text = re.sub(r"<[^>]+>", " ", value)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _bibliographic_text(
        title: str, source: str, publisher: str, publication_date: Optional[datetime]
    ) -> str:
        parts = [part for part in (title, source, publisher) if part]
        if publication_date:
            parts.append(publication_date.strftime("%Y-%m-%d"))
        description = "; ".join(parts) if parts else "No bibliographic metadata available"
        return f"Bibliographic record only: {description}."

    def _metadata(
        self, record: Dict[str, Any], doi: str, title: str, source: str, publisher: str
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"provider": "Crossref"}
        values = {
            "DOI": doi,
            "publisher": publisher,
            "container_title": source,
            "URL": self._first_string(record.get("URL")),
            "type": self._first_string(record.get("type")),
            "title": title,
        }
        for key, value in values.items():
            if value:
                metadata[key.lower() if key != "DOI" else "doi"] = value
        authors = self._authors(record)
        if authors:
            metadata["authors"] = authors
        for source_key, metadata_key in (
            ("ISSN", "issn"),
            ("reference-count", "reference_count"),
            ("is-referenced-by-count", "citation_count"),
            ("license", "license"),
            ("link", "links"),
            ("published", "published"),
            ("published-print", "published_print"),
            ("published-online", "published_online"),
            ("issued", "issued"),
            ("created", "created"),
            ("updated", "updated"),
        ):
            value = record.get(source_key)
            if value not in (None, "", [], {}):
                metadata[metadata_key] = value
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

        params = {"query.bibliographic": query, "rows": top_k}
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
            raise RuntimeError(f"Crossref API request timed out: {exc}") from exc
        except requests.exceptions.HTTPError as exc:
            status_code = getattr(response, "status_code", None)
            if status_code == 429:
                raise RuntimeError("Crossref API rate limit exceeded") from exc
            raise RuntimeError(f"Crossref API request failed: {exc}") from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Crossref API request failed: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"Invalid JSON response from Crossref: {exc}") from exc

        if not isinstance(data, dict):
            raise RuntimeError("Unexpected response structure from Crossref")
        message = data.get("message")
        items = message.get("items") if isinstance(message, dict) else None
        if items is None:
            return []
        if not isinstance(items, list):
            raise RuntimeError("Unexpected response structure: Crossref message.items is not a list")

        results: List[ExternalEvidence] = []
        for index, record in enumerate(items[:top_k]):
            if not isinstance(record, dict):
                logger.warning("Skipping Crossref record %d: entry is not an object", index)
                continue

            doi = self._doi(record)
            title = self._first_list_string(record.get("title")) or "Untitled Crossref work"
            source = (
                self._first_list_string(record.get("container-title"))
                or self._first_string(record.get("publisher"))
                or "Crossref"
            )
            publisher = self._first_string(record.get("publisher"))
            publication_date = self._date(record)
            url = f"https://doi.org/{doi}" if doi else self._first_string(record.get("URL"))
            abstract = self._clean_abstract(record.get("abstract"))
            text = abstract or self._bibliographic_text(title, source, publisher, publication_date)
            work_id = doi or self._work_id(record, index)
            evidence_id = f"crossref_{work_id}"

            metadata = self._metadata(record, doi, title, source, publisher)
            metadata["text_limitation"] = (
                "Crossref primarily provides bibliographic metadata; metadata alone is not proof of a scientific claim."
            )
            results.append(
                ExternalEvidence(
                    evidence_id=evidence_id,
                    text=text,
                    source=source,
                    title=title,
                    url=url,
                    domain="science",
                    retrieved_date=datetime.utcnow(),
                    metadata=metadata,
                    publication_date=publication_date,
                    language=self._first_string(record.get("language")) or None,
                    source_type="scientific_literature",
                    authority_score=SCHOLARLY_METADATA_AUTHORITY_SCORE,
                )
            )
        return results
    supported_domains = frozenset({"medical", "science"})
