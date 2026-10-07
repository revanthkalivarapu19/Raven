# src/external_retrieval/europe_pmc_source.py
"""External evidence provider for the Europe PMC RESTful Web Service.

Europe PMC is a free, key-less literature index maintained by EMBL-EBI that
aggregates PubMed, PubMed Central, Agricola, the European Patent Office and
preprint servers.  This provider only *retrieves* bibliographic evidence; it
never interprets or adjudicates a claim.
"""

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from .base_source import BaseExternalSource
from .source_schema import ExternalEvidence

logger = logging.getLogger(__name__)

# Constant authority score for indexed, peer-reviewed medical literature.
# ``EvidenceFusion`` normalises ``authority_score`` by dividing by 10, so this
# constant is expressed on the project's 0-10 scale and contributes ~0.85 to
# the normalised authority term.  It reflects source authority (curated,
# indexed scholarly literature), never the truth of any particular claim.
PEER_REVIEWED_AUTHORITY_SCORE = 8.5


class EuropePMCSource(BaseExternalSource):
    """Concrete external source for the Europe PMC Articles RESTful API.

    The API requires no key, so this provider has no credential handling.
    """

    def __init__(self, timeout: int = 30) -> None:
        self.name = "europe_pmc"
        self.timeout = timeout
        self.base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_string(record: Dict[str, Any], key: str) -> str:
        """Return a stripped string field, or an empty string when unusable."""
        value = record.get(key)
        if isinstance(value, str):
            return value.strip()
        return ""

    @staticmethod
    def _publication_date(record: Dict[str, Any]) -> Optional[datetime]:
        """Best-effort publication date from the Europe PMC date fields.

        Tries, in order: ``firstPublicationDate``, ``printPublicationDate``,
        ``electronicPublicationDate``, ``journalInfo.dateOfPublication`` and
        finally the bare ``pubYear``.  Returns ``None`` when nothing parses.
        """
        journal_info = record.get("journalInfo")
        if not isinstance(journal_info, dict):
            journal_info = {}

        for key in ("firstPublicationDate", "printPublicationDate",
                    "electronicPublicationDate", "dateOfCreation"):
            parsed = EuropePMCSource._parse_day(record.get(key))
            if parsed is not None:
                return parsed

        # "2026 Sep" style journal issue dates.
        issue_date = journal_info.get("dateOfPublication")
        if isinstance(issue_date, str) and issue_date.strip():
            for fmt in ("%Y %b", "%Y %B"):
                try:
                    return datetime.strptime(issue_date.strip(), fmt)
                except ValueError:
                    continue

        year = record.get("pubYear") or journal_info.get("yearOfPublication")
        if isinstance(year, int) or (isinstance(year, str) and year.strip().isdigit()):
            try:
                return datetime(int(str(year).strip()), 1, 1)
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _parse_day(value: Any) -> Optional[datetime]:
        """Parse an ISO ``YYYY-MM-DD`` (or ISO datetime) string."""
        if not isinstance(value, str) or not value.strip():
            return None
        text = value.strip()
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            return None

    @staticmethod
    def _evidence_id(record: Dict[str, Any], index: int) -> str:
        """Stable identifier derived from the record's persistent identifiers.

        Prefers PMID, then PMCID, then the Europe PMC ``id``/``source`` pair so
        the same article always yields the same ``evidence_id`` across calls.
        """
        for key in ("pmid", "pmcid"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return f"europe_pmc_{value.strip().lower()}"

        record_id = record.get("id")
        if isinstance(record_id, str) and record_id.strip():
            epmc_source = record.get("source")
            if isinstance(epmc_source, str) and epmc_source.strip():
                return f"europe_pmc_{epmc_source.strip().lower()}_{record_id.strip().lower()}"

        fallback = f"{record.get('title', '')}_{record.get('doi', '')}_{index}"
        return f"europe_pmc_{hashlib.sha256(fallback.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _journal(record: Dict[str, Any]) -> Dict[str, Any]:
        journal_info = record.get("journalInfo")
        if not isinstance(journal_info, dict):
            return {}
        journal = journal_info.get("journal")
        return journal if isinstance(journal, dict) else {}

    @staticmethod
    def _journal_title(record: Dict[str, Any]) -> str:
        journal = EuropePMCSource._journal(record)
        for key in ("isoabbreviation", "medlineAbbreviation", "title"):
            value = journal.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _publication_types(record: Dict[str, Any]) -> List[str]:
        pub_type_list = record.get("pubTypeList")
        if not isinstance(pub_type_list, dict):
            return []
        pub_types = pub_type_list.get("pubType")
        if not isinstance(pub_types, list):
            return []
        return [p.strip() for p in pub_types if isinstance(p, str) and p.strip()]

    @staticmethod
    def _url(record: Dict[str, Any]) -> str:
        """Resolve a stable landing page for the record.

        Prefers the Europe PMC full-text/landing URL, then the DOI resolver,
        and finally the canonical Europe PMC abstract URL.
        """
        full_text_list = record.get("fullTextUrlList")
        candidates: List[Dict[str, Any]] = []
        if isinstance(full_text_list, dict):
            urls = full_text_list.get("fullTextUrl")
            if isinstance(urls, list):
                candidates = [u for u in urls if isinstance(u, dict)]

        for entry in candidates:
            if entry.get("site") == "Europe_PMC" and entry.get("documentStyle") == "html":
                url = entry.get("url")
                if isinstance(url, str) and url.strip():
                    return url.strip()

        doi = record.get("doi")
        if isinstance(doi, str) and doi.strip():
            return f"https://doi.org/{doi.strip()}"

        record_id = record.get("id")
        epmc_source = record.get("source")
        if isinstance(record_id, str) and record_id.strip() and isinstance(epmc_source, str) and epmc_source.strip():
            return f"https://europepmc.org/article/{epmc_source.strip()}/{record_id.strip()}"

        return ""

    @staticmethod
    def _build_text(record: Dict[str, Any], title: str, journal: str,
                    pub_date: Optional[datetime]) -> str:
        """Return the abstract, or a factual bibliographic description.

        No medical interpretation is performed here: the fallback only restates
        fields the API already returned.
        """
        abstract = record.get("abstractText")
        if isinstance(abstract, str) and abstract.strip():
            return abstract.strip()

        parts = [part for part in (title, journal) if part]
        if pub_date is not None:
            parts.append(str(pub_date.year))
        authors = record.get("authorString")
        if isinstance(authors, str) and authors.strip():
            parts.append(authors.strip())

        if not parts:
            return "No abstract or bibliographic metadata available from Europe PMC."

        return "No abstract available. Bibliographic record: " + " — ".join(parts) + "."

    # ------------------------------------------------------------------
    # BaseExternalSource interface
    # ------------------------------------------------------------------

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

        params = {
            "query": query,
            "format": "json",
            "resultType": "core",
            "pageSize": top_k,
        }
        if language and language.strip():
            params["LANG"] = language.strip()

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
            raise RuntimeError(f"Europe PMC API request timed out: {exc}") from exc
        except requests.exceptions.HTTPError as exc:
            status_code = getattr(response, "status_code", None)
            if status_code == 429:
                raise RuntimeError("Europe PMC API rate limit exceeded") from exc
            raise RuntimeError(f"Europe PMC API request failed: {exc}") from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Europe PMC API request failed: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"Invalid JSON response from Europe PMC: {exc}") from exc

        if not isinstance(data, dict):
            raise RuntimeError("Unexpected response structure: JSON is not a dictionary")

        result_list = data.get("resultList")
        if result_list is None:
            return []
        if not isinstance(result_list, dict):
            raise RuntimeError("Unexpected response structure: 'resultList' is not a dictionary")

        results_field = result_list.get("result")
        if results_field is None:
            return []
        if not isinstance(results_field, list):
            raise RuntimeError("Unexpected response structure: 'resultList.result' is not a list")

        results: List[ExternalEvidence] = []
        for idx, record in enumerate(results_field[:top_k]):
            if not isinstance(record, dict):
                logger.warning("Skipping Europe PMC record %d: entry is not an object", idx)
                continue

            title = self._first_string(record, "title") or "Untitled Europe PMC record"
            journal_title = self._journal_title(record)
            pub_date = self._publication_date(record)
            url = self._url(record)
            text = self._build_text(record, title, journal_title, pub_date)

            epmc_source = self._first_string(record, "source")
            source = journal_title or "Europe PMC"

            authors = self._first_string(record, "authorString")
            cited_by = record.get("citedByCount")
            pub_types = self._publication_types(record)

            metadata: Dict[str, Any] = {
                "provider": "Europe PMC",
                "authority_score_basis": "Curated, indexed peer-reviewed literature (0-10 project scale).",
                "text_limitation": (
                    "Europe PMC returns abstracts and bibliographic metadata, not full article text."
                ),
            }
            for key, meta_key in (
                ("pmid", "pmid"),
                ("pmcid", "pmcid"),
                ("doi", "doi"),
                ("id", "epmc_id"),
            ):
                value = self._first_string(record, key)
                if value:
                    metadata[meta_key] = value
            if epmc_source:
                metadata["epmc_source"] = epmc_source
            if journal_title:
                metadata["journal"] = journal_title
            journal = self._journal(record)
            issn = journal.get("issn")
            if isinstance(issn, str) and issn.strip():
                metadata["issn"] = issn.strip()
            if authors:
                metadata["authors"] = authors
            if isinstance(cited_by, int):
                metadata["cited_by_count"] = cited_by
            if pub_types:
                metadata["publication_types"] = pub_types
            page_info = self._first_string(record, "pageInfo")
            if page_info:
                metadata["page_info"] = page_info
            is_open_access = record.get("isOpenAccess")
            if isinstance(is_open_access, str) and is_open_access.strip():
                metadata["is_open_access"] = is_open_access.strip()
            for key in ("inEPMC", "inPMC", "hasPDF", "hasTextMinedTerms"):
                value = record.get(key)
                if isinstance(value, str) and value.strip():
                    metadata[key] = value.strip()
            pub_year = record.get("pubYear")
            if isinstance(pub_year, (str, int)) and str(pub_year).strip():
                metadata["pub_year"] = str(pub_year).strip()

            ev = ExternalEvidence(
                evidence_id=self._evidence_id(record, idx),
                text=text,
                source=source,
                title=title,
                url=url,
                domain="medical",
                retrieved_date=datetime.utcnow(),
                metadata=metadata,
                publication_date=pub_date,
                language=self._first_string(record, "language") or None,
                source_type="medical_literature",
                authority_score=PEER_REVIEWED_AUTHORITY_SCORE,
            )
            results.append(ev)

        return results
    supported_domains = frozenset({"medical", "science"})
