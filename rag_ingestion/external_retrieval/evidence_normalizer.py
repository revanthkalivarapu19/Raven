import logging
import time
from typing import Optional
from urllib.parse import urlsplit

from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.external_retrieval.source_schema import ExternalEvidence
from rag_ingestion.external_retrieval.web_retriever import WebRetrievalResult
from rag_ingestion.evidence_quality import has_substantive_text

logger = logging.getLogger(__name__)

class ExternalEvidenceNormalizer:
    """Normalizes and quality filters external evidence into the internal Evidence schema."""

    def __init__(self, min_text_length: int = 50, min_fact_check_length: int = 20):
        self.min_text_length = min_text_length
        self.min_fact_check_length = min_fact_check_length

    def normalize(self, external_evidence: ExternalEvidence, web_result: Optional[WebRetrievalResult] = None) -> Evidence:
        """
        Normalize an ExternalEvidence object and optional WebRetrievalResult into a standard Evidence object.
        Raises ValueError if the evidence is invalid or does not meet quality standards.
        """
        text = external_evidence.text
        if web_result and web_result.text:
            text = web_result.text

        if not text or not text.strip():
            raise ValueError("Evidence text is empty or whitespace-only.")

        # Quality filter
        text = text.strip()
        min_length = self.min_fact_check_length if external_evidence.source_type == "fact_check_api" else self.min_text_length
        if len(text) < min_length:
            raise ValueError(f"Evidence text is too short (length {len(text)} < {min_length}).")

        full_page_extracted = bool(web_result and web_result.text and not web_result.error)
        metadata = external_evidence.metadata or {}
        if metadata.get("metadata_only") and not full_page_extracted:
            raise ValueError("Evidence is metadata-only and has no extracted source text.")
        quality_started = time.perf_counter()
        substantive = has_substantive_text(text)
        logger.info(
            "performance_diag phase=quality_filtering elapsed_ms=%.1f accepted=%s",
            (time.perf_counter() - quality_started) * 1000.0,
            substantive,
        )
        if not substantive:
            raise ValueError("Evidence contains no substantive source text.")

        if not external_evidence.source or not external_evidence.source.strip():
            raise ValueError("Evidence source is missing.")

        if external_evidence.url:
            if not external_evidence.url.startswith("http://") and not external_evidence.url.startswith("https://"):
                raise ValueError(f"Unsupported URL scheme: {external_evidence.url}")
        else:
            raise ValueError("Evidence URL is missing.")

        if not external_evidence.domain or not external_evidence.domain.strip():
            raise ValueError("Evidence domain is missing.")

        if not external_evidence.retrieved_date:
            raise ValueError("Evidence retrieved_date is missing.")

        # Map fields
        metadata = external_evidence.metadata.copy() if external_evidence.metadata else {}
        metadata["external_source_type"] = external_evidence.source_type
        metadata.setdefault("content_quality", "source_text")
        if web_result and web_result.text and not web_result.error:
            metadata["content_quality"] = "source_text"
            metadata["full_page_extracted"] = True
            metadata["full_page_url"] = web_result.final_url or external_evidence.url
        elif metadata.get("text_limitation") or metadata.get("metadata_only"):
            metadata["content_quality"] = "metadata_or_snippet"
        if external_evidence.url:
            hostname = (urlsplit(external_evidence.url).hostname or "").lower()
            if hostname.startswith("www."):
                hostname = hostname[4:]
            metadata.setdefault("source_family", hostname or external_evidence.source)
        if web_result and web_result.status_code:
            metadata["web_status_code"] = web_result.status_code
            if web_result.error:
                metadata["web_error"] = web_result.error
            if not web_result.text and web_result.error:
                metadata["web_retrieval_failed"] = True

        return Evidence(
            evidence_id=external_evidence.evidence_id,
            chunk_id=external_evidence.evidence_id,  # Map external evidence entirely as a single chunk
            document_id=external_evidence.evidence_id,
            text=text,
            domain=external_evidence.domain,
            source=external_evidence.source,
            language=external_evidence.language or "unknown",
            retrieved_date=external_evidence.retrieved_date,
            # External providers do not calculate BGE/FAISS similarity here.
            # Keep this explicitly missing rather than fabricating a maximum score.
            similarity_score=None,
            retrieval_method="external",
            metadata=metadata,
            url=external_evidence.url,
            title=external_evidence.title,
            publication_date=external_evidence.publication_date,
            authority_score=external_evidence.authority_score
        )
