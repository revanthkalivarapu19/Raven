import gc
import logging
from typing import List, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict

from rag_ingestion.retrieval.retrieval_manager import RetrievalManager
from rag_ingestion.external_retrieval.source_manager import ExternalSourceManager, ProviderError, ExternalSourceResult
from rag_ingestion.external_retrieval.evidence_normalizer import ExternalEvidenceNormalizer
from rag_ingestion.external_retrieval.fact_check_source import FactCheckSource
from rag_ingestion.external_retrieval.gdelt_source import GDELTSource
from rag_ingestion.external_retrieval.news_api_source import NewsAPISource
from rag_ingestion.external_retrieval.europe_pmc_source import EuropePMCSource
from rag_ingestion.external_retrieval.crossref_source import CrossrefSource
from rag_ingestion.external_retrieval.wikimedia_source import WikimediaSource
from rag_ingestion.external_retrieval.world_bank_source import WorldBankSource
from rag_ingestion.external_retrieval.ipu_parline_source import IPUParlineSource
from rag_ingestion.retrieval.evidence_fusion import EvidenceFusionManager
from rag_ingestion.external_retrieval.source_schema import ExternalEvidence
from rag_ingestion.external_retrieval.web_retriever import WebRetrievalResult, WebSourceRetriever
from rag_ingestion.verification.verification_manager import EvidenceVerificationManager
from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification.verification_schema import OverallVerification
from rag_ingestion.verification.verification_schema import VerificationFailure
from rag_ingestion.config.domains import canonical_domain
from rag_ingestion.config.config import config as _config  # Loads .env before providers access environment variables.

logger = logging.getLogger(__name__)

MAX_RETRIEVAL_QUERIES_PER_ATTEMPT = 4


def _canonical_url(url: str) -> Optional[str]:
    """Return a stable HTTP(S) URL key, or None when the URL is unusable."""
    try:
        parsed = urlsplit(url.strip())
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return None
        query = urlencode(
            [(key, value) for key, value in parse_qsl(parsed.query) if not key.lower().startswith("utm_")]
        )
        return urlunsplit(
            (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), query, "")
        )
    except (AttributeError, TypeError, ValueError):
        return None


def _meaningful(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    return value.strip().lower() not in {
        "no content available.",
        "no textual summary available.",
    }


def _api_evidence_is_sufficient(evidence: List[ExternalEvidence], minimum: int) -> bool:
    """Require distinct usable URLs and distinct source/publisher identities."""
    urls = set()
    sources = set()
    for item in evidence:
        url_key = _canonical_url(item.url)
        if (
            not _meaningful(item.text)
            or not _meaningful(item.source)
            or url_key is None
        ):
            continue

        publisher = item.metadata.get("publisher_site") if item.metadata else None
        source_identity = publisher if _meaningful(publisher) else item.source
        urls.add(url_key)
        sources.add(source_identity.strip().lower().removeprefix("www."))

    return len(urls) >= minimum and len(sources) >= minimum


def _fallback_candidates(evidence: List[ExternalEvidence]) -> List[tuple[str, ExternalEvidence]]:
    """Order valid API URLs by metadata quality and provider preference."""
    candidates = []
    for index, item in enumerate(evidence):
        url_key = _canonical_url(item.url)
        if url_key is None or not _meaningful(item.source):
            continue
        quality = sum((_meaningful(item.title), _meaningful(item.text), _meaningful(item.source)))
        provider_rank = {"gdelt": 0, "fact_check_api": 1}.get(item.source_type, 2)
        candidates.append(((-quality, provider_rank, index), url_key, item))

    candidates.sort(key=lambda candidate: candidate[0])
    seen_urls = set()
    unique = []
    for _, url_key, item in candidates:
        if url_key not in seen_urls:
            seen_urls.add(url_key)
            unique.append((url_key, item))
    return unique


def _web_result_for_url(
    url: Optional[str], web_results: dict[str, WebRetrievalResult]
) -> Optional[WebRetrievalResult]:
    url_key = _canonical_url(url) if isinstance(url, str) else None
    return web_results.get(url_key) if url_key is not None else None


class ClaimProcessingResult(BaseModel):
    """Result of the end‑to‑end claim processing pipeline (Phase 5G)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    claim: str
    domain: str
    local_evidence: List[Evidence]
    external_evidence: List[Evidence]
    external_source_errors: List[ProviderError]
    fused_evidence: List[Evidence]
    verification_result: Optional[OverallVerification] = None
    verification_failure: Optional[VerificationFailure] = None
    local_retrieval_error: Optional[ProviderError] = None


class ClaimProcessingPipeline:
    """Orchestrates the full retrieval → verification flow.

    The component is deliberately lightweight – it merely wires together the
    existing building‑block managers that were implemented in Phases 5A‑5F.
    No business logic is duplicated; each manager is called with the values
    supplied by the caller.
    """

    def __init__(
        self,
        retrieval_manager: RetrievalManager | None = None,
        external_manager: ExternalSourceManager | None = None,
        normalizer: ExternalEvidenceNormalizer | None = None,
        fusion_manager: EvidenceFusionManager | None = None,
        verification_manager: EvidenceVerificationManager | None = None,
        web_retriever: WebSourceRetriever | None = None,
        minimum_sufficient_external_evidence: int = 2,
        max_web_fallback_urls: int = 3,
    ) -> None:
        self.retrieval_manager = retrieval_manager or RetrievalManager()
        if minimum_sufficient_external_evidence < 1:
            raise ValueError("minimum_sufficient_external_evidence must be positive")
        if max_web_fallback_urls < 0:
            raise ValueError("max_web_fallback_urls cannot be negative")
        self.external_manager = external_manager or self._default_external_manager()
        self.normalizer = normalizer or ExternalEvidenceNormalizer()
        self.fusion_manager = fusion_manager or EvidenceFusionManager()
        self.verification_manager = verification_manager or EvidenceVerificationManager()
        self.web_retriever = web_retriever or WebSourceRetriever()
        self.minimum_sufficient_external_evidence = minimum_sufficient_external_evidence
        self.max_web_fallback_urls = max_web_fallback_urls

    @staticmethod
    def _default_external_manager() -> ExternalSourceManager:
        manager = ExternalSourceManager()
        manager.register(GDELTSource())
        manager.register(NewsAPISource())
        manager.register(FactCheckSource())
        manager.register(EuropePMCSource())
        manager.register(CrossrefSource())
        manager.register(WikimediaSource())
        manager.register(WorldBankSource())
        manager.register(IPUParlineSource())
        return manager

    def _release_retrieval_resources(self) -> None:
        """Release embedding model state before handing off to Ollama."""
        embedders = []
        embedder = getattr(self.retrieval_manager, "embedder", None)
        if embedder is not None:
            embedders.append(embedder)
        query_service = getattr(self.retrieval_manager, "query_service", None)
        query_embedder = getattr(query_service, "embedder", None)
        if query_embedder is not None:
            embedders.append(query_embedder)

        seen = set()
        for candidate in embedders:
            if id(candidate) in seen:
                continue
            seen.add(id(candidate))
            if hasattr(candidate, "_model"):
                candidate._model = None
            if hasattr(candidate, "_dimension"):
                candidate._dimension = None
        gc.collect()

    def _retrieve_query(
        self,
        query: str,
        domain: str,
        top_k_local: int,
        top_k_external: int,
        fallback_urls_seen: set[str] | None = None,
        fallback_urls_remaining: list[int] | None = None,
    ) -> tuple[List[Evidence], List[Evidence], Optional[ProviderError], List[ProviderError]]:
        """Retrieve and normalize one query without fusion or verification."""
        local_error = None
        try:
            local_evidence = self.retrieval_manager.search(
                claim=query, domain=domain, top_k=top_k_local
            )
        except FileNotFoundError as exc:
            logger.warning("Local retrieval unavailable for domain %s: %s", domain, exc)
            local_evidence = []
            local_error = ProviderError(provider_name="local_retrieval", exception=exc)

        external_result = self.external_manager.search(
            query=query, domain=domain, top_k=top_k_external
        )
        web_results: dict[str, WebRetrievalResult] = {}
        fallback_urls_seen = fallback_urls_seen if fallback_urls_seen is not None else set()
        fallback_urls_remaining = (
            fallback_urls_remaining
            if fallback_urls_remaining is not None
            else [self.max_web_fallback_urls]
        )
        if not _api_evidence_is_sufficient(
            external_result.evidence, self.minimum_sufficient_external_evidence
        ):
            candidates = []
            for url_key, api_evidence in _fallback_candidates(external_result.evidence):
                if url_key not in fallback_urls_seen:
                    candidates.append((url_key, api_evidence))
                if len(candidates) >= fallback_urls_remaining[0]:
                    break
            for url_key, api_evidence in candidates:
                fallback_urls_seen.add(url_key)
                fallback_urls_remaining[0] -= 1
                try:
                    web_result = self.web_retriever.retrieve(api_evidence.url)
                    if web_result.error or not _meaningful(web_result.text):
                        raise RuntimeError(web_result.error or "Web retrieval returned no usable text")
                    web_results[url_key] = web_result
                except Exception as exc:
                    external_result.errors.append(
                        ProviderError(provider_name="web_retriever", exception=exc)
                    )

        normalized = []
        for item in external_result.evidence:
            try:
                web_result = _web_result_for_url(item.url, web_results)
                normalized.append(
                    self.normalizer.normalize(item, web_result=web_result)
                    if web_result is not None else self.normalizer.normalize(item)
                )
            except Exception as exc:
                external_result.errors.append(
                    ProviderError(provider_name="normalizer", exception=exc)
                )
        return local_evidence, normalized, local_error, list(external_result.errors)

    def _process_multiple_queries(
        self, claim: str, domain: str, queries: List[str],
        top_k_local: int, top_k_external: int, top_k_fused: int,
    ) -> ClaimProcessingResult:
        local_by_id = {}
        external_by_id = {}
        local_error = None
        external_errors = []
        normalized_queries = list(dict.fromkeys(
            query.strip() for query in queries if query and query.strip()
        ))[:MAX_RETRIEVAL_QUERIES_PER_ATTEMPT]
        fallback_urls_seen = set()
        fallback_urls_remaining = [self.max_web_fallback_urls]
        for query in normalized_queries:
            local_items, external_items, query_local_error, query_errors = self._retrieve_query(
                query, domain, top_k_local, top_k_external,
                fallback_urls_seen, fallback_urls_remaining,
            )
            for item in local_items:
                local_by_id[item.evidence_id] = item
            for item in external_items:
                external_by_id[item.evidence_id] = item
            if local_error is None and query_local_error is not None:
                local_error = query_local_error
            external_errors.extend(query_errors)

        local_evidence = list(local_by_id.values())
        external_evidence = list(external_by_id.values())
        fused_evidence = self.fusion_manager.fuse(
            local_evidence=local_evidence,
            external_evidence=external_evidence,
            top_k=top_k_fused,
        )
        verification_result = None
        verification_failure = None
        self._release_retrieval_resources()
        try:
            verification_result = self.verification_manager.verify(
                claim=claim, evidence_list=fused_evidence
            )
        except Exception as exc:
            verification_failure = VerificationFailure.from_exception(
                claim=claim, evidence_count=len(fused_evidence), exc=exc
            )
        return ClaimProcessingResult(
            claim=claim,
            domain=domain,
            local_evidence=local_evidence,
            local_retrieval_error=local_error,
            external_evidence=external_evidence,
            external_source_errors=external_errors,
            fused_evidence=fused_evidence,
            verification_result=verification_result,
            verification_failure=verification_failure,
        )

    def process(
        self,
        claim: str,
        domain: str,
        top_k_local: int = 5,
        top_k_external: int = 5,
        top_k_fused: int = 5,
        retrieval_query: str | None = None,
        retrieval_queries: List[str] | None = None,
    ) -> ClaimProcessingResult:
        """Execute the full pipeline and return a structured result.

        Steps:
        1. Local retrieval via :class:`RetrievalManager`.
        2. External retrieval via :class:`ExternalSourceManager`.
        3. Normalise each :class:`ExternalEvidence` to internal :class:`Evidence`.
        4. Fuse local and normalised external evidence.
        5. Verify the fused evidence.
        """
        domain = canonical_domain(domain)
        if retrieval_queries is not None:
            return self._process_multiple_queries(
                claim=claim,
                domain=domain,
                queries=retrieval_queries,
                top_k_local=top_k_local,
                top_k_external=top_k_external,
                top_k_fused=top_k_fused,
            )
        retrieval_query = (retrieval_query or claim).strip()

        logger.info(
            "Evidence retrieval query: %s",
            retrieval_query,
        )

        # 1. Local retrieval
        local_retrieval_error: Optional[ProviderError] = None
        try:
            local_evidence: List[Evidence] = self.retrieval_manager.search(
                claim=retrieval_query, domain=domain, top_k=top_k_local
            )
        except FileNotFoundError as exc:
            # A missing domain index means local retrieval is unavailable; it
            # is not an empty result and must not prevent external retrieval.
            logger.warning("Local retrieval unavailable for domain %s: %s", domain, exc)
            local_evidence = []
            local_retrieval_error = ProviderError(
                provider_name="local_retrieval",
                exception=exc,
            )
        logger.debug("Local retrieval returned %d items", len(local_evidence))

        # 2. External retrieval
        external_result: ExternalSourceResult = self.external_manager.search(
            query=retrieval_query, domain=domain, top_k=top_k_external
        )
        logger.debug(
            "External retrieval returned %d items with %d errors",
            len(external_result.evidence),
            len(external_result.errors),
        )

        # 3. If API evidence is insufficient, fetch a bounded set of API-provided URLs.
        web_results: dict[str, WebRetrievalResult] = {}
        if not _api_evidence_is_sufficient(
            external_result.evidence,
            self.minimum_sufficient_external_evidence,
        ):
            candidates = _fallback_candidates(external_result.evidence)
            for url_key, api_evidence in candidates[: self.max_web_fallback_urls]:
                try:
                    web_result = self.web_retriever.retrieve(api_evidence.url)
                except Exception as exc:
                    logger.warning("Web fallback failed for %s: %s", api_evidence.url, exc)
                    external_result.errors.append(
                        ProviderError(provider_name="web_retriever", exception=exc)
                    )
                    continue

                if web_result.error or not _meaningful(web_result.text):
                    error = RuntimeError(
                        web_result.error or "Web retrieval returned no usable text"
                    )
                    logger.warning("Web fallback failed for %s: %s", api_evidence.url, error)
                    external_result.errors.append(
                        ProviderError(provider_name="web_retriever", exception=error)
                    )
                    continue
                web_results[url_key] = web_result

        # 4. Normalisation – successful page text enriches its originating API evidence.
        normalized_external: List[Evidence] = []
        for ev in external_result.evidence:
            try:
                web_result = _web_result_for_url(ev.url, web_results)
                if web_result is None:
                    norm_ev = self.normalizer.normalize(ev)
                else:
                    norm_ev = self.normalizer.normalize(ev, web_result=web_result)
                normalized_external.append(norm_ev)
            except Exception as exc:
                # Normalisation errors are treated as provider‑level failures.
                logger.warning(
                    "Normalization failed for external evidence %s: %s", ev.evidence_id, exc
                )
                external_result.errors.append(
                    ProviderError(provider_name="normalizer", exception=exc)
                )

        # 5. Fusion
        fused_evidence: List[Evidence] = self.fusion_manager.fuse(
            local_evidence=local_evidence,
            external_evidence=normalized_external,
            top_k=top_k_fused,
        )
        logger.debug("Fusion produced %d items", len(fused_evidence))

        # 6. Verification – only the fused evidence are examined
        verification_result: Optional[OverallVerification] = None
        verification_failure: Optional[VerificationFailure] = None
        self._release_retrieval_resources()
        try:
            verification_result = self.verification_manager.verify(
                claim=claim, evidence_list=fused_evidence
            )
        except Exception as exc:
            logger.error("Verification failed: %s", exc)
            verification_failure = VerificationFailure.from_exception(
                claim=claim, evidence_count=len(fused_evidence), exc=exc
            )

        return ClaimProcessingResult(
            claim=claim,
            domain=domain,
            local_evidence=local_evidence,
            local_retrieval_error=local_retrieval_error,
            external_evidence=normalized_external,
            external_source_errors=[
                err if isinstance(err, ProviderError) else ProviderError(
                    provider_name=getattr(err, "provider_name", "unknown"),
                    exception=getattr(err, "exception", Exception(str(err)))
                )
                for err in external_result.errors
            ],
            fused_evidence=fused_evidence,
            verification_result=verification_result,
            verification_failure=verification_failure,
        )
