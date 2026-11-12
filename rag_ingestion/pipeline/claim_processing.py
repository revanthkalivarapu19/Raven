import gc
import logging
import os
import time
from typing import List, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field

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
from rag_ingestion.verification.claim_scope import ClaimScope
from rag_ingestion.verification.directness_gate import evaluate_directness
from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification.verification_schema import OverallVerification
from rag_ingestion.verification.verification_schema import VerificationFailure
from rag_ingestion.config.domains import canonical_domain
from rag_ingestion.config.config import config as _config  # Loads .env before providers access environment variables.
from rag_ingestion.utils.temporal_context import build_temporal_context
from rag_ingestion.llm_resilience import (
    VERIFICATION_BUDGET_RESERVE_SECONDS,
    remaining_request_budget,
)

logger = logging.getLogger(__name__)

MAX_RETRIEVAL_QUERIES_PER_ATTEMPT = 3
MAX_EVIDENCE_ITEMS_TO_VERIFY = 2
MIN_RETRIEVAL_BUDGET_SECONDS = 1.0
# Limit each claim to one full-page upgrade; provider snippets remain available
# when a source host is slow or inaccessible.
MAX_SNIPPET_UPGRADES_PER_ATTEMPT = 1


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


def _evidence_drop_reason(exc: Exception) -> str:
    """Classify an evidence rejection without logging potentially sensitive text."""
    message = str(exc).casefold()
    if "empty" in message or "whitespace" in message:
        return "empty_content"
    if "unsupported" in message and "url" in message:
        return "unsupported_source"
    if "url" in message and "missing" in message:
        return "missing_url"
    if "too short" in message or "quality" in message:
        return "low_quality"
    if "historical" in message or "timeframe" in message or "temporal" in message:
        return "temporal_mismatch"
    if "unsupported" in message or "source" in message:
        return "unsupported_source"
    if (
        "schema" in message
        or "validation" in message
        or "invalid" in message
        or type(exc).__name__ == "ValidationError"
    ):
        return "invalid_schema"
    if "deadline" in message or "timeout" in message:
        return "deadline_guard"
    return "other"


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
        discovery_only = bool((item.metadata or {}).get("discovery_only"))
        provider_rank = {"gdelt": 0, "fact_check_api": 1}.get(item.source_type, 2)
        candidates.append(((0 if discovery_only else 1, -quality, provider_rank, index), url_key, item))

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
    provider_statuses: dict[str, str] = Field(default_factory=dict)
    provider_diagnostics: dict[str, dict[str, object]] = Field(default_factory=dict)
    retrieval_diagnostics: dict[str, object] = Field(default_factory=dict)

    @property
    def external_provider_state(self) -> str:
        applicable = [
            status for status in self.provider_statuses.values()
            if not status.startswith("skipped_")
        ]
        if not applicable:
            return "not_applicable"
        succeeded = any(status in {"success", "zero_results", "partial_failure"} for status in applicable)
        failed = any(status in {"unavailable", "timed_out", "rate_limited", "request_failed", "partial_failure"} for status in applicable)
        if succeeded and failed:
            return "partial"
        if failed:
            return "provider_unavailable"
        return "results_available" if self.external_evidence else "zero_results"


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
        max_web_fallback_urls: int = 1,
    ) -> None:
        self.retrieval_manager = retrieval_manager or RetrievalManager()
        if minimum_sufficient_external_evidence < 1:
            raise ValueError("minimum_sufficient_external_evidence must be positive")
        if max_web_fallback_urls < 0:
            raise ValueError("max_web_fallback_urls cannot be negative")
        self.external_manager = external_manager or self._default_external_manager()
        self.normalizer = normalizer or ExternalEvidenceNormalizer()
        self.fusion_manager = fusion_manager or EvidenceFusionManager()
        self.semantic_embedder_unavailable_reason: str | None = None
        if isinstance(self.fusion_manager, EvidenceFusionManager):
            self.fusion_manager.semantic_embedder = getattr(self.retrieval_manager, "embedder", None)
        self.verification_manager = verification_manager or EvidenceVerificationManager()
        # Page upgrades are best-effort: cap each fetch so a slow source page
        # cannot use the budget reserved for semantic ranking and verification.
        self.web_retriever = web_retriever or WebSourceRetriever(timeout=4)
        self.minimum_sufficient_external_evidence = minimum_sufficient_external_evidence
        self.max_web_fallback_urls = max_web_fallback_urls

    def mark_semantic_embedder_unavailable(self, error: BaseException) -> None:
        """Disable only BGE-backed work after a confirmed worker initialization failure."""
        self.semantic_embedder_unavailable_reason = f"{type(error).__name__}: {error}"
        if isinstance(self.fusion_manager, EvidenceFusionManager):
            self.fusion_manager.semantic_embedder = None
        logger.error(
            "semantic_ranking unavailable=true fallback=generic_retrieval_ranker reason=%r",
            self.semantic_embedder_unavailable_reason,
        )

    @staticmethod
    def _default_external_manager() -> ExternalSourceManager:
        manager = ExternalSourceManager()
        news_enabled = bool(os.getenv("NEWS_API_KEY", "").strip())
        gdelt_enabled = os.getenv("ENABLE_GDELT", "false").strip().lower() in {
            "1", "true", "yes", "on"
        }
        fact_check_enabled = bool(os.getenv("GOOGLE_FACTCHECK_API_KEY", "").strip())
        manager.provider_configuration = {
            "news_api": news_enabled,
            "gdelt": gdelt_enabled,
            "fact_check": fact_check_enabled,
            "europe_pmc": True,
            "crossref": True,
            "wikimedia": True,
            "world_bank": True,
            "ipu_parline": True,
        }
        # A generic encyclopedia lookup is the fastest broad factual source;
        # run it before slower news and discovery providers.
        manager.register(WikimediaSource())
        if news_enabled:
            manager.register(NewsAPISource(timeout=4))
        else:
            logger.info("Provider news_api disabled because API key is unavailable")

        if fact_check_enabled:
            manager.register(FactCheckSource(timeout=4))
        else:
            logger.info("Provider fact_check disabled because API key is unavailable")

        if gdelt_enabled:
            # Optional final news fallback: make one short attempt only after
            # NewsAPI and supplementary fact-check retrieval.
            manager.register(GDELTSource(timeout=1, max_retries=0))
        else:
            logger.info("Provider gdelt disabled by configuration")

        manager.register(EuropePMCSource())
        manager.register(CrossrefSource())
        manager.register(WorldBankSource())
        manager.register(IPUParlineSource())
        return manager

    def _release_retrieval_resources(self) -> None:
        """Retain the process-level embedding model for subsequent attempts."""
        return None

    def _upgrade_metadata_snippets(
        self,
        claim: str,
        evidence_items: List[Evidence],
        max_fetches: int = MAX_SNIPPET_UPGRADES_PER_ATTEMPT,
        excluded_urls: Optional[set[str]] = None,
        page_fetch_counter: list[int] | None = None,
    ) -> List[Evidence]:
        """Fetch full page text for a bounded set of top-ranked external snippets."""
        max_fetches = min(max_fetches, MAX_SNIPPET_UPGRADES_PER_ATTEMPT)
        if max_fetches <= 0:
            return evidence_items
        candidates = [
            item
            for item in evidence_items
            if item.retrieval_method == "external"
            and (item.metadata or {}).get("content_quality") == "metadata_or_snippet"
        ]
        if not candidates:
            return evidence_items

        if isinstance(self.fusion_manager, EvidenceFusionManager):
            ranked = self.fusion_manager.fuse(
                local_evidence=[],
                external_evidence=candidates,
                top_k=len(candidates),
                claim=claim,
                # Snippets are candidates for source-text upgrade; filtering
                # them here could prevent fetching the text needed to assess relevance.
                apply_minimum_relevance=False,
            )
        else:
            ranked = candidates

        selected: dict[str, tuple[str, Evidence]] = {}
        for item in ranked:
            canonical_url = _canonical_url(item.url) if item.url else None
            if canonical_url is None:
                logger.info(
                    "evidence_diag page_upgrade_skip evidence_id=%s reason=missing_url",
                    item.evidence_id,
                )
                continue
            if excluded_urls and canonical_url in excluded_urls:
                logger.info(
                    "evidence_diag page_upgrade_skip evidence_id=%s "
                    "reason=previous_fetch_attempt",
                    item.evidence_id,
                )
                continue
            selected.setdefault(canonical_url, (canonical_url, item))
            if len(selected) >= max_fetches:
                break

        upgraded: dict[str, Evidence] = {}
        for canonical_url, item in selected.values():
            remaining = remaining_request_budget()
            if remaining is not None and remaining <= VERIFICATION_BUDGET_RESERVE_SECONDS:
                logger.info(
                    "evidence_diag page_upgrade_skip evidence_id=%s reason=verification_budget_reserve remaining_seconds=%.2f reserve_seconds=%.2f",
                    item.evidence_id, remaining, VERIFICATION_BUDGET_RESERVE_SECONDS,
                )
                break

            try:
                if page_fetch_counter is not None:
                    page_fetch_counter[0] += 1
                result = self.web_retriever.retrieve(canonical_url)
            except Exception as exc:
                logger.info(
                    "evidence_diag page_upgrade_fallback evidence_id=%s "
                    "reason=fetch_exception exception_type=%s",
                    item.evidence_id,
                    type(exc).__name__,
                )
                continue
            text = (result.text or "").strip()
            minimum_length = max(50, getattr(self.normalizer, "min_text_length", 50))
            if (
                result.error
                or not _meaningful(text)
                or len(text) < minimum_length
            ):
                logger.info(
                    "evidence_diag page_upgrade_fallback evidence_id=%s "
                    "reason=%s",
                    item.evidence_id,
                    "fetch_failed" if result.error else "empty_or_short_content",
                )
                continue

            metadata = dict(item.metadata or {})
            metadata["content_quality"] = "source_text"
            metadata["full_page_extracted"] = True
            metadata["full_page_url"] = result.final_url or canonical_url
            upgraded[item.evidence_id] = item.model_copy(
                update={
                    "text": text,
                    "url": canonical_url,
                    "metadata": metadata,
                }
            )
            logger.info(
                "evidence_diag page_upgrade_success evidence_id=%s text_chars=%d",
                item.evidence_id,
                len(text),
            )

        return [upgraded.get(item.evidence_id, item) for item in evidence_items]

    def _retrieve_query(
        self,
        claim: str,
        query: str,
        domain: str,
        top_k_local: int,
        top_k_external: int,
        fallback_urls_seen: set[str] | None = None,
        fallback_urls_remaining: list[int] | None = None,
        *,
        include_external: bool = True,
        local_evidence_override: List[Evidence] | None = None,
        stop_when_external=None,
        page_fetch_counter: list[int] | None = None,
        stage_durations: dict[str, float] | None = None,
    ) -> tuple[List[Evidence], List[Evidence], Optional[ProviderError], List[ProviderError], dict[str, str], dict[str, dict[str, object]]]:
        """Retrieve and normalize one query without fusion or verification."""
        logger.info("evidence_diag query=%r", query)
        local_error = None
        if local_evidence_override is not None:
            local_evidence = list(local_evidence_override)
        elif getattr(self, "semantic_embedder_unavailable_reason", None):
            local_evidence = []
            local_error = ProviderError(
                provider_name="local_retrieval",
                exception=RuntimeError(getattr(self, "semantic_embedder_unavailable_reason", "BGE embedder unavailable")),
            )
            logger.warning(
                "evidence_diag retrieval_skipped stage=local reason=bge_unavailable "
                "fallback=external_retrieval error=%r",
                getattr(self, "semantic_embedder_unavailable_reason", "BGE embedder unavailable"),
            )
        else:
            local_started = time.perf_counter()
            try:
                local_evidence = self.retrieval_manager.search(
                    claim=query, domain=domain, top_k=top_k_local
                )
            except FileNotFoundError as exc:
                logger.warning("Local retrieval unavailable for domain %s: %s", domain, exc)
                local_evidence = []
                local_error = ProviderError(provider_name="local_retrieval", exception=exc)
                logger.info(
                    "evidence_diag retrieval_exception stage=local provider=local_retrieval "
                    "reason=other exception_type=%s",
                    type(exc).__name__,
                )
            logger.info(
                "performance_diag phase=local_faiss_retrieval elapsed_seconds=%.4f "
                "query=%r evidence_count=%d",
                time.perf_counter() - local_started, query, len(local_evidence),
            )

        if not include_external:
            logger.info("evidence_diag external_deferred reason=faiss_first query=%r", query)
            return local_evidence, [], local_error, [], {}, {}

        # Unknown is a routing result, not a provider domain. Let providers
        # apply their own supported-domain filters while preserving Unknown in
        # Raven's session and fused evidence metadata.
        provider_domain = None if str(domain).strip().casefold() == "unknown" else domain
        external_kwargs = {"query": query, "domain": provider_domain, "top_k": top_k_external}
        if isinstance(self.external_manager, ExternalSourceManager):
            external_kwargs["temporal_context"] = build_temporal_context(claim, [])
            if stop_when_external is not None:
                external_kwargs["stop_when"] = stop_when_external
        external_started = time.perf_counter()
        try:
            external_result = self.external_manager.search(**external_kwargs)
        except Exception as exc:
            logger.info(
                "evidence_diag retrieval_exception stage=external "
                "reason=%s exception_type=%s",
                _evidence_drop_reason(exc),
                type(exc).__name__,
            )
            raise
        logger.info(
            "performance_diag phase=external_provider_retrieval elapsed_seconds=%.4f "
            "query=%r provider_count=%d evidence_count=%d",
            time.perf_counter() - external_started, query,
            len(getattr(self.external_manager, "providers", [])),
            len(getattr(external_result, "evidence", [])),
        )
        if stage_durations is not None:
            stage_durations["external_provider_retrieval"] = stage_durations.get("external_provider_retrieval", 0.0) + (time.perf_counter() - external_started)
        external_raw = getattr(
            external_result, "raw_result_count", len(external_result.evidence)
        )
        provider_counts = getattr(external_result, "provider_counts", {})
        logger.info(
            "evidence_diag raw_counts query=%r local=%d external=%d "
            "provider_counts=%s provider_deduplicated=%d",
            query,
            len(local_evidence),
            external_raw,
            provider_counts,
            getattr(external_result, "duplicate_count", 0),
        )
        for error in external_result.errors:
            logger.info(
                "evidence_diag provider_error provider=%s category=%s "
                "exception_type=%s",
                getattr(error, "provider_name", "unknown"),
                getattr(error, "category", "other_failure"),
                type(getattr(error, "exception", error)).__name__,
            )
        web_results: dict[str, WebRetrievalResult] = {}
        fallback_urls_seen = fallback_urls_seen if fallback_urls_seen is not None else set()
        fallback_urls_remaining = (
            fallback_urls_remaining
            if fallback_urls_remaining is not None
            else [self.max_web_fallback_urls]
        )
        has_discovery_candidates = any(
            (item.metadata or {}).get("discovery_only")
            for item in external_result.evidence
        )
        if has_discovery_candidates:
            logger.info("evidence_diag page_upgrade_deferred reason=rank_before_fetch")

        normalized = []
        normalization_drops = 0
        normalization_started = time.perf_counter()
        for item in external_result.evidence:
            try:
                web_result = _web_result_for_url(item.url, web_results)
                normalized.append(
                    self.normalizer.normalize(item, web_result=web_result)
                    if web_result is not None else self.normalizer.normalize(item)
                )
            except Exception as exc:
                normalization_drops += 1
                logger.info(
                    "evidence_diag drop stage=normalization evidence_id=%s "
                    "provider=%s reason=%s exception_type=%s",
                    item.evidence_id,
                    item.source_type or item.source,
                    _evidence_drop_reason(exc),
                    type(exc).__name__,
                )
                external_result.errors.append(
                    ProviderError(provider_name="normalizer", exception=exc)
                )
        logger.info(
            "performance_diag phase=external_normalization elapsed_seconds=%.4f "
            "input_count=%d output_count=%d",
            time.perf_counter() - normalization_started,
            len(external_result.evidence), len(normalized),
        )
        if stage_durations is not None:
            stage_durations["normalization"] = stage_durations.get("normalization", 0.0) + (time.perf_counter() - normalization_started)
        logger.info(
            "evidence_diag normalized query=%r raw=%d manager_deduplicated=%d "
            "normalized=%d dropped=%d",
            query,
            external_raw,
            len(external_result.evidence),
            len(normalized),
            normalization_drops,
        )
        return (
            local_evidence, normalized, local_error, list(external_result.errors),
            dict(getattr(external_result, "provider_statuses", {})),
            dict(getattr(external_result, "provider_diagnostics", {})),
        )

    def _process_multiple_queries(
        self, claim: str, domain: str, queries: List[str],
        top_k_local: int, top_k_external: int, top_k_fused: int,
        claim_scope: ClaimScope | None = None,
        claim_id: str = "C1",
    ) -> ClaimProcessingResult:
        normalized_queries = list(dict.fromkeys(
            query.strip() for query in queries if query and query.strip()
        ))[:MAX_RETRIEVAL_QUERIES_PER_ATTEMPT]
        fallback_urls_seen = set()
        fallback_urls_remaining = [
            min(self.max_web_fallback_urls, MAX_SNIPPET_UPGRADES_PER_ATTEMPT)
        ]
        local_by_id: dict[str, Evidence] = {}
        local_by_query: dict[str, list[Evidence]] = {}
        stage_durations = {"external_provider_retrieval": 0.0, "normalization": 0.0, "page_upgrade_fetch": 0.0}
        local_error = None
        page_fetch_calls = [0]

        # Collect local evidence first, but defer semantic verification until
        # retrieval is complete so every atomic claim gets one bounded pass.
        local_retrieval_started = time.perf_counter()
        for query in normalized_queries:
            remaining = remaining_request_budget()
            if remaining is not None and remaining <= MIN_RETRIEVAL_BUDGET_SECONDS:
                break
            local_items, _, query_local_error, _, _, _ = self._retrieve_query(
                claim, query, domain, top_k_local, top_k_external,
                fallback_urls_seen, fallback_urls_remaining,
                include_external=False,
                page_fetch_counter=page_fetch_calls,
                stage_durations=stage_durations,
            )
            local_by_query[query] = local_items
            for item in local_items:
                local_by_id[item.evidence_id] = item
            if local_error is None and query_local_error is not None:
                local_error = query_local_error
        logger.info(
            "performance_diag phase=local_faiss_total elapsed_seconds=%.4f "
            "query_count=%d unique_evidence=%d",
            time.perf_counter() - local_retrieval_started,
            len(local_by_query), len(local_by_id),
        )

        local_retrieval_seconds = time.perf_counter() - local_retrieval_started
        local_evidence = list(local_by_id.values())
        fusion_started = time.perf_counter()
        fusion_kwargs = {"local_evidence": local_evidence, "external_evidence": [], "top_k": top_k_fused}
        if isinstance(self.fusion_manager, EvidenceFusionManager):
            fusion_kwargs.update(claim=claim, domain=domain)
        local_fused = self.fusion_manager.fuse(**fusion_kwargs)
        logger.info(
            "performance_diag phase=evidence_fusion_local elapsed_seconds=%.4f "
            "input_count=%d output_count=%d",
            time.perf_counter() - fusion_started,
            len(local_evidence), len(local_fused),
        )
        logger.info(
            "evidence_diag external_retrieval reason=single_verification_pass "
            "local_candidates=%d local_ranked=%d",
            len(local_evidence), len(local_fused),
        )
        begin_attempt = getattr(self.external_manager, "begin_attempt", None)
        if callable(begin_attempt):
            begin_attempt()
        external_by_id: dict[str, Evidence] = {}
        external_errors: list[ProviderError] = []
        external_retrieval_started = time.perf_counter()
        provider_statuses: dict[str, str] = {}
        provider_diagnostics: dict[str, dict[str, object]] = {}
        query_evidence_ids = {q: {item.evidence_id for item in local_by_query.get(q, [])} for q in normalized_queries}

        def local_evidence_is_sufficient() -> bool:
            direct_count = 0
            for item in local_fused:
                if evaluate_directness(claim, item.text, "SUPPORTS").accepted:
                    direct_count += 1
            return direct_count >= self.minimum_sufficient_external_evidence

        def stop_external_after_fused_evidence(candidates):
            if not _api_evidence_is_sufficient(
                candidates, self.minimum_sufficient_external_evidence
            ):
                return False
            normalized_candidates = []
            for candidate in candidates:
                try:
                    normalized_candidates.append(self.normalizer.normalize(candidate))
                except Exception:
                    continue
            fusion_kwargs = {
                "local_evidence": local_evidence,
                "external_evidence": normalized_candidates,
                "top_k": top_k_fused,
            }
            if isinstance(self.fusion_manager, EvidenceFusionManager):
                fusion_kwargs.update(claim=claim, domain=domain)
            accepted = self.fusion_manager.fuse(**fusion_kwargs)
            accepted_external = [
                item for item in accepted if item.retrieval_method == "external"
            ]
            direct_count = sum(
                evaluate_directness(claim, item.text, "SUPPORTS").accepted
                for item in accepted_external
            )
            return direct_count >= self.minimum_sufficient_external_evidence

        external_queries = [] if local_evidence_is_sufficient() else normalized_queries
        if not external_queries:
            logger.info(
                "evidence_diag external_retrieval_skip reason=local_evidence_sufficient "
                "local_ranked=%d minimum=%d",
                len(local_fused), self.minimum_sufficient_external_evidence,
            )
        logger.info(
            "performance_diag phase=external_query_plan query_count=%d",
            len(external_queries),
        )
        for query_index, query in enumerate(external_queries, start=1):
            remaining = remaining_request_budget()
            if remaining is not None and remaining <= VERIFICATION_BUDGET_RESERVE_SECONDS:
                external_errors.append(ProviderError(
                    provider_name="retrieval_budget",
                    exception=RuntimeError("verification budget reserve reached; remaining queries skipped"),
                    category="budget_reserved",
                ))
                logger.info(
                    "evidence_diag external_query_skip reason=verification_budget_reserve remaining_seconds=%.2f reserve_seconds=%.2f",
                    remaining, VERIFICATION_BUDGET_RESERVE_SECONDS,
                )
                break
            _, external_items, _, query_errors, query_statuses, query_diagnostics = self._retrieve_query(
                claim, query, domain, top_k_local, top_k_external,
                fallback_urls_seen, fallback_urls_remaining,
                local_evidence_override=local_by_query.get(query, []),
                stop_when_external=stop_external_after_fused_evidence,
                page_fetch_counter=page_fetch_calls,
                stage_durations=stage_durations,
            )
            for name, status in query_statuses.items():
                prior = provider_statuses.get(name)
                if status.startswith("skipped_") and prior:
                    continue
                if prior in {"success", "zero_results"} and status in {
                    "unavailable", "timed_out", "rate_limited", "request_failed",
                }:
                    provider_statuses[name] = "partial_failure"
                elif prior != "partial_failure":
                    provider_statuses[name] = status
            external_errors.extend(query_errors)
            for provider_name, diagnostic in query_diagnostics.items():
                provider_diagnostics[f"{provider_name}:Q{query_index}"] = {
                    **diagnostic,
                "attempted": not (
                    str(diagnostic.get("status", "")).startswith("skipped")
                    or str(diagnostic.get("status", "")).casefold() == "disabled"
                ),
                    "provider_result_count": int(diagnostic.get("candidate_count", 0) or 0),
                    "query_id": f"Q{query_index}",
                    "query": query,
                    "provider_query": query,
                }
            for item in external_items:
                external_by_id[item.evidence_id] = item
                query_evidence_ids.setdefault(query, set()).add(item.evidence_id)
            if stop_external_after_fused_evidence(list(external_by_id.values())):
                logger.info(
                    "evidence_diag external_query_stop reason=direct_evidence_sufficient "
                    "completed_queries=%d remaining_queries=%d",
                    query_index, max(0, len(external_queries) - query_index),
                )
                break
            if query_index < len(external_queries):
                combined_fused = self.fusion_manager.fuse(
                    local_evidence=local_evidence,
                    external_evidence=list(external_by_id.values()),
                    top_k=top_k_fused,
                    **({"claim": claim, "domain": domain}
                       if isinstance(self.fusion_manager, EvidenceFusionManager) else {}),
                )
                accepted_external_count = sum(
                    item.retrieval_method == "external" for item in combined_fused
                )
                logger.info(
                    "evidence_diag component_query query=%r providers=%s accepted_external=%d",
                    query,
                    sorted(query_diagnostics),
                    accepted_external_count,
                )
                # Relevance alone cannot prove the claim. Complete the bounded
                # query plan so an alternate formulation can reach a direct
                # supporting or contradicting source.

        external_retrieval_seconds = time.perf_counter() - external_retrieval_started

        page_upgrade_started = time.perf_counter()
        external_evidence = self._upgrade_metadata_snippets(
            claim, list(external_by_id.values()),
            max_fetches=min(MAX_SNIPPET_UPGRADES_PER_ATTEMPT, self.max_web_fallback_urls),
            excluded_urls=fallback_urls_seen,
            page_fetch_counter=page_fetch_calls,
        )
        stage_durations["page_upgrade_fetch"] += time.perf_counter() - page_upgrade_started
        logger.info(
            "performance_diag phase=external_page_upgrade elapsed_seconds=%.4f "
            "evidence_count=%d",
            time.perf_counter() - page_upgrade_started, len(external_evidence),
        )
        page_upgrade_seconds = time.perf_counter() - page_upgrade_started
        fusion_started = time.perf_counter()
        fusion_kwargs = {
            "local_evidence": local_evidence,
            "external_evidence": external_evidence,
            "top_k": top_k_fused,
        }
        if isinstance(self.fusion_manager, EvidenceFusionManager):
            fusion_kwargs.update(claim=claim, domain=domain, semantic_ranking=True)
        fused_evidence = self.fusion_manager.fuse(**fusion_kwargs)
        # SearXNG snippets are URL-discovery metadata, never evidence. If the
        # bounded page-fetch stage could not extract source text, discard the
        # candidate before it can reach verification.
        before_discovery_filter = len(external_evidence)
        external_evidence = [
            item for item in external_evidence
            if not (item.metadata or {}).get("discovery_only")
            or (
                (item.metadata or {}).get("content_quality") == "source_text"
                and (item.metadata or {}).get("full_page_extracted") is True
            )
        ]
        if len(external_evidence) != before_discovery_filter:
            logger.info(
                "evidence_diag drop stage=searxng_discovery_only count=%d reason=source_text_unavailable",
                before_discovery_filter - len(external_evidence),
            )
            # Recompute fusion without snippets that failed source-page checks.
            fusion_kwargs["external_evidence"] = external_evidence
            fused_evidence = self.fusion_manager.fuse(**fusion_kwargs)
        logger.info(
            "performance_diag phase=evidence_fusion_external elapsed_seconds=%.4f "
            "input_count=%d output_count=%d",
            time.perf_counter() - fusion_started,
            len(local_evidence) + len(external_evidence), len(fused_evidence),
        )
        fusion_seconds = time.perf_counter() - fusion_started
        fused_ids = {item.evidence_id for item in fused_evidence}
        type_to_provider = {
            "news_api": "news_api", "fact_check_api": "fact_check",
            "gdelt": "gdelt", "searxng_search": "searxng",
            "encyclopedic": "wikimedia",
        }
        for diagnostic in provider_diagnostics.values():
            provider_name = diagnostic.get("provider")
            query_ids = query_evidence_ids.get(diagnostic.get("query"), set())
            diagnostic["normalized_count"] = sum(
                1 for item in external_evidence
                if item.evidence_id in query_ids
                if type_to_provider.get((item.metadata or {}).get("external_source_type"),
                                        (item.metadata or {}).get("external_source_type")) == provider_name
            )
            diagnostic["accepted"] = sum(
                1 for item in fused_evidence
                if item.retrieval_method == "external"
                and type_to_provider.get((item.metadata or {}).get("external_source_type"),
                                         (item.metadata or {}).get("external_source_type")) == provider_name
            )
        for query, query_ids in query_evidence_ids.items():
            logger.info(
                "evidence_diag query_fusion query=%r distinct_ids=%d fused_ranked=%d final=%d",
                query, len(query_ids), len(query_ids & fused_ids), len(fused_evidence),
            )
        logger.info(
            "evidence_diag final_before_verification local=%d external=%d fused=%d",
            len(local_evidence), len(external_evidence), len(fused_evidence),
        )
        self._release_retrieval_resources()
        verification_started = time.perf_counter()
        verification_result, verification_failure = self._verify_once(
            claim, fused_evidence, claim_scope=claim_scope, claim_id=claim_id,
            reason="final_ranked_evidence_bundle",
        )
        verification_seconds = time.perf_counter() - verification_started
        fusion_stage_durations = (
            dict(self.fusion_manager.last_stage_durations)
            if isinstance(self.fusion_manager, EvidenceFusionManager) else {}
        )
        stage_durations.update({
            "local_retrieval": local_retrieval_seconds,
            "external_provider_retrieval_total": external_retrieval_seconds,
            "evidence_fusion_ranking": fusion_seconds,
            "verification": verification_seconds,
            **fusion_stage_durations,
        })
        logger.info(
            "stage_durations claim_id=%s "
            + " ".join(f"{name}={duration:.4f}s" for name, duration in stage_durations.items()),
            claim_id,
        )
        return ClaimProcessingResult(
            claim=claim, domain=domain, local_evidence=local_evidence,
            local_retrieval_error=local_error, external_evidence=external_evidence,
            external_source_errors=external_errors, fused_evidence=fused_evidence,
            verification_result=verification_result,
            verification_failure=verification_failure,
            provider_statuses=provider_statuses,
            provider_diagnostics=provider_diagnostics,
            retrieval_diagnostics={
                "local_results": len(local_evidence),
                "local_accepted": len(local_fused),
                "external_queries": len(external_queries),
                "providers": list(provider_diagnostics.values()),
                "fetched_pages": sum(
                    1 for item in external_evidence
                    if (item.metadata or {}).get("full_page_extracted") is True
                ),
                "quality_rejected": max(0, len(external_by_id) - len(external_evidence)),
                "candidate_count": len(local_evidence) + len(external_by_id),
                "filtered_count": len(local_evidence) + len(external_evidence),
                "ranked_evidence": len(fused_evidence),
                "ranked_count": len(fused_evidence),
                "selected_count": len(fused_evidence[:MAX_EVIDENCE_ITEMS_TO_VERIFY]),
                "final_evidence_count": len(fused_evidence),
                "normalized_count": len(external_evidence),
                "quality_filter_count": max(0, len(external_by_id) - len(external_evidence)),
                "ranking_count": len(fused_evidence),
                "verification_count": (
                    len(verification_result.results)
                    if verification_result is not None
                    else len(fused_evidence) if verification_failure is not None else 0
                ),
                "verification_passes": 1,
                "verification_pass_id": f"{claim_id}:V1",
                "verification_seconds": verification_seconds,
                "stage_durations_seconds": stage_durations,
                "semantic_ranking": {
                    "available": getattr(self, "semantic_embedder_unavailable_reason", None) is None,
                    "fallback_active": getattr(self, "semantic_embedder_unavailable_reason", None) is not None,
                    "error": getattr(self, "semantic_embedder_unavailable_reason", None),
                },
                "ollama_calls": int(bool(fused_evidence[:MAX_EVIDENCE_ITEMS_TO_VERIFY])),
                "provider_call_count": sum(
                    1 for diagnostic in provider_diagnostics.values()
                    if diagnostic.get("attempted")
                ),
                "page_fetch_calls": page_fetch_calls[0],
                "accepted_evidence_count": sum(
                    1 for item in (verification_result.results if verification_result else [])
                    if item.relationship in {"SUPPORTS", "CONTRADICTS"}
                ),
                "final_state": (
                    "VERIFICATION_FAILED" if verification_failure is not None
                    else "RETRIEVAL_FAILED" if not fused_evidence and any(
                        status in {"unavailable", "timed_out", "rate_limited", "request_failed", "partial_failure"}
                        for status in provider_statuses.values()
                    )
                    else "NO_RELEVANT_EVIDENCE" if not fused_evidence
                    else "VERIFIED" if verification_result and verification_result.overall_assessment
                    in {"SUPPORTED", "CONTRADICTED"}
                    else "EVIDENCE_FOUND"
                ),
            },
        )

    def _verify_once(
        self, claim: str, fused_evidence: List[Evidence],
        claim_scope: ClaimScope | None = None,
        *, claim_id: str = "C1", reason: str = "final_ranked_evidence_bundle",
    ) -> tuple[Optional[OverallVerification], Optional[VerificationFailure]]:
        """Run one bounded verification operation and validate its contract."""
        fusion_manager = getattr(self, "fusion_manager", None)
        if isinstance(fusion_manager, EvidenceFusionManager) and claim:
            # Fusion ranks the full evidence set; for the bounded verifier bundle,
            # prefer the candidates whose claim terms align in one sentence.
            # This prevents a broad high-similarity page from displacing a direct
            # relationship passage at the verification boundary.
            verification_evidence = [
                item for _, item in sorted(
                    enumerate(fused_evidence),
                    key=lambda pair: (
                        (
                            0.6 * max(
                                fusion_manager._passage_alignment_score(claim, pair[1]),
                                fusion_manager._entity_relation_alignment_score(claim, pair[1]),
                            )
                            + 0.4 * fusion_manager._entity_overlap_score(claim, pair[1])
                        ),
                        -pair[0],
                    ),
                    reverse=True,
                )[:MAX_EVIDENCE_ITEMS_TO_VERIFY]
            ]
        else:
            verification_evidence = fused_evidence[:MAX_EVIDENCE_ITEMS_TO_VERIFY]
        if len(verification_evidence) < len(fused_evidence):
            logger.info(
                "evidence_diag verifier_input_capped total=%d verified=%d",
                len(fused_evidence), len(verification_evidence),
            )
        logger.info(
            "evidence_diag verifier_called=yes claim_id=%s verification_pass_id=%s "
            "evidence_count=%d evidence_ids=%s model_request=%s",
            claim_id, f"{claim_id}:V1",
            len(verification_evidence),
            [item.evidence_id for item in verification_evidence],
            "yes" if verification_evidence else "no",
        )
        for item in verification_evidence:
            logger.info(
                "evidence_verifier_selection claim=%r evidence_id=%s source=%r title=%r provider=%r excerpt=%r",
                claim, item.evidence_id, item.source, item.title,
                (item.metadata or {}).get("provider") or (item.metadata or {}).get("external_source_type") or item.retrieval_method,
                item.text[:900],
            )
        try:
            verification_kwargs = {
                "claim": claim, "evidence_list": verification_evidence,
                "claim_id": claim_id,
                "verification_pass_id": f"{claim_id}:V1",
                "caller": "ClaimProcessingPipeline._verify_once",
                "retry_number": 0,
                "reason": reason,
            }
            if claim_scope:
                verification_kwargs["claim_scope"] = claim_scope
            result = self.verification_manager.verify(**verification_kwargs)
            if not isinstance(result, OverallVerification):
                raise TypeError("verification manager returned an invalid result")
            logger.info("evidence_diag verifier_completed=yes")
            return result, None
        except Exception as exc:
            logger.info(
                "evidence_diag verifier_completed=no exception_type=%s",
                type(exc).__name__,
            )
            logger.error("Verification failed: %s", exc)
            return None, VerificationFailure.from_exception(
                claim=claim, evidence_count=len(verification_evidence), exc=exc
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
        claim_scope: ClaimScope | None = None,
        claim_id: str = "C1",
    ) -> ClaimProcessingResult:
        """Retrieve FAISS evidence first and fall back to external sources as needed."""
        domain = canonical_domain(domain)
        queries = retrieval_queries if retrieval_queries is not None else [retrieval_query or claim]
        return self._process_multiple_queries(
            claim=claim,
            domain=domain,
            queries=queries,
            top_k_local=top_k_local,
            top_k_external=top_k_external,
            top_k_fused=top_k_fused,
            claim_scope=claim_scope,
            claim_id=claim_id,
        )
