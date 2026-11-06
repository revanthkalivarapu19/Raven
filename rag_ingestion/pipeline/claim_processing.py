import gc
import logging
import os
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
from rag_ingestion.verification.claim_scope import ClaimScope
from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification.verification_schema import OverallVerification
from rag_ingestion.verification.verification_schema import VerificationFailure
from rag_ingestion.config.domains import canonical_domain
from rag_ingestion.config.config import config as _config  # Loads .env before providers access environment variables.
from rag_ingestion.utils.temporal_context import build_temporal_context
from rag_ingestion.llm_resilience import remaining_request_budget

logger = logging.getLogger(__name__)

MAX_RETRIEVAL_QUERIES_PER_ATTEMPT = 4
MIN_RETRIEVAL_BUDGET_SECONDS = 1.0
MAX_SNIPPET_UPGRADES_PER_ATTEMPT = 3


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
        if os.getenv("NEWS_API_KEY", "").strip():
            manager.register(NewsAPISource())
        else:
            logger.info("Provider news_api disabled because API key is unavailable")

        enable_gdelt = os.getenv("ENABLE_GDELT", "false").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if enable_gdelt:
            manager.register(GDELTSource(max_retries=0))
        else:
            logger.info("Provider gdelt disabled by configuration")

        if os.getenv("GOOGLE_FACTCHECK_API_KEY", "").strip():
            manager.register(FactCheckSource())
        else:
            logger.info("Provider fact_check disabled because API key is unavailable")

        manager.register(EuropePMCSource())
        manager.register(CrossrefSource())
        manager.register(WikimediaSource())
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
            if remaining is not None and remaining <= MIN_RETRIEVAL_BUDGET_SECONDS:
                logger.info(
                    "evidence_diag page_upgrade_skip evidence_id=%s reason=deadline_guard",
                    item.evidence_id,
                )
                break

            try:
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
            minimum_length = max(50, self.normalizer.min_text_length)
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
    ) -> tuple[List[Evidence], List[Evidence], Optional[ProviderError], List[ProviderError]]:
        """Retrieve and normalize one query without fusion or verification."""
        logger.info("evidence_diag query=%r", query)
        local_error = None
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

        external_kwargs = {"query": query, "domain": domain, "top_k": top_k_external}
        if isinstance(self.external_manager, ExternalSourceManager):
            external_kwargs["temporal_context"] = build_temporal_context(claim, [])
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
        if not _api_evidence_is_sufficient(
            external_result.evidence, self.minimum_sufficient_external_evidence
        ) and fallback_urls_remaining[0] > 0:
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
                    logger.info(
                        "evidence_diag drop stage=web_fallback evidence_id=%s "
                        "reason=%s exception_type=%s",
                        api_evidence.evidence_id,
                        _evidence_drop_reason(exc),
                        type(exc).__name__,
                    )
                    external_result.errors.append(
                        ProviderError(provider_name="web_retriever", exception=exc)
                    )

        normalized = []
        normalization_drops = 0
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
            "evidence_diag normalized query=%r raw=%d manager_deduplicated=%d "
            "normalized=%d dropped=%d",
            query,
            external_raw,
            len(external_result.evidence),
            len(normalized),
            normalization_drops,
        )
        return local_evidence, normalized, local_error, list(external_result.errors)

    def _process_multiple_queries(
        self, claim: str, domain: str, queries: List[str],
        top_k_local: int, top_k_external: int, top_k_fused: int,
        claim_scope: ClaimScope | None = None,
    ) -> ClaimProcessingResult:
        local_by_id = {}
        external_by_id = {}
        local_error = None
        external_errors = []
        normalized_queries = list(dict.fromkeys(
            query.strip() for query in queries if query and query.strip()
        ))[:MAX_RETRIEVAL_QUERIES_PER_ATTEMPT]
        begin_attempt = getattr(self.external_manager, "begin_attempt", None)
        if callable(begin_attempt):
            begin_attempt()
        fallback_urls_seen = set()
        fallback_urls_remaining = [
            min(self.max_web_fallback_urls, MAX_SNIPPET_UPGRADES_PER_ATTEMPT)
        ]
        query_evidence_ids: dict[str, set[str]] = {}
        for query in normalized_queries:
            remaining = remaining_request_budget()
            if remaining is not None and remaining <= MIN_RETRIEVAL_BUDGET_SECONDS:
                external_errors.append(ProviderError(
                    provider_name="retrieval_budget",
                    exception=RuntimeError("request deadline nearly reached; remaining queries skipped"),
                    category="timeout",
                ))
                break
            local_items, external_items, query_local_error, query_errors = self._retrieve_query(
                claim, query, domain, top_k_local, top_k_external,
                fallback_urls_seen, fallback_urls_remaining,
            )
            query_ids = set()
            local_duplicates = 0
            external_duplicates = 0
            for item in local_items:
                query_ids.add(item.evidence_id)
                if item.evidence_id in local_by_id:
                    local_duplicates += 1
                    logger.info(
                        "evidence_diag drop stage=query_dedup evidence_id=%s "
                        "reason=duplicate kind=local query=%r",
                        item.evidence_id,
                        query,
                    )
                local_by_id[item.evidence_id] = item
            for item in external_items:
                query_ids.add(item.evidence_id)
                if item.evidence_id in external_by_id:
                    external_duplicates += 1
                    logger.info(
                        "evidence_diag drop stage=query_dedup evidence_id=%s "
                        "reason=duplicate kind=external query=%r",
                        item.evidence_id,
                        query,
                    )
                external_by_id[item.evidence_id] = item
            query_evidence_ids[query] = query_ids
            if local_error is None and query_local_error is not None:
                local_error = query_local_error
            external_errors.extend(query_errors)
            logger.info(
                "evidence_diag query_counts query=%r local_raw=%d external_normalized=%d "
                "deduped_added=%d local_duplicates=%d external_duplicates=%d",
                query,
                len(local_items),
                len(external_items),
                len(local_items) + len(external_items)
                - local_duplicates
                - external_duplicates,
                local_duplicates,
                external_duplicates,
            )

        local_evidence = list(local_by_id.values())
        external_evidence = self._upgrade_metadata_snippets(
            claim,
            list(external_by_id.values()),
            max_fetches=MAX_SNIPPET_UPGRADES_PER_ATTEMPT - len(fallback_urls_seen),
            excluded_urls=fallback_urls_seen,
        )
        logger.info(
            "evidence_diag cross_query_deduped local=%d external=%d combined=%d",
            len(local_evidence),
            len(external_evidence),
            len(local_evidence) + len(external_evidence),
        )
        fusion_kwargs = {"local_evidence": local_evidence, "external_evidence": external_evidence, "top_k": top_k_fused}
        if isinstance(self.fusion_manager, EvidenceFusionManager):
            fusion_kwargs["claim"] = claim
            fusion_kwargs["domain"] = domain
        fused_evidence = self.fusion_manager.fuse(**fusion_kwargs)
        fused_ids = {item.evidence_id for item in fused_evidence}
        for query, query_ids in query_evidence_ids.items():
            logger.info(
                "evidence_diag query_fusion query=%r distinct_ids=%d fused_ranked=%d "
                "final=%d",
                query,
                len(query_ids),
                len(query_ids & fused_ids),
                len(fused_evidence),
            )
        logger.info(
            "evidence_diag final_before_verification local=%d external=%d fused=%d",
            len(local_evidence),
            len(external_evidence),
            len(fused_evidence),
        )
        self._release_retrieval_resources()
        verification_result, verification_failure = self._verify_once(
            claim, fused_evidence, claim_scope=claim_scope
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

    def _verify_once(
        self, claim: str, fused_evidence: List[Evidence],
        claim_scope: ClaimScope | None = None,
    ) -> tuple[Optional[OverallVerification], Optional[VerificationFailure]]:
        """Run one bounded verification operation and validate its contract."""
        logger.info(
            "evidence_diag verifier_called=yes evidence_count=%d model_request=%s",
            len(fused_evidence),
            "yes" if fused_evidence else "no",
        )
        try:
            verification_kwargs = {
                "claim": claim, "evidence_list": fused_evidence,
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
                claim=claim, evidence_count=len(fused_evidence), exc=exc
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
                claim_scope=claim_scope,
            )
        retrieval_query = (retrieval_query or claim).strip()

        logger.info(
            "Evidence retrieval query: %s",
            retrieval_query,
        )
        logger.info("evidence_diag query=%r", retrieval_query)

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
        logger.info("evidence_diag local_raw query=%r count=%d", retrieval_query, len(local_evidence))

        # 2. External retrieval
        external_kwargs = {"query": retrieval_query, "domain": domain, "top_k": top_k_external}
        if isinstance(self.external_manager, ExternalSourceManager):
            external_kwargs["temporal_context"] = build_temporal_context(claim, [])
        external_result: ExternalSourceResult = self.external_manager.search(**external_kwargs)
        external_raw = getattr(
            external_result, "raw_result_count", len(external_result.evidence)
        )
        logger.info(
            "evidence_diag external_raw query=%r count=%d provider_counts=%s "
            "provider_deduplicated=%d",
            retrieval_query,
            external_raw,
            getattr(external_result, "provider_counts", {}),
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
        logger.debug(
            "External retrieval returned %d items with %d errors",
            len(external_result.evidence),
            len(external_result.errors),
        )

        # 3. If API evidence is insufficient, fetch a bounded set of API-provided URLs.
        web_results: dict[str, WebRetrievalResult] = {}
        web_fetch_attempted: set[str] = set()
        if not _api_evidence_is_sufficient(
            external_result.evidence,
            self.minimum_sufficient_external_evidence,
        ):
            candidates = _fallback_candidates(external_result.evidence)
            for url_key, api_evidence in candidates[
                : min(self.max_web_fallback_urls, MAX_SNIPPET_UPGRADES_PER_ATTEMPT)
            ]:
                web_fetch_attempted.add(url_key)
                try:
                    web_result = self.web_retriever.retrieve(api_evidence.url)
                except Exception as exc:
                    logger.info(
                        "evidence_diag drop stage=web_fallback evidence_id=%s "
                        "reason=%s exception_type=%s",
                        api_evidence.evidence_id,
                        _evidence_drop_reason(exc),
                        type(exc).__name__,
                    )
                    logger.warning("Web fallback failed (%s)", type(exc).__name__)
                    external_result.errors.append(
                        ProviderError(provider_name="web_retriever", exception=exc)
                    )
                    continue

                if web_result.error or not _meaningful(web_result.text):
                    error = RuntimeError(
                        web_result.error or "Web retrieval returned no usable text"
                    )
                    logger.info(
                        "evidence_diag drop stage=web_fallback evidence_id=%s "
                        "reason=%s exception_type=%s",
                        api_evidence.evidence_id,
                        _evidence_drop_reason(error),
                        type(error).__name__,
                    )
                    logger.warning("Web fallback returned no usable text")
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
                logger.info(
                    "evidence_diag drop stage=normalization evidence_id=%s "
                    "provider=%s reason=%s exception_type=%s",
                    ev.evidence_id,
                    ev.source_type or ev.source,
                    _evidence_drop_reason(exc),
                    type(exc).__name__,
                )
                # Normalisation errors are treated as provider‑level failures.
                logger.warning(
                    "Normalization failed for external evidence %s (%s)",
                    ev.evidence_id,
                    type(exc).__name__,
                )
                external_result.errors.append(
                    ProviderError(provider_name="normalizer", exception=exc)
                )
        logger.info(
            "evidence_diag normalized query=%r raw=%d manager_deduplicated=%d "
            "normalized=%d dropped=%d",
            retrieval_query,
            external_raw,
            len(external_result.evidence),
            len(normalized_external),
            len(external_result.evidence) - len(normalized_external),
        )
        normalized_external = self._upgrade_metadata_snippets(
            claim,
            normalized_external,
            max_fetches=MAX_SNIPPET_UPGRADES_PER_ATTEMPT
            - len(web_fetch_attempted),
            excluded_urls=web_fetch_attempted,
        )

        # 5. Fusion
        fusion_kwargs = {"local_evidence": local_evidence, "external_evidence": normalized_external, "top_k": top_k_fused}
        if isinstance(self.fusion_manager, EvidenceFusionManager):
            fusion_kwargs["claim"] = claim
            fusion_kwargs["domain"] = domain
        fused_evidence: List[Evidence] = self.fusion_manager.fuse(**fusion_kwargs)
        logger.debug("Fusion produced %d items", len(fused_evidence))
        logger.info(
            "evidence_diag final_before_verification local=%d external=%d fused=%d",
            len(local_evidence),
            len(normalized_external),
            len(fused_evidence),
        )

        # 6. Verification – only the fused evidence are examined
        self._release_retrieval_resources()
        verification_result, verification_failure = self._verify_once(
            claim, fused_evidence, claim_scope=claim_scope
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
