# src/external_retrieval/source_manager.py
"""External Source Manager.
Orchestrates multiple external evidence providers, combining their results
into a single normalized list of ExternalEvidence objects.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from .base_source import BaseExternalSource
from .source_schema import ExternalEvidence
from rag_ingestion.config.domains import canonical_domain, validate_domain
from rag_ingestion.llm_resilience import remaining_request_budget

logger = logging.getLogger(__name__)


@dataclass
class ProviderError:
    """Records a provider failure so callers can inspect what went wrong.

    Attributes:
        provider_name: The ``name`` attribute of the provider that failed.
        exception: The original exception raised by the provider.
    """

    provider_name: str
    exception: Exception
    category: str = "other_failure"


def classify_provider_failure(exc: Exception) -> str:
    """Translate provider-specific failures into stable manager categories."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    status = status if status is not None else getattr(exc, "status_code", None)
    text = str(exc).lower()
    if status == 429 or "429" in text or "rate limit" in text or "rate-limit" in text:
        return "rate_limited"
    if status in {401, 403} or "api key" in text or "authentication" in text:
        return "authentication/configuration"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "unsupported" in text or "historical period" in text:
        return "unsupported_for_claim"
    if "malformed" in text or "invalid json" in text or "html" in text or "xml" in text:
        return "malformed_response"
    if "connection" in text or "unavailable" in text or status in {502, 503, 504}:
        return "unavailable"
    return "other_failure"


@dataclass
class ExternalSourceResult:
    """Return value of :meth:`ExternalSourceManager.search`.

    Attributes:
        evidence: The combined, deduplicated, deterministically-ordered list
            of :class:`ExternalEvidence` objects from all successful providers.
        errors: A list of :class:`ProviderError` entries – one per provider
            that raised an exception during its ``search`` call.  An empty
            list means every registered provider succeeded.
    """

    evidence: List[ExternalEvidence] = field(default_factory=list)
    errors: List[ProviderError] = field(default_factory=list)
    provider_counts: Dict[str, int] = field(default_factory=dict)
    raw_result_count: int = 0
    duplicate_count: int = 0


class ExternalSourceManager:
    """Registry and dispatcher for external evidence providers.

    Usage::

        manager = ExternalSourceManager()
        manager.register(my_provider)
        result = manager.search("claim text", domain="medical", top_k=5)
        for ev in result.evidence:
            print(ev.title)
        for err in result.errors:
            print(f"Provider {err.provider_name} failed: {err.exception}")

    **Duplicate evidence IDs** – When multiple providers return evidence with
    the same ``evidence_id``, the *first occurrence wins*.  Provider ordering
    is deterministic (registration order), and each provider's result list is
    iterated in the order returned by the provider.  Subsequent duplicates are
    silently dropped so that evidence identity remains unique.

    **top_k semantics** – The ``top_k`` parameter is forwarded to *every*
    registered provider individually.  Each provider may return up to
    ``top_k`` results.  The manager does **not** further truncate the
    combined list, so the caller may receive up to
    ``top_k × len(providers)`` results (minus duplicates).
    """

    def __init__(self) -> None:
        self._providers: List[BaseExternalSource] = []
        self._rate_limited: set[str] = set()
        self._failed: set[str] = set()

    def begin_attempt(self) -> None:
        """Reset provider circuit-breakers for one bounded retrieval attempt."""
        self._rate_limited.clear()
        self._failed.clear()

    # ------------------------------------------------------------------
    # Provider management
    # ------------------------------------------------------------------

    def register(self, source: BaseExternalSource) -> None:
        """Register an external source provider.

        Providers are queried in registration order.

        Raises:
            TypeError: If *source* is not a :class:`BaseExternalSource`.
        """
        if not isinstance(source, BaseExternalSource):
            raise TypeError(
                f"Expected a BaseExternalSource instance, got {type(source).__name__}"
            )
        self._providers.append(source)

    @property
    def providers(self) -> List[BaseExternalSource]:
        """Return the list of registered providers (read-only copy)."""
        return list(self._providers)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        domain: Optional[str] = None,
        top_k: int = 5,
        language: Optional[str] = None,
        country: Optional[str] = None,
        temporal_context: Optional[Dict[str, object]] = None,
    ) -> ExternalSourceResult:
        """Query all registered providers and combine their results.

        Args:
            query: The claim or search string.
            domain: Optional domain filter forwarded to each provider.
            top_k: Maximum results *per provider*.

        Returns:
            An :class:`ExternalSourceResult` containing the combined evidence
            and any provider errors.
        """
        normalized_domain = validate_domain(domain) if domain is not None else None
        all_evidence: List[ExternalEvidence] = []
        errors: List[ProviderError] = []
        seen_ids: set = set()
        provider_counts: Dict[str, int] = {}
        raw_result_count = 0
        duplicate_count = 0

        providers = self._providers
        if normalized_domain == "political":
            providers = sorted(
                self._providers,
                key=lambda provider: not provider.news_capable,
            )

        for provider in providers:
            provider_started = time.monotonic()
            remaining = remaining_request_budget()
            if remaining is not None and remaining <= 0:
                logger.info(
                    "evidence_diag provider_skip provider=%s reason=deadline_guard",
                    provider.name,
                )
                errors.append(ProviderError(provider_name=provider.name, exception=RuntimeError("request deadline exceeded"), category="timeout"))
                continue
            if temporal_context and temporal_context.get("has_explicit_timeframe") and not provider.supports_historical:
                logger.info(
                    "evidence_diag provider_skip provider=%s reason=temporal_mismatch",
                    provider.name,
                )
                errors.append(ProviderError(
                    provider_name=provider.name,
                    exception=RuntimeError("provider does not support the requested historical period"),
                    category="unsupported_for_claim",
                ))
                continue
            if provider.name in self._rate_limited:
                logger.info(
                    "evidence_diag provider_skip provider=%s reason=rate_limited",
                    provider.name,
                )
                errors.append(ProviderError(
                    provider_name=provider.name,
                    exception=RuntimeError("provider skipped after HTTP 429 during this retrieval attempt"),
                    category="rate_limited",
                ))
                continue
            if provider.name in self._failed:
                logger.info(
                    "evidence_diag provider_skip provider=%s reason=prior_provider_failure",
                    provider.name,
                )
                errors.append(ProviderError(
                    provider_name=provider.name,
                    exception=RuntimeError("provider skipped after failure during this retrieval attempt"),
                    category="unavailable",
                ))
                continue
            if normalized_domain is not None and provider.supported_domains is not None:
                if normalized_domain not in provider.supported_domains:
                    logger.info(
                        "evidence_diag provider_skip provider=%s reason=unsupported_source",
                        provider.name,
                    )
                    continue
            try:
                search_args = {"query": query, "domain": normalized_domain, "top_k": top_k}
                if language is not None:
                    search_args["language"] = language
                if country is not None:
                    search_args["country"] = country
                results = provider.search(**search_args)
            except Exception as exc:
                category = classify_provider_failure(exc)
                logger.warning(
                    "Provider %s failed (%s; category=%s)",
                    provider.name,
                    type(exc).__name__,
                    category,
                )
                logger.info(
                    "evidence_diag provider_error provider=%s category=%s "
                    "exception_type=%s",
                    provider.name,
                    category,
                    type(exc).__name__,
                )
                errors.append(ProviderError(provider_name=provider.name, exception=exc, category=category))
                if category == "rate_limited":
                    self._rate_limited.add(provider.name)
                else:
                    # A provider gets one chance per retrieval attempt.  This
                    # prevents opaque provider failures from multiplying across
                    # every generated query; the graph may own a later retry.
                    self._failed.add(provider.name)
                continue

            logger.info("Provider %s completed in %.3fs", provider.name, time.monotonic() - provider_started)
            provider_count = len(results)
            provider_counts[provider.name] = provider_count
            raw_result_count += provider_count
            logger.info(
                "evidence_diag provider_result provider=%s raw=%d",
                provider.name,
                provider_count,
            )

            # Deduplicate: first occurrence wins (deterministic by
            # registration order then by result position).
            for ev in results:
                if ev.evidence_id not in seen_ids:
                    seen_ids.add(ev.evidence_id)
                    all_evidence.append(ev)
                else:
                    duplicate_count += 1
                    logger.info(
                        "evidence_diag drop stage=provider_dedup provider=%s "
                        "evidence_id=%s reason=duplicate",
                        provider.name,
                        ev.evidence_id,
                    )

        return ExternalSourceResult(
            evidence=all_evidence,
            errors=errors,
            provider_counts=provider_counts,
            raw_result_count=raw_result_count,
            duplicate_count=duplicate_count,
        )
