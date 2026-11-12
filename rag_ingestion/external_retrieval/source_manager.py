# src/external_retrieval/source_manager.py
"""External Source Manager.
Orchestrates multiple external evidence providers, combining their results
into a single normalized list of ExternalEvidence objects.
"""

from __future__ import annotations

import logging
import re
import time
import requests
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict

from .base_source import BaseExternalSource
from .source_schema import ExternalEvidence
from rag_ingestion.config.domains import canonical_domain, validate_domain
from rag_ingestion.llm_resilience import (
    VERIFICATION_BUDGET_RESERVE_SECONDS,
    remaining_request_budget,
)

logger = logging.getLogger(__name__)


def _provider_http_status(exc: Exception) -> int | None:
    """Find an HTTP status on a provider exception or any wrapped cause."""
    current: Exception | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if status is None:
            status = getattr(current, "status_code", None)
        if status is None:
            match = re.search(r"\bHTTP\s+(\d{3})\b", str(current), flags=re.IGNORECASE)
            status = int(match.group(1)) if match else None
        if status is not None:
            try:
                return int(status)
            except (TypeError, ValueError):
                pass
        current = current.__cause__ or current.__context__
    return None


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
    status = _provider_http_status(exc)
    text = str(exc).lower()
    cause = exc
    cause_names = []
    while cause is not None:
        cause_names.append(type(cause).__name__.casefold())
        cause = cause.__cause__ or cause.__context__
    if any("proxyerror" in name for name in cause_names):
        return "proxy_unavailable"
    if any("timeout" in name for name in cause_names):
        return "timeout"
    if any("connectionerror" in name or "connecterror" in name for name in cause_names):
        return "unavailable"
    if any("jsondecode" in name or "decodeerror" in name for name in cause_names):
        return "malformed_response"
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
    provider_statuses: Dict[str, str] = field(default_factory=dict)
    provider_diagnostics: Dict[str, Dict[str, object]] = field(default_factory=dict)

    @property
    def external_status(self) -> str:
        """Summarize provider outcome without confusing failure with no matches."""
        applicable = [
            status for status in self.provider_statuses.values()
            if not status.startswith("skipped_")
        ]
        if not applicable:
            return "not_applicable"
        successful = [status for status in applicable if status in {"success", "zero_results"}]
        failed = [status for status in applicable if status in {"request_failed", "rate_limited", "timed_out", "unavailable"}]
        if successful and failed:
            return "partial"
        if failed:
            return "provider_unavailable"
        if self.evidence:
            return "results_available"
        return "zero_results"


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
        self.provider_configuration: Dict[str, bool] = {}

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
        stop_when: Optional[Callable[[List[ExternalEvidence]], bool]] = None,
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
        provider_statuses: Dict[str, str] = {}
        provider_diagnostics: Dict[str, Dict[str, object]] = {
            name: {
                "provider": name, "enabled": False, "query": query,
                "status": "disabled", "http_status": None,
                "runtime_ms": 0, "candidate_count": 0,
                "failure_category": "disabled",
            }
            for name, enabled in self.provider_configuration.items() if not enabled
        }
        raw_result_count = 0
        duplicate_count = 0

        providers = self._providers
        if normalized_domain == "political":
            political_order = {
                "news_api": 0,
                "fact_check": 1,
                "gdelt": 2,
                "searxng": 4,
            }
            providers = sorted(
                self._providers,
                key=lambda provider: (
                    political_order.get(provider.name, 3),
                    not provider.news_capable,
                ),
            )

        for provider_index, provider in enumerate(providers):
            provider_started = time.monotonic()
            provider_started_at = time.time()
            if provider.name == "gdelt" and any(
                item.source_type == "news_api" for item in all_evidence
            ):
                provider_statuses[provider.name] = "skipped_newsapi_candidates"
                provider_diagnostics[provider.name] = {
                    "provider": provider.name, "enabled": True, "query": query,
                    "status": "skipped_newsapi_candidates", "http_status": None,
                    "runtime_ms": 0, "candidate_count": 0,
                    "failure_category": None,
                }
                continue
            remaining = remaining_request_budget()
            if remaining is not None and remaining <= VERIFICATION_BUDGET_RESERVE_SECONDS:
                provider_statuses[provider.name] = "skipped_verification_reserve"
                provider_diagnostics[provider.name] = {
                    "provider": provider.name, "enabled": True, "query": query,
                    "status": "skipped", "http_status": None, "runtime_ms": 0,
                    "candidate_count": 0, "failure_category": None,
                }
                logger.info(
                    "evidence_diag provider_skip provider=%s reason=verification_budget_reserve remaining_seconds=%.2f reserve_seconds=%.2f",
                    provider.name,
                    remaining,
                    VERIFICATION_BUDGET_RESERVE_SECONDS,
                )
                continue
            if temporal_context and temporal_context.get("has_explicit_timeframe") and not provider.supports_historical:
                provider_statuses[provider.name] = "skipped_unsupported_timeframe"
                provider_diagnostics[provider.name] = {
                    "provider": provider.name, "enabled": True, "query": query,
                    "status": "skipped_unsupported_timeframe", "http_status": None,
                    "runtime_ms": 0, "candidate_count": 0,
                    "failure_category": "unsupported_for_claim",
                }
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
                provider_statuses[provider.name] = "skipped_after_rate_limit"
                provider_diagnostics[provider.name] = {
                    "provider": provider.name, "enabled": True, "query": query,
                    "status": "skipped_after_rate_limit", "http_status": 429,
                    "runtime_ms": 0, "candidate_count": 0,
                    "failure_category": "rate_limited",
                }
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
                provider_statuses[provider.name] = "skipped_after_failure"
                provider_diagnostics[provider.name] = {
                    "provider": provider.name, "enabled": True, "query": query,
                    "status": "skipped_after_failure", "http_status": None,
                    "runtime_ms": 0, "candidate_count": 0,
                    "failure_category": "provider_unavailable",
                }
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
                    provider_statuses[provider.name] = "skipped_unsupported_domain"
                    provider_diagnostics[provider.name] = {
                        "provider": provider.name, "enabled": True, "query": query,
                        "status": "skipped_unsupported_domain", "http_status": None,
                        "runtime_ms": 0, "candidate_count": 0,
                        "failure_category": "unsupported_for_claim",
                    }
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
                http_status = _provider_http_status(exc)
                root_cause = exc
                while root_cause.__cause__ is not None:
                    root_cause = root_cause.__cause__
                logger.warning(
                    "Provider %s failed (%s; category=%s)",
                    provider.name,
                    type(exc).__name__,
                    category,
                )
                logger.info(
                    "evidence_diag provider_attempt provider=%s started_at=%.3f "
                    "duration_seconds=%.3f outcome=failure category=%s "
                    "failure_scope=%s proxy_detected=%s "
                    "exception_type=%s root_exception_type=%s http_status=%s candidate_count=0",
                    provider.name,
                    provider_started_at,
                    time.monotonic() - provider_started,
                    category,
                    "configuration" if category in {"proxy_unavailable", "authentication/configuration"}
                    else "transient" if category in {"timeout", "rate_limited", "unavailable"}
                    else "undetermined",
                    bool(requests.utils.get_environ_proxies(getattr(provider, "base_url", "https://example.org"))),
                    type(exc).__name__,
                    type(root_cause).__name__,
                    http_status if http_status is not None else "not_available",
                )
                errors.append(ProviderError(provider_name=provider.name, exception=exc, category=category))
                provider_statuses[provider.name] = (
                    "rate_limited" if category == "rate_limited"
                    else "timed_out" if category == "timeout"
                    else "unavailable" if category in {"unavailable", "proxy_unavailable"}
                    else "request_failed"
                )
                diagnostic_category = (
                    "timeout" if category == "timeout" else
                    "rate_limited" if category == "rate_limited" else
                    "parse_error" if category == "malformed_response" else
                    "connection_error" if category in {"unavailable", "proxy_unavailable"} else
                    "http_error" if http_status is not None else
                    "invalid_response"
                )
                provider_diagnostics[provider.name] = {
                    "provider": provider.name, "enabled": True, "query": query,
                    "status": "error", "http_status": http_status,
                    "runtime_ms": round((time.monotonic() - provider_started) * 1000),
                    "candidate_count": 0, "failure_category": diagnostic_category,
                }
                if category == "rate_limited":
                    self._rate_limited.add(provider.name)
                else:
                    # A provider gets one chance per retrieval attempt.  This
                    # prevents opaque provider failures from multiplying across
                    # every generated query; the graph may own a later retry.
                    self._failed.add(provider.name)
                continue

            provider_count = len(results)
            provider_statuses[provider.name] = "success" if provider_count else "zero_results"
            provider_diagnostics[provider.name] = {
                "provider": provider.name, "enabled": True, "query": query,
                "status": "success" if provider_count else "empty_results",
                "http_status": getattr(provider, "last_http_status", None),
                "runtime_ms": round((time.monotonic() - provider_started) * 1000),
                "candidate_count": provider_count,
                "failure_category": None if provider_count else "empty_results",
            }
            provider_counts[provider.name] = provider_count
            raw_result_count += provider_count
            logger.info(
                "evidence_diag provider_attempt provider=%s started_at=%.3f "
                "duration_seconds=%.3f outcome=success candidate_count=%d",
                provider.name,
                provider_started_at,
                time.monotonic() - provider_started,
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

            if stop_when is not None:
                try:
                    sufficient = bool(stop_when(list(all_evidence)))
                except Exception as exc:
                    sufficient = False
                    logger.info(
                        "evidence_diag provider_stop_check_failed exception_type=%s",
                        type(exc).__name__,
                    )
                if sufficient:
                    for remaining_provider in providers[provider_index + 1:]:
                        provider_statuses.setdefault(
                            remaining_provider.name, "skipped_sufficient_evidence"
                        )
                        provider_diagnostics.setdefault(remaining_provider.name, {
                            "provider": remaining_provider.name, "enabled": True,
                            "query": query, "status": "skipped_sufficient_evidence",
                            "http_status": None, "runtime_ms": 0,
                            "candidate_count": 0, "failure_category": None,
                        })
                    logger.info(
                        "evidence_diag external_stop reason=fusion_sufficient "
                        "providers_attempted=%d candidates=%d",
                        provider_index + 1, len(all_evidence),
                    )
                    break

        result = ExternalSourceResult(
            evidence=all_evidence,
            errors=errors,
            provider_counts=provider_counts,
            raw_result_count=raw_result_count,
            duplicate_count=duplicate_count,
            provider_statuses=provider_statuses,
            provider_diagnostics=provider_diagnostics,
        )
        logger.info(
            "evidence_diag external_provider_summary state=%s attempted=%s successful=%s failed=%s skipped=%s candidates=%d",
            result.external_status,
            sorted(name for name, status in provider_statuses.items() if not status.startswith("skipped_")),
            sorted(name for name, status in provider_statuses.items() if status in {"success", "zero_results"}),
            sorted(name for name, status in provider_statuses.items() if status in {"request_failed", "rate_limited", "timed_out", "unavailable"}),
            sorted(name for name, status in provider_statuses.items() if status.startswith("skipped_")),
            raw_result_count,
        )
        return result
