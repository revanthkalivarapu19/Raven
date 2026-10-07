# src/external_retrieval/source_manager.py
"""External Source Manager.
Orchestrates multiple external evidence providers, combining their results
into a single normalized list of ExternalEvidence objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from .base_source import BaseExternalSource
from .source_schema import ExternalEvidence
from rag_ingestion.config.domains import canonical_domain, validate_domain

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

        for provider in self._providers:
            if normalized_domain is not None and provider.supported_domains is not None:
                if normalized_domain not in provider.supported_domains:
                    continue
            try:
                search_args = {"query": query, "domain": normalized_domain, "top_k": top_k}
                if language is not None:
                    search_args["language"] = language
                if country is not None:
                    search_args["country"] = country
                results = provider.search(**search_args)
            except Exception as exc:
                logger.warning(
                    "Provider %s failed: %s", provider.name, exc,
                )
                errors.append(ProviderError(provider_name=provider.name, exception=exc))
                continue

            # Deduplicate: first occurrence wins (deterministic by
            # registration order then by result position).
            for ev in results:
                if ev.evidence_id not in seen_ids:
                    seen_ids.add(ev.evidence_id)
                    all_evidence.append(ev)

        return ExternalSourceResult(evidence=all_evidence, errors=errors)
