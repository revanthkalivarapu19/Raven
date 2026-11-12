# src/external_retrieval/base_source.py
"""Abstract base class for external evidence providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import List, Optional, FrozenSet

from .source_schema import ExternalEvidence
from rag_ingestion.llm_resilience import (
    VERIFICATION_BUDGET_RESERVE_SECONDS,
    remaining_request_budget,
)


class BaseExternalSource(ABC):
    """Abstract interface for an external evidence source provider."""

    name: str
    # None means the provider is intentionally general-purpose.
    supported_domains: Optional[FrozenSet[str]] = None
    # Providers that cannot search an explicitly historical period can opt out.
    supports_historical: bool = True
    # Capability metadata used for selection and diagnostics.
    search_capable: bool = True
    fetch_only: bool = False
    returns_full_text: bool = False
    news_capable: bool = False
    supports_temporal_filtering: bool = False
    requires_authentication: bool = False

    def request_timeout(self) -> float:
        """Return this provider's timeout bounded by the active request deadline."""
        configured = float(getattr(self, "timeout", 30.0))
        global_timeout = os.getenv("TIMEOUT_SECONDS", "").strip()
        if global_timeout:
            try:
                configured = min(configured, max(0.001, float(global_timeout)))
            except ValueError:
                pass
        remaining = remaining_request_budget()
        if remaining is None:
            return configured
        provider_budget = max(0.001, remaining - VERIFICATION_BUDGET_RESERVE_SECONDS)
        return max(0.001, min(configured, provider_budget))

    @abstractmethod
    def search(
        self,
        query: str,
        domain: Optional[str] = None,
        top_k: int = 5,
        language: Optional[str] = None,
        country: Optional[str] = None,
    ) -> List[ExternalEvidence]:
        """Search the external provider for evidence relevant to the query.

        Args:
            query: The claim or search query string.
            domain: An optional domain constraint (e.g., "medical", "politics").
            top_k: The maximum number of results to retrieve from this specific provider.
            language: Optional provider-supported language filter.
            country: Optional provider-supported country or region filter.

        Returns:
            A list of normalized ExternalEvidence objects.
        """
        raise NotImplementedError
