# src/external_retrieval/base_source.py
"""Abstract base class for external evidence providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, FrozenSet

from .source_schema import ExternalEvidence


class BaseExternalSource(ABC):
    """Abstract interface for an external evidence source provider."""

    name: str
    # None means the provider is intentionally general-purpose.
    supported_domains: Optional[FrozenSet[str]] = None

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
