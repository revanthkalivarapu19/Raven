# src/external_retrieval/__init__.py
"""External Retrieval abstraction package.
Provides foundational interfaces for retrieving evidence from external sources.
"""

from .source_schema import ExternalEvidence
from .base_source import BaseExternalSource
from .source_manager import (
    ExternalSourceManager,
    ExternalSourceResult,
    ProviderError,
)
from .gdelt_source import GDELTSource
from .fact_check_source import FactCheckSource
from .europe_pmc_source import EuropePMCSource
from .crossref_source import CrossrefSource
from .wikimedia_source import WikimediaSource
from .world_bank_source import WorldBankSource
from .ipu_parline_source import IPUParlineSource

__all__ = [
    "ExternalEvidence",
    "BaseExternalSource",
    "ExternalSourceManager",
    "ExternalSourceResult",
    "ProviderError",
    "GDELTSource",
    "FactCheckSource",
    "EuropePMCSource",
    "CrossrefSource",
    "WikimediaSource",
    "WorldBankSource",
    "IPUParlineSource",
]
