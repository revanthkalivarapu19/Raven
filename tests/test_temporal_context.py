from datetime import datetime

from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.utils.temporal_context import build_temporal_context
from rag_ingestion.pipeline.claim_processing import ClaimProcessingPipeline
from rag_ingestion.external_retrieval.source_manager import ExternalSourceManager
from rag_ingestion.external_retrieval.base_source import BaseExternalSource
from rag_ingestion.external_retrieval.source_schema import ExternalEvidence


def _evidence(publication_date=None):
    return Evidence(
        evidence_id="e1", chunk_id="c1", document_id="d1", text="text",
        domain="general", source="test", language="en", metadata={},
        publication_date=publication_date,
    )


def test_explicit_historical_year_and_different_evidence_year():
    result = build_temporal_context(
        "The event occurred in 2020.", [_evidence(datetime(2026, 1, 1))]
    )
    assert result["claim_timeframe"] == "2020"
    assert result["evidence_timeframes"] == ["2026"]
    assert "different publication years" in result["temporal_note"]


def test_missing_claim_date_is_not_fabricated():
    result = build_temporal_context("The event occurred.", [_evidence()])
    assert result["claim_timeframe"] == "Not specified"
    assert result["evidence_timeframes"] == []
    assert result["has_explicit_timeframe"] is False


def test_relative_date_is_preserved():
    result = build_temporal_context("The event happened last Friday.", [])
    assert result["claim_timeframe"] == "last Friday"


def test_on_weekday_is_preserved_without_inventing_a_date():
    result = build_temporal_context("The event happened on Friday.", [])
    assert result["claim_timeframe"] == "on Friday"
    assert result["has_explicit_timeframe"] is False


def test_evidence_without_publication_date_has_no_fabricated_timeframe():
    result = build_temporal_context("The event occurred in 2020.", [_evidence()])
    assert result["claim_timeframe"] == "2020"
    assert result["evidence_timeframes"] == []
    assert "temporal_note" not in result


def test_retrieve_query_uses_original_claim_for_temporal_context():
    class Provider(BaseExternalSource):
        name = "capture"
        supported_domains = frozenset({"general"})
        supports_historical = False
        calls = []

        def search(self, **kwargs):
            self.calls.append(kwargs)
            return []

    provider = Provider()
    manager = ExternalSourceManager()
    manager.register(provider)
    pipeline = ClaimProcessingPipeline.__new__(ClaimProcessingPipeline)
    pipeline.retrieval_manager = type("Local", (), {"search": lambda *args, **kwargs: []})()
    pipeline.external_manager = manager
    pipeline.normalizer = type("Normalizer", (), {"normalize": lambda self, item: item})()
    pipeline.web_retriever = type("Web", (), {})()
    pipeline.minimum_sufficient_external_evidence = 2
    pipeline.max_web_fallback_urls = 0

    pipeline._retrieve_query(
        "An event occurred in 2020.", "event 2020", "general", 1, 1
    )
    assert provider.calls == []
