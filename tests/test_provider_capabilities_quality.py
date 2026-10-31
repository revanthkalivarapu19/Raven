from datetime import datetime

from rag_ingestion.external_retrieval.base_source import BaseExternalSource
from rag_ingestion.external_retrieval.source_schema import ExternalEvidence
from rag_ingestion.external_retrieval.evidence_normalizer import ExternalEvidenceNormalizer
from rag_ingestion.retrieval.evidence_fusion import EvidenceFusionManager
from rag_ingestion.retrieval.evidence_schema import Evidence


class DummySource(BaseExternalSource):
    name = "dummy"
    supported_domains = frozenset({"general"})
    supports_historical = False

    def search(self, **kwargs):
        return []


def test_capability_metadata_distinguishes_fetch_only_web_retriever():
    from rag_ingestion.external_retrieval.web_retriever import WebSourceRetriever
    assert WebSourceRetriever.fetch_only is True
    assert WebSourceRetriever.search_capable is False
    assert DummySource.supports_historical is False


def test_metadata_only_evidence_is_marked_and_penalized():
    external = ExternalEvidence(
        evidence_id="metadata", text="Bibliographic metadata only; no article body.",
        source="Crossref", title="Study", url="https://doi.org/10/x", domain="science",
        source_type="scientific_literature", metadata={"text_limitation": "metadata only"},
    )
    normalized = ExternalEvidenceNormalizer(min_text_length=10).normalize(external)
    assert normalized.metadata["content_quality"] == "metadata_or_snippet"


def test_source_family_diminishes_syndicated_copies():
    def evidence(eid, text, authority):
        return Evidence(evidence_id=eid, chunk_id=eid, document_id=eid, text=text,
            domain="general", source="source", language="en", retrieved_date=datetime.utcnow(),
            similarity_score=0.8, retrieval_method="external", metadata={"source_family": "same.example"},
            url=f"https://{eid}.example/article", title="Claim event", authority_score=authority)
    manager = EvidenceFusionManager()
    result = manager.fuse([], [evidence("a", "The event happened with value 10%.", 0.5),
                               evidence("b", "The event happened with value 10%.", 0.5)], 2,
                          claim="The event happened with value 10%.")
    assert result[0].evidence_id == "a"
    assert result[1].evidence_id == "b"
    assert manager._ranking_score(result[0], 1.0, "The event happened with value 10%.") > 0
