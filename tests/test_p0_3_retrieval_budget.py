from types import SimpleNamespace

from rag_ingestion.pipeline.claim_processing import (
    ClaimProcessingPipeline,
    MAX_RETRIEVAL_QUERIES_PER_ATTEMPT,
)
from tests.test_p0_2_pipeline_boundary import (
    ExternalResult,
    NormalizerFake,
    RetrievalFake,
    FusionFake,
    VerificationFake,
)
from rag_ingestion.external_retrieval.source_schema import ExternalEvidence


class CountingExternal:
    def __init__(self):
        self.queries = []

    def search(self, query, domain, top_k):
        self.queries.append(query)
        return ExternalResult([ExternalEvidence(
            evidence_id=f"external-{query}", title="Test", text="Text",
            url="https://example.test/shared", source="test", domain=domain,
        )], [])


class CountingWeb:
    def __init__(self):
        self.urls = []

    def retrieve(self, url):
        self.urls.append(url)
        return SimpleNamespace(text="fallback text", error=None)


def _pipeline():
    pipeline = ClaimProcessingPipeline.__new__(ClaimProcessingPipeline)
    pipeline.retrieval_manager = RetrievalFake()
    pipeline.external_manager = CountingExternal()
    pipeline.normalizer = NormalizerFake()
    pipeline.fusion_manager = FusionFake()
    pipeline.verification_manager = VerificationFake()
    pipeline.web_retriever = CountingWeb()
    pipeline.minimum_sufficient_external_evidence = 2
    pipeline.max_web_fallback_urls = 1
    pipeline._release_retrieval_resources = lambda: None
    return pipeline


def test_query_budget_truncates_execution_and_deduplicates():
    pipeline = _pipeline()
    queries = [f"query-{index}" for index in range(6)] + ["query-1"]

    pipeline.process(claim="claim", domain="general", retrieval_queries=queries)

    assert len(pipeline.retrieval_manager.queries) == MAX_RETRIEVAL_QUERIES_PER_ATTEMPT
    assert pipeline.external_manager.queries == pipeline.retrieval_manager.queries


def test_fallback_budget_is_shared_and_urls_are_deduplicated():
    pipeline = _pipeline()

    pipeline.process(
        claim="claim",
        domain="general",
        retrieval_queries=["alpha", "beta", "gamma"],
    )

    assert len(pipeline.web_retriever.urls) == 1
    assert len(pipeline.fusion_manager.calls) == 1
    assert len(pipeline.verification_manager.calls) == 1
