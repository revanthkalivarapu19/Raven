from types import SimpleNamespace
import time

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
from rag_ingestion.external_retrieval.gdelt_source import GDELTSource
from rag_ingestion.llm_resilience import request_deadline


class CountingExternal:
    def __init__(self):
        self.queries = []

    def search(self, query, domain, top_k):
        self.queries.append(query)
        return ExternalResult([ExternalEvidence(
            evidence_id=f"external-{query}", title="Test", text="Text",
            url="https://example.test/shared", source="test", domain=domain,
            metadata={"content_quality": "metadata_or_snippet"},
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
    class InsufficientVerification:
        def __init__(self):
            self.calls = []

        def verify(self, claim, evidence_list, **_kwargs):
            from rag_ingestion.verification.verification_schema import OverallVerification
            self.calls.append((claim, evidence_list))
            return OverallVerification(
                claim=claim, results=[], overall_assessment="INSUFFICIENT_EVIDENCE",
                overall_confidence=0.0,
            )

    pipeline.verification_manager = InsufficientVerification()
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
    # The bounded query plan is completed so alternate wording can find direct evidence.
    assert pipeline.external_manager.queries == pipeline.retrieval_manager.queries


def test_pipeline_logs_per_query_evidence_counts_and_verifier_call(caplog):
    pipeline = _pipeline()

    with caplog.at_level("INFO", logger="rag_ingestion.pipeline.claim_processing"):
        result = pipeline.process(
            claim="claim",
            domain="general",
            retrieval_queries=["query-one"],
        )

    diagnostics = "\n".join(record.message for record in caplog.records)
    assert "evidence_diag query='query-one'" in diagnostics
    assert "raw_counts query='query-one' local=1 external=1" in diagnostics
    assert "normalized=1 dropped=0" in diagnostics
    assert "fused_ranked=2" in diagnostics
    assert "final_before_verification local=1 external=1 fused=2" in diagnostics
    assert "verifier_called=yes claim_id=C1 verification_pass_id=C1:V1 evidence_count=2" in diagnostics
    assert len(result.fused_evidence) == 2


def test_fallback_budget_is_shared_and_urls_are_deduplicated():
    pipeline = _pipeline()

    pipeline.process(
        claim="claim",
        domain="general",
        retrieval_queries=["alpha", "beta", "gamma"],
    )

    assert len(pipeline.web_retriever.urls) == 1
    # Local sufficiency, between-query stopping, and final fusion are recorded.
    assert len(pipeline.fusion_manager.calls) == 4
    # Retrieval is completed before the single bounded semantic pass.
    assert len(pipeline.verification_manager.calls) == 1


def test_provider_timeout_is_bounded_by_request_budget():
    provider = GDELTSource(timeout=30, max_retries=0)
    with request_deadline(time.monotonic() + 0.01):
        assert 0 < provider.request_timeout() <= 0.011
