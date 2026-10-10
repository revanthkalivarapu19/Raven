from types import SimpleNamespace

from rag_ingestion.pipeline.claim_processing import ClaimProcessingPipeline
from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification.verification_schema import OverallVerification
from tests.test_p0_2_pipeline_boundary import ExternalResult, NormalizerFake


class Retrieval:
    def search(self, claim, domain, top_k):
        return [Evidence(
            evidence_id=f"local-{claim}", chunk_id="chunk", document_id="doc",
            text="text", domain=domain, source="test", language="en", metadata={}
        )]


class External:
    def search(self, query, domain, top_k):
        from rag_ingestion.external_retrieval.source_schema import ExternalEvidence
        return ExternalResult([ExternalEvidence(
            evidence_id=f"external-{query}", title="title", text="text",
            url=f"https://example.test/{query}", source="test", domain=domain
        )], [])


class Fusion:
    def __init__(self): self.calls = []
    def fuse(self, local_evidence, external_evidence, top_k):
        self.calls.append((local_evidence, external_evidence, top_k))
        return (local_evidence + external_evidence)[:top_k]


class Verifier:
    def __init__(self, result=None, error=None):
        self.calls = []
        self.result = result
        self.error = error

    def verify(self, claim, evidence_list):
        self.calls.append((claim, evidence_list))
        if self.error:
            raise self.error
        return self.result


def _pipeline(verifier):
    pipeline = ClaimProcessingPipeline.__new__(ClaimProcessingPipeline)
    pipeline.retrieval_manager = Retrieval()
    pipeline.external_manager = External()
    pipeline.normalizer = NormalizerFake()
    pipeline.fusion_manager = Fusion()
    pipeline.verification_manager = verifier
    pipeline.web_retriever = SimpleNamespace(retrieve=lambda url: None)
    pipeline.minimum_sufficient_external_evidence = 2
    pipeline.max_web_fallback_urls = 0
    pipeline._release_retrieval_resources = lambda: None
    return pipeline


def valid_result(claim):
    return OverallVerification(
        claim=claim, results=[], overall_assessment="SUPPORTED", overall_confidence=0.8
    )


def test_verifier_receives_bounded_evidence_once_and_original_claim():
    claim = "The original claim."
    verifier = Verifier(result=valid_result(claim))
    pipeline = _pipeline(verifier)

    result = pipeline.process(
        claim=claim, domain="general", retrieval_queries=["one", "two", "three"],
        top_k_fused=2,
    )

    assert len(verifier.calls) == 1
    assert verifier.calls[0][0] == claim
    assert len(verifier.calls[0][1]) == 2
    assert result.verification_failure is None


def test_verifier_timeout_becomes_failure():
    verifier = Verifier(error=TimeoutError("verification timed out"))
    result = _pipeline(verifier).process(
        claim="claim", domain="general", retrieval_queries=["query"]
    )

    assert result.verification_result is None
    assert result.verification_failure is not None
    assert result.verification_failure.error_type == "TimeoutError"


def test_verifier_exception_becomes_failure():
    verifier = Verifier(error=RuntimeError("verifier unavailable"))
    result = _pipeline(verifier).process(
        claim="claim", domain="general", retrieval_queries=["query"]
    )

    assert result.verification_result is None
    assert result.verification_failure is not None


def test_malformed_verifier_result_becomes_failure():
    verifier = Verifier(result={"overall_assessment": "SUPPORTED"})
    result = _pipeline(verifier).process(
        claim="claim", domain="general", retrieval_queries=["query"]
    )

    assert result.verification_result is None
    assert result.verification_failure is not None
    assert result.verification_failure.error_type == "TypeError"


def test_valid_verification_succeeds_once():
    claim = "claim"
    verifier = Verifier(result=valid_result(claim))
    result = _pipeline(verifier).process(
        claim=claim, domain="general", retrieval_queries=["query"]
    )

    assert len(verifier.calls) == 1
    assert result.verification_result.overall_assessment == "SUPPORTED"
