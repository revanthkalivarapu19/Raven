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
    def __init__(self):
        self.queries = []

    def search(self, query, domain, top_k):
        self.queries.append(query)
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

    def verify(self, claim, evidence_list, **_kwargs):
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
    from rag_ingestion.verification.verification_schema import VerificationResult
    return OverallVerification(
        claim=claim,
        results=[VerificationResult(
            evidence_id="local-claim" if claim == "claim" else "local-The original claim.",
            claim=claim, relationship="SUPPORTS", confidence=0.8,
            reasoning="The controlled evidence supports the claim.", verifier="test",
            claim_aligned=True, directly_establishes_proposition=True,
        )],
        overall_assessment="SUPPORTED", overall_confidence=0.8
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
    pipeline = _pipeline(verifier)
    result = pipeline.process(
        claim="claim", domain="general", retrieval_queries=["query"]
    )

    assert result.verification_result is None
    assert result.verification_failure is not None
    assert result.verification_failure.error_type == "TimeoutError"
    assert len(verifier.calls) == 1
    assert pipeline.external_manager.queries == ["query"]


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


def test_verifier_bundle_prefers_sentence_aligned_evidence():
    from rag_ingestion.retrieval.evidence_fusion import EvidenceFusionManager

    claim = "The Falcon transports cargo to the station each month."
    topical = Evidence(
        evidence_id="topical", chunk_id="topical", document_id="topical",
        text="The Falcon has a cargo hold. The station receives supplies from several vehicles.",
        title="Falcon", domain="general", source="test", language="en", metadata={},
    )
    direct = Evidence(
        evidence_id="direct", chunk_id="direct", document_id="direct",
        text="The Falcon transports cargo to the station each month.",
        title="Study", domain="general", source="test", language="en", metadata={},
    )

    class Capture:
        def verify(self, claim, evidence_list, **_kwargs):
            self.evidence_ids = [item.evidence_id for item in evidence_list]
            return OverallVerification(
                claim=claim, results=[], overall_assessment="INSUFFICIENT_EVIDENCE",
                overall_confidence=0.0,
            )

    pipeline = ClaimProcessingPipeline.__new__(ClaimProcessingPipeline)
    pipeline.fusion_manager = EvidenceFusionManager()
    pipeline.verification_manager = Capture()
    pipeline._verify_once(claim, [topical, direct])

    assert pipeline.verification_manager.evidence_ids == ["direct", "topical"]


def test_verifier_bundle_preserves_entities_when_predicate_and_topic_overlap():
    from rag_ingestion.retrieval.evidence_fusion import EvidenceFusionManager

    claim = "The Falcon transports cargo to the station each month."
    evidence = [
        Evidence(
            evidence_id="diurnal", chunk_id="diurnal", document_id="diurnal",
            text="The Falcon transports cargo to the station each month, enabling resupply.",
            title="Falcon missions", domain="general", source="test", language="en", metadata={},
        ),
        Evidence(
            evidence_id="decoy", chunk_id="decoy", document_id="decoy",
            text="A cargo vessel approaches the station. Falcon vehicles are used for supply missions.",
            title="Supply vehicles", domain="general", source="test", language="en", metadata={},
        ),
        Evidence(
            evidence_id="partial", chunk_id="partial", document_id="partial",
            text="The station receives cargo each month from visiting vehicles.",
            title="Station logistics", domain="general", source="test", language="en", metadata={},
        ),
    ]

    class Capture:
        def verify(self, claim, evidence_list, **_kwargs):
            self.evidence_ids = [item.evidence_id for item in evidence_list]
            return OverallVerification(
                claim=claim, results=[], overall_assessment="INSUFFICIENT_EVIDENCE",
                overall_confidence=0.0,
            )

    pipeline = ClaimProcessingPipeline.__new__(ClaimProcessingPipeline)
    pipeline.fusion_manager = EvidenceFusionManager()
    pipeline.verification_manager = Capture()
    pipeline._verify_once(claim, evidence)

    assert pipeline.verification_manager.evidence_ids[0] == "diurnal"
