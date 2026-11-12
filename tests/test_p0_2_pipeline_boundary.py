import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from rag_ingestion.pipeline.claim_processing import ClaimProcessingPipeline
from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.retrieval.evidence_fusion import EvidenceFusionManager
from rag_ingestion.external_retrieval.source_schema import ExternalEvidence
from rag_ingestion.verification.verification_schema import (
    OverallVerification,
    VerificationFailure,
)
from rag_ingestion.config.domains import canonical_domain
from agents.query_planner import plan_queries


@dataclass
class ExternalResult:
    evidence: list
    errors: list


class RetrievalFake:
    def __init__(self, fail=False):
        self.queries = []
        self.fail = fail

    def search(self, claim, domain, top_k):
        self.queries.append(claim)
        if self.fail:
            raise FileNotFoundError("missing test index")
        return [Evidence(
            evidence_id=f"local-{claim}",
            chunk_id=f"chunk-{claim}",
            document_id="test-document",
            text=f"Local evidence for {claim}",
            domain=domain,
            source="test-local",
            language="en",
            metadata={},
        )]


class ExternalFake:
    def __init__(self):
        self.queries = []

    def search(self, query, domain, top_k):
        self.queries.append(query)
        return ExternalResult([ExternalEvidence(
            evidence_id=f"external-{query}",
            title=f"Test source for {query}",
            text=f"Evidence text for {query}",
            url=f"https://example.test/{query.replace(' ', '-')}",
            source="test-publisher",
            domain=domain,
        )], [])


class NormalizerFake:
    def normalize(self, evidence, **kwargs):
        return Evidence(
            evidence_id=evidence.evidence_id,
            chunk_id=f"external-{evidence.evidence_id}",
            document_id="test-external-document",
            text=evidence.text,
            domain=evidence.domain,
            source=evidence.source,
            language=evidence.language or "en",
            retrieval_method="external",
            metadata=evidence.metadata,
            url=evidence.url,
            title=evidence.title,
            publication_date=evidence.publication_date,
            authority_score=evidence.authority_score,
        )


class FusionFake:
    def __init__(self):
        self.calls = []

    def fuse(self, local_evidence, external_evidence, top_k):
        self.calls.append((local_evidence, external_evidence, top_k))
        return list(local_evidence) + list(external_evidence)


class VerificationFake:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def verify(self, claim, evidence_list, **_kwargs):
        self.calls.append((claim, evidence_list))
        if self.result is not None:
            return self.result
        from rag_ingestion.verification.verification_schema import VerificationResult
        return OverallVerification(
            claim=claim,
            results=[VerificationResult(
                evidence_id=item.evidence_id, claim=claim,
                relationship="SUPPORTS", confidence=0.8,
                reasoning="Controlled test fixture.", verifier="test",
                claim_aligned=True, directly_establishes_proposition=True,
            ) for item in evidence_list],
            overall_assessment="SUPPORTED" if evidence_list else "INSUFFICIENT_EVIDENCE",
            overall_confidence=0.8 if evidence_list else 0.0,
        )


def _pipeline(retrieval=None, verification=None):
    pipeline = ClaimProcessingPipeline.__new__(ClaimProcessingPipeline)
    pipeline.retrieval_manager = retrieval or RetrievalFake()
    pipeline.external_manager = ExternalFake()
    pipeline.normalizer = NormalizerFake()
    pipeline.fusion_manager = FusionFake()
    pipeline.verification_manager = verification or VerificationFake()
    pipeline.web_retriever = SimpleNamespace(retrieve=lambda url: None)
    pipeline.minimum_sufficient_external_evidence = 2
    pipeline.max_web_fallback_urls = 0
    pipeline._release_retrieval_resources = lambda: None
    return pipeline


def test_multi_query_aggregates_evidence_and_fuses_once():
    pipeline = _pipeline()
    queries = ["alpha search", "beta search", "gamma search"]

    result = pipeline.process(
        claim="The original claim.", domain="general", retrieval_queries=queries
    )

    assert pipeline.retrieval_manager.queries == queries
    assert len(result.local_evidence) == 3
    assert len(result.external_evidence) == 3
    assert len(pipeline.fusion_manager.calls) >= 1
    assert pipeline.external_manager.queries == queries


def test_verification_runs_once_with_original_claim():
    verification = VerificationFake()
    pipeline = _pipeline(verification=verification)
    claim = "The Federal Reserve reduced interest rates by 0.25 percent."

    pipeline.process(
        claim=claim,
        domain="finance",
        retrieval_queries=["Federal Reserve rates", "0.25 percent decision", "rate reduction"],
    )

    assert len(verification.calls) == 1
    assert verification.calls[0][0] == claim


def test_pipeline_passes_original_claim_into_fusion():
    claim = "Smoking increases the risk of lung cancer."

    class ClaimCapturingFusion(EvidenceFusionManager):
        def __init__(self):
            super().__init__()
            self.claims = []

        def fuse(self, *args, **kwargs):
            self.claims.append(kwargs.get("claim"))
            return super().fuse(*args, **kwargs)

    pipeline = _pipeline()
    pipeline.fusion_manager = ClaimCapturingFusion()
    pipeline.process(
        claim=claim,
        domain="medical",
        retrieval_queries=["smoking lung cancer risk", "smoking cancer risk", "lung cancer smoking"],
    )

    assert pipeline.fusion_manager.claims
    assert set(pipeline.fusion_manager.claims) == {claim}


def test_single_retrieval_query_remains_backward_compatible():
    pipeline = _pipeline()

    pipeline.process(
        claim="The original claim.",
        domain="general",
        retrieval_query="focused retrieval query",
    )

    assert pipeline.retrieval_manager.queries == ["focused retrieval query"]


def test_failed_retrieval_does_not_manufacture_success():
    verification = VerificationFake(result=OverallVerification(
        claim="The original claim.",
        results=[],
        overall_assessment="INSUFFICIENT_EVIDENCE",
        overall_confidence=0.0,
    ))
    pipeline = _pipeline(retrieval=RetrievalFake(fail=True), verification=verification)

    result = pipeline.process(
        claim="The original claim.",
        domain="general",
        retrieval_queries=["one", "two", "three"],
    )

    assert result.local_evidence == []
    assert len(pipeline.fusion_manager.calls) == 4
    assert len(verification.calls) == 1
    assert result.verification_result is not None
    assert result.verification_result.overall_assessment != "SUPPORTED"


def _load_graph_boundary():
    graph_path = Path(__file__).resolve().parents[1] / "agents" / "graph.py"
    tree = ast.parse(graph_path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    node = [node for node in functions if node.name == "evidence_verification_node"][-1]
    namespace = {
        "RavenState": dict,
        "Dict": dict,
        "Any": object,
        "logger": logging.getLogger(__name__),
        "OverallVerification": OverallVerification,
        "VerificationFailure": VerificationFailure,
        "canonical_domain": canonical_domain,
        "plan_queries": plan_queries,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(graph_path), "exec"), namespace)
    return namespace


def test_graph_calls_pipeline_once_and_not_managers_directly():
    namespace = _load_graph_boundary()
    calls = []

    class PipelineBoundaryFake:
        def process(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                local_evidence=[], external_evidence=[], fused_evidence=[],
                verification_result=None, verification_failure=None,
                local_retrieval_error=None, external_source_errors=[],
            )

    namespace["_build_retrieval_queries"] = lambda **kwargs: ["query one", "query two"]
    namespace["get_claim_processing_pipeline"] = lambda: PipelineBoundaryFake()

    result = namespace["evidence_verification_node"]({
        "claim": "The original claim.",
        "processed_text": "Context for the original claim.",
        "domain": {"domain": "general"},
        "errors": [],
    })

    assert len(calls) == 1
    assert calls[0]["retrieval_queries"] == ["The original claim."]
    assert calls[0]["claim"] == "The original claim."
    assert result["fused_evidence"] == []
    assert result["verification_result"] is None
    assert result["verification_failure"].error_type == "TypeError"
