from types import SimpleNamespace
from datetime import datetime
import time

import pytest

from agents import graph
from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification.verification_schema import (
    OverallVerification,
    VerificationFailure,
)
from tests.test_p0_4_verification_bounds import (
    Verifier,
    _pipeline as _claim_processing_pipeline,
)


CLAIM = "The test authority changed its published rate by 0.25 percent."


def _evidence():
    return Evidence(
        evidence_id="evidence-1",
        chunk_id="chunk-1",
        document_id="document-1",
        text="The test authority changed its published rate by 0.25 percent.",
        domain="general",
        source="test-source",
        language="en",
        retrieved_date=datetime(2026, 1, 1),
        metadata={},
    )


def _verification(assessment):
    return OverallVerification(
        claim=CLAIM,
        results=[],
        overall_assessment=assessment,
        overall_confidence=0.8 if assessment == "SUPPORTED" else 0.0,
    )


class FakePipeline:
    def __init__(self, mode="success"):
        self.mode = mode
        self.calls = []
        self.evidence = _evidence()

    def process(self, **kwargs):
        self.calls.append(kwargs)
        if self.mode == "failure":
            return SimpleNamespace(
                local_evidence=[], external_evidence=[], fused_evidence=[],
                verification_result=None,
                verification_failure=VerificationFailure(
                    claim=kwargs["claim"], evidence_count=0,
                    error_type="TimeoutError", message="test timeout",
                ),
                local_retrieval_error=None, external_source_errors=[],
            )
        assessment = {
            "success": "SUPPORTED",
            "contradicted": "CONTRADICTED",
            "insufficient": "INSUFFICIENT_EVIDENCE",
            "conflict": "CONFLICTING_EVIDENCE",
            "retrieval_failure": "INSUFFICIENT_EVIDENCE",
            "no_evidence": None,
        }[self.mode]
        evidence = (
            []
            if self.mode in {"retrieval_failure", "no_evidence"}
            else [self.evidence]
        )
        return SimpleNamespace(
            local_evidence=evidence,
            external_evidence=[],
            fused_evidence=evidence,
            verification_result=_verification(assessment) if assessment else None,
            verification_failure=None,
            local_retrieval_error=(RuntimeError("retrieval failed")
                                    if self.mode == "retrieval_failure" else None),
            external_source_errors=[],
        )


class FakeClaimAgent:
    def extract_claim(self, text):
        return CLAIM


class FakeDomainAgent:
    def detect_domain(self, claim):
        return {"domain": "general", "confidence": 1.0, "reason": "test"}


class FakeReflection:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.calls = 0

    def reflect(self, **kwargs):
        self.calls += 1
        decision, confidence = next(self.decisions)
        return {
            "decision": decision,
            "confidence": confidence,
            "is_evidence_sufficient": decision == "YES",
            "contradictions_found": False,
            "claim_supported": decision == "YES",
            "claim_contradicted": False,
            "evidence_items_contradict_each_other": False,
        }


def _patch_runtime(monkeypatch, pipeline, reflection):
    monkeypatch.setattr(graph, "process_input", lambda **kwargs: {
        "processed_text": kwargs.get("text") or CLAIM,
        "pre_translation_text": kwargs.get("text") or CLAIM,
        "original_language": "en", "language_confidence": 1.0,
        "was_translated": False, "had_text": True, "had_image": False,
        "segments": [], "ocr_data": {}, "ocr_confidence": None,
        "quality_flags": [], "translation_info": {}, "errors": [],
    })
    monkeypatch.setattr(graph, "get_claim_agent", lambda: FakeClaimAgent())
    monkeypatch.setattr(graph, "get_domain_agent", lambda: FakeDomainAgent())
    monkeypatch.setattr(graph, "get_claim_processing_pipeline", lambda: pipeline)
    monkeypatch.setattr(graph, "get_reflection_agent", lambda: reflection)
    monkeypatch.setattr(graph, "get_persona_agents", lambda: [])
    monkeypatch.setattr(graph, "get_supervisor_agent", lambda: lambda value: {
        "final_verdict": "Real", "confidence_score": 80,
        "trust_score": 80, "reasoning_summary": "test",
        "key_evidence_used": [],
    })
    monkeypatch.setattr(graph, "get_xai_agent", lambda: SimpleNamespace(
        explain=lambda **kwargs: {
            "final_verdict": kwargs["supervisor_result"]["final_verdict"],
            "confidence": 0.8,
        }
    ))


def _run(monkeypatch, mode="success", decisions=None):
    pipeline = FakePipeline(mode)
    reflection = FakeReflection(decisions or [("YES", 0.9)])
    _patch_runtime(monkeypatch, pipeline, reflection)
    monkeypatch.setattr(graph, "_build_retrieval_queries", lambda **kwargs: [
        "dynamic query one", "dynamic query two"
    ])
    result = graph.build_raven_graph().invoke({
        "text": CLAIM, "errors": [], "attempt_count": 1,
        "maximum_attempts_reached": False,
    })
    return result, pipeline, reflection


def test_normal_success_reaches_end_with_output(monkeypatch):
    result, pipeline, _ = _run(monkeypatch)
    assert result["final_output"]
    assert len(pipeline.calls) == 1


@pytest.mark.parametrize("mode", ["insufficient", "conflict", "failure", "retrieval_failure"])
def test_unsafe_verification_states_reach_end_unverified(monkeypatch, mode):
    result, _, _ = _run(monkeypatch, mode=mode)
    assert result["supervisor_result"]["final_verdict"] == "Unverified"
    assert result["final_output"]["final_verdict"] == "Unverified"


def test_retry_path_executes_another_attempt_and_terminates(monkeypatch):
    result, pipeline, reflection = _run(
        monkeypatch, decisions=[("NO", 0.1), ("YES", 0.9)]
    )
    assert len(pipeline.calls) == 2
    assert reflection.calls == 2
    assert result["final_output"]


def test_retry_exhaustion_respects_max_attempts(monkeypatch):
    result, pipeline, reflection = _run(
        monkeypatch, decisions=[("NO", 0.1), ("NO", 0.1), ("NO", 0.1)]
    )
    assert len(pipeline.calls) == graph.MAX_ATTEMPTS
    assert reflection.calls == graph.MAX_ATTEMPTS
    assert result["final_output"]


def test_pipeline_boundary_and_dynamic_queries(monkeypatch):
    pipeline = FakePipeline()
    reflection = FakeReflection([("YES", 0.9)])
    _patch_runtime(monkeypatch, pipeline, reflection)
    planned = ["runtime query A", "runtime query B", "runtime query C"]
    planner_calls = []

    def planner(**kwargs):
        planner_calls.append(kwargs)
        return planned

    monkeypatch.setattr(graph, "_build_retrieval_queries", planner)
    graph.build_raven_graph().invoke({
        "text": CLAIM, "errors": [], "attempt_count": 1,
        "maximum_attempts_reached": False,
    })

    assert len(planner_calls) == 1
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0]["retrieval_queries"] == planned
    assert pipeline.calls[0]["claim"] == CLAIM


def test_expired_request_skips_retrieval_and_preserves_safe_state(monkeypatch):
    monkeypatch.setattr(graph, "get_claim_processing_pipeline", lambda: (_ for _ in ()).throw(AssertionError("pipeline must not run")))
    result = graph.evidence_verification_node({
        "claim": CLAIM,
        "domain": {"domain": "general"},
        "request_deadline_at": time.monotonic() - 1,
        "errors": [],
    })
    assert result["verification_result"] is None
    assert any("deadline" in error for error in result["errors"])


def test_nearly_expired_request_does_not_retry(monkeypatch):
    state = {
        "claim": CLAIM,
        "domain": {"domain": "general"},
        "request_deadline_at": time.monotonic() + 0.01,
        "reflection_result": {"decision": "NO", "confidence": 0.0},
        "verification_result": None,
        "verification_failure": None,
        "errors": [],
        "attempt_count": 1,
    }
    assert graph.should_retry_after_reflection(state) == "continue"


def test_insufficient_verification_preserves_fused_evidence_and_result(monkeypatch):
    pipeline = FakePipeline("insufficient")
    monkeypatch.setattr(graph, "get_claim_processing_pipeline", lambda: pipeline)
    monkeypatch.setattr(graph, "_build_retrieval_queries", lambda **kwargs: ["query"])

    result = graph.evidence_verification_node({
        "claim": CLAIM,
        "domain": {"domain": "general"},
        "errors": [],
    })

    assert result["fused_evidence"] == [_evidence()]
    assert result["verification_result"].overall_assessment == "INSUFFICIENT_EVIDENCE"


def test_supported_verification_preserves_fused_evidence_and_result(monkeypatch):
    pipeline = FakePipeline("success")
    monkeypatch.setattr(graph, "get_claim_processing_pipeline", lambda: pipeline)
    monkeypatch.setattr(graph, "_build_retrieval_queries", lambda **kwargs: ["query"])

    result = graph.evidence_verification_node({
        "claim": CLAIM,
        "domain": {"domain": "general"},
        "errors": [],
    })

    assert result["fused_evidence"] == [_evidence()]
    assert result["verification_result"] == _verification("SUPPORTED")


def test_contradicted_verification_preserves_fused_evidence_and_result(monkeypatch):
    pipeline = FakePipeline("contradicted")
    monkeypatch.setattr(graph, "get_claim_processing_pipeline", lambda: pipeline)
    monkeypatch.setattr(graph, "_build_retrieval_queries", lambda **kwargs: ["query"])

    result = graph.evidence_verification_node({
        "claim": CLAIM,
        "domain": {"domain": "general"},
        "errors": [],
    })

    assert result["fused_evidence"] == [_evidence()]
    assert result["verification_result"] == _verification("CONTRADICTED")


def test_verifier_timeout_preserves_already_retrieved_fused_evidence(monkeypatch):
    verifier = Verifier(error=TimeoutError("verification timed out"))
    claim_processing = _claim_processing_pipeline(verifier)
    processed_results = []

    class CapturingPipeline:
        def process(self, **kwargs):
            result = claim_processing.process(**kwargs)
            processed_results.append(result)
            return result

    monkeypatch.setattr(
        graph, "get_claim_processing_pipeline", lambda: CapturingPipeline()
    )
    monkeypatch.setattr(graph, "_build_retrieval_queries", lambda **kwargs: ["query"])

    result = graph.evidence_verification_node({
        "claim": CLAIM,
        "domain": {"domain": "general"},
        "errors": [],
    })

    expected = processed_results[0]
    assert expected.fused_evidence
    assert result["fused_evidence"] == expected.fused_evidence
    assert result["verification_result"] is None
    assert result["verification_failure"] == expected.verification_failure
    assert result["verification_failure"].error_type == "TimeoutError"
    assert result["verification_failure"].evidence_count == len(
        expected.fused_evidence
    )


def test_true_no_evidence_preserves_empty_evidence_and_none_verification(monkeypatch):
    pipeline = FakePipeline("no_evidence")
    monkeypatch.setattr(graph, "get_claim_processing_pipeline", lambda: pipeline)
    monkeypatch.setattr(graph, "_build_retrieval_queries", lambda **kwargs: ["query"])

    result = graph.evidence_verification_node({
        "claim": CLAIM,
        "domain": {"domain": "general"},
        "errors": [],
    })

    assert result["fused_evidence"] == []
    assert result["verification_result"] is None
    assert result["verification_failure"].error_type == "TypeError"
    assert "no valid verification result or failure" in result["verification_failure"].message


def test_rate_limited_provider_does_not_trigger_same_request_graph_retry():
    state = {
        "claim": CLAIM,
        "domain": {"domain": "political"},
        "attempt_count": 1,
        "verification_failure": VerificationFailure(
            claim=CLAIM, evidence_count=0,
            error_type="TimeoutError", message="provider unavailable",
        ),
        "verification_result": None,
        "errors": [
            "external_retrieval_error: provider=gdelt category=rate_limited"
        ],
    }

    assert graph.should_retry_after_reflection(state) == "continue"
