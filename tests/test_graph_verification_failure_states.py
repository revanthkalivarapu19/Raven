"""Failure propagation and supervisor safety for verification stage exits."""

from types import SimpleNamespace

import pytest

from agents import graph


def _pipeline_result(*, verification_result=None, verification_failure=None):
    return SimpleNamespace(
        local_evidence=[],
        external_evidence=[],
        external_source_errors=[],
        fused_evidence=[],
        local_retrieval_error=None,
        verification_result=verification_result,
        verification_failure=verification_failure,
    )


def _state(claim="A test claim.", domain="Science"):
    return {
        "claim": claim,
        "domain": {"domain": domain},
        "claim_scope": "GENERAL",
        "errors": [],
    }


def test_standalone_claim_is_preserved_without_invoking_rewriter(monkeypatch):
    class FailedExtraction:
        def extract_claim(self, _text):
            raise AssertionError("standalone proposition should pass through unchanged")

    monkeypatch.setattr(graph, "get_claim_agent", lambda: FailedExtraction())
    text = "Raw input must not become a claim."
    result = graph.claim_extraction_node({"text": text})

    assert result["claim"] == text
    assert result["atomic_claims"][0].text == text


@pytest.mark.parametrize(
    ("failure", "error_type"),
    [
        (TimeoutError("verifier timed out"), "TimeoutError"),
        (ConnectionError("verifier transport failed"), "ConnectionError"),
        (OSError("retrieval failed before verification"), "OSError"),
    ],
)
def test_runtime_failures_propagate_and_supervisor_returns_unverified(
    monkeypatch, failure, error_type
):
    class Pipeline:
        def process(self, **_kwargs):
            raise failure

    monkeypatch.setattr(graph, "get_claim_processing_pipeline", lambda: Pipeline())
    stage = graph.evidence_verification_node(_state())

    assert stage["verification_result"] is None
    assert stage["verification_failure"].error_type == error_type
    final = graph.supervisor_node({**_state(), **stage})
    assert final["supervisor_result"]["final_verdict"] == "Unverified"
    assert final["supervisor_result"]["confidence_score"] == 0


@pytest.mark.parametrize(
    ("claim", "domain", "message"),
    [
        ("", "Science", "missing claim"),
    ],
)
def test_prerequisite_failures_are_explicit_and_supervisor_safe(
    monkeypatch, claim, domain, message
):
    result = graph.evidence_verification_node(_state(claim, domain))

    assert result["verification_result"] is None
    assert result["verification_failure"].error_type == "UpstreamPrerequisiteError"
    assert message in result["verification_failure"].message
    final = graph.supervisor_node({**_state(claim, domain), **result})
    assert final["supervisor_result"]["final_verdict"] == "Unverified"
    assert final["supervisor_result"]["confidence_score"] == 0


def test_empty_invalid_pipeline_verification_becomes_failure(monkeypatch):
    class Pipeline:
        def process(self, **_kwargs):
            return _pipeline_result()

    monkeypatch.setattr(graph, "get_claim_processing_pipeline", lambda: Pipeline())
    stage = graph.evidence_verification_node(_state())

    assert stage["verification_result"] is None
    assert stage["verification_failure"].error_type == "TypeError"
    final = graph.supervisor_node({**_state(), **stage})
    assert final["supervisor_result"]["final_verdict"] == "Unverified"


def test_verifier_timeout_failure_from_pipeline_is_preserved(monkeypatch):
    from rag_ingestion.verification.verification_schema import VerificationFailure

    failure = VerificationFailure(
        claim="A test claim.",
        evidence_count=1,
        error_type="TimeoutError",
        message="verifier timed out",
    )

    class Pipeline:
        def process(self, **_kwargs):
            return _pipeline_result(verification_failure=failure)

    monkeypatch.setattr(graph, "get_claim_processing_pipeline", lambda: Pipeline())
    stage = graph.evidence_verification_node(_state())

    assert stage["verification_failure"] is failure
    assert stage["verification_result"] is None
    final = graph.supervisor_node({**_state(), **stage})
    assert final["supervisor_result"]["final_verdict"] == "Unverified"
    assert final["verification_failure"] is failure


def test_provider_availability_metadata_reaches_graph_state(monkeypatch):
    from rag_ingestion.verification.verification_schema import OverallVerification

    class Pipeline:
        def process(self, **_kwargs):
            return SimpleNamespace(
                local_evidence=[], external_evidence=[], external_source_errors=[],
                fused_evidence=[], local_retrieval_error=None,
                verification_result=OverallVerification(
                    claim="A test claim.", overall_assessment="INSUFFICIENT_EVIDENCE",
                    overall_confidence=0.0, results=[],
                ), verification_failure=None,
                provider_statuses={"news": "unavailable", "catalog": "zero_results"},
                external_provider_state="partial",
            )

    monkeypatch.setattr(graph, "_build_retrieval_queries", lambda **_kwargs: ["query"])
    monkeypatch.setattr(graph, "get_claim_processing_pipeline", lambda: Pipeline())
    result = graph.evidence_verification_node(_state())

    assert result["provider_statuses"] == {"news": "unavailable", "catalog": "zero_results"}
    assert result["external_provider_state"] == "partial"
