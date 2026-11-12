"""Unseen propositions exercise generic verification and API failure states."""

import json

import pytest
from fastapi.testclient import TestClient

from agents.claim_decomposition import decompose_claim
from api import main
from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification import ollama_verifier
from rag_ingestion.verification.verification_manager import EvidenceVerificationManager
from rag_ingestion.verification.verification_schema import OverallVerification, VerificationFailure, VerificationResult


def _evidence(text: str) -> Evidence:
    return Evidence(evidence_id="unseen-1", chunk_id="c", document_id="d", text=text,
                    domain="general", source="controlled new source", language="en", metadata={})


@pytest.mark.parametrize(("claim", "passage", "model_relation", "expected"), [
    ("The new malaria vaccine reduced admissions by 40%.",
     "The new malaria vaccine lowered hospital admissions by 40%.", "SUPPORTED", "SUPPORTED"),
    ("The comet passed Mars in 2023.",
     "The comet passed Jupiter in 2023; Mars was not in its path.", "CONTRADICTED", "CONTRADICTED"),
    ("The tablet uses a neural processing chip.",
     "The tablet has a high resolution display but does not specify its processor.", "INSUFFICIENT", "INSUFFICIENT_EVIDENCE"),
    ("The central bank increased the benchmark rate by 0.25%.",
     "The central bank raised its benchmark rate. The increase was 0.25%.", "SUPPORTED", "SUPPORTED"),
    ("The football club won the final 3–1.",
     "The football club lost the final 3–1.", "CONTRADICTED", "CONTRADICTED"),
])
def test_new_domain_claims_use_structured_semantics(monkeypatch, claim, passage, model_relation, expected):
    monkeypatch.setattr(ollama_verifier._LLM_RESILIENCE, "generate", lambda _call: {
        "response": json.dumps({
            "results": [{"evidence_id": "unseen-1", "relationship": model_relation}],
        })
    })
    verification = EvidenceVerificationManager(timeout=1).verify(claim, [_evidence(passage)])
    assert verification.overall_assessment == expected
    excerpt = verification.results[0].direct_evidence_excerpt
    if model_relation == "INSUFFICIENT":
        assert excerpt is None
    else:
        assert excerpt and excerpt in passage


def test_unseen_multi_event_input_decomposes_into_independent_claims():
    source = "A researcher met a minister, A governor spoke with a mayor, and Negotiations resumed in Port Azure."
    pieces = [
        "A researcher met a minister",
        "A governor spoke with a mayor",
        "Negotiations resumed in Port Azure.",
    ]
    result = decompose_claim(source, fallback_claim=source,
                             generate=lambda _prompt: json.dumps({"claims": pieces}))
    assert [item.id for item in result.claims] == ["C1", "C2", "C3"]
    assert [item.text for item in result.claims] == pieces


def _api_state(assessment="INSUFFICIENT_EVIDENCE", *, evidence=False, failure=False, providers=None):
    item = _evidence("A source describes a claimed event.")
    relationship = "SUPPORTS" if assessment == "SUPPORTED" else "INSUFFICIENT"
    verification = OverallVerification(
        claim="A new claim.", results=[VerificationResult(
            evidence_id=item.evidence_id, claim="A new claim.", relationship=relationship,
            confidence=0.9 if relationship == "SUPPORTS" else 0.0,
            reasoning="controlled", verifier="fixture", claim_aligned=relationship == "SUPPORTS",
            directly_establishes_proposition=relationship == "SUPPORTS",
        )] if evidence else [], overall_assessment=assessment,
        overall_confidence=0.9 if assessment == "SUPPORTED" else 0.0,
    )
    return {
        "claim": "A new claim.", "text": "A new claim.",
        "fused_evidence": [item] if evidence else [],
        "verification_result": verification,
        "verification_failure": VerificationFailure(claim="A new claim.", evidence_count=1,
                                                     error_type="TimeoutError", message="timed out") if failure else None,
        "provider_statuses": providers or {},
        "supervisor_result": {"final_verdict": "Unverified", "confidence_score": 0,
                              "reasoning_summary": "Safety review returned Unverified."},
        "final_output": {"final_verdict": "Unverified", "confidence": 0,
                         "explanation": "Safety review returned Unverified."},
    }


@pytest.mark.parametrize(("kwargs", "expected_state", "message"), [
    ({"providers": {"news_api": "timed_out"}}, "RETRIEVAL_FAILED", "Source retrieval failed"),
    ({"providers": {"news_api": "zero_results"}}, "NO_RELEVANT_EVIDENCE", "No reliable evidence"),
    ({"evidence": True, "failure": True}, "VERIFICATION_FAILED", "verification could not complete"),
    ({"evidence": True, "assessment": "CONFLICTING_EVIDENCE"}, "CONFLICTING_EVIDENCE", "sources conflict"),
    ({"evidence": True}, "EVIDENCE_FOUND", "did not directly establish"),
])
def test_api_preserves_uncertain_failure_reason(monkeypatch, kwargs, expected_state, message):
    monkeypatch.setattr(main, "run_pipeline", lambda **_unused: _api_state(**kwargs))
    response = TestClient(main.app).post("/api/verify", json={"text": "A new claim.", "image": None})
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "uncertain"
    assert body["verification_state"] == expected_state
    assert message in body["why"]


def test_conflict_without_evidence_is_reported_as_no_evidence(monkeypatch):
    monkeypatch.setattr(main, "run_pipeline", lambda **_unused: _api_state("CONFLICTING_EVIDENCE"))
    body = TestClient(main.app).post("/api/verify", json={"text": "A new claim.", "image": None}).json()
    assert body["verification_state"] == "NO_RELEVANT_EVIDENCE"
    assert "conflict" not in body["why"].lower()
