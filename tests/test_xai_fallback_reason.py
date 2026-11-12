import time

import pytest

from agents import graph
from rag_ingestion.verification.verification_schema import OverallVerification


CLAIM = "The Reserve Bank of India reduced the repo rate by 0.50% to support economic growth."


def _state(deadline, assessment="INSUFFICIENT_EVIDENCE"):
    verification = OverallVerification(
        claim=CLAIM,
        claim_scope="EVENT_SPECIFIC",
        results=[],
        overall_assessment=assessment,
        overall_confidence=0.0,
    )
    supervisor = {
        "final_verdict": "Unverified",
        "confidence_score": 0.0,
        "trust_score": 0.0,
        "reasoning_summary": "Verification found insufficient evidence.",
        "key_evidence_used": [],
    }
    return {
        "claim": CLAIM,
        "claim_scope": "EVENT_SPECIFIC",
        "domain": {"domain": "Finance"},
        "fused_evidence": [],
        "reflection_result": {},
        "verification_result": verification,
        "supervisor_result": supervisor,
        "request_deadline_at": deadline,
        "request_diagnostics": {"deadline_reached": deadline <= time.monotonic()},
        "errors": [],
    }, verification, supervisor


def _force_xai_budget_skip(monkeypatch):
    monkeypatch.setattr(graph, "_stage_budget_available", lambda *_args: False)


def test_actual_deadline_reached_allows_deadline_fallback(monkeypatch):
    _force_xai_budget_skip(monkeypatch)
    state, _, _ = _state(time.monotonic() - 1)

    result = graph.xai_node(state)

    assert "request deadline was reached" in result["final_output"]["explanation"].lower()
    assert "request_deadline_exceeded: XAI skipped" in result["errors"]


def test_budget_guard_skip_does_not_claim_deadline_was_reached(monkeypatch):
    _force_xai_budget_skip(monkeypatch)
    state, _, _ = _state(time.monotonic() + 60)

    result = graph.xai_node(state)

    explanation = result["final_output"]["explanation"].lower()
    assert "deadline was reached" not in explanation
    assert "request deadline" not in explanation
    assert "remaining execution budget" in explanation
    assert "xai_skipped: insufficient_remaining_budget" in result["errors"]


def test_insufficient_verification_is_preserved_in_budget_fallback(monkeypatch):
    _force_xai_budget_skip(monkeypatch)
    state, verification, supervisor = _state(time.monotonic() + 60)

    result = graph.xai_node(state)

    explanation = result["final_output"]["explanation"].lower()
    assert result["final_output"]["final_verdict"] == "Unverified"
    assert "unverified" in explanation
    assert "evidence was insufficient" in explanation
    assert state["verification_result"] is verification
    assert state["supervisor_result"] is supervisor


def test_xai_uses_immediate_deterministic_explanation(monkeypatch):
    state, verification, supervisor = _state(time.monotonic() + 1000)
    expected = {"final_verdict": "Unverified", "confidence_score": 0.0}
    monkeypatch.setattr(
        graph,
        "get_xai_agent",
        lambda: pytest.fail("XAI is explanatory only and must not block the final response"),
    )

    result = graph.xai_node(state)

    assert result["final_output"]["final_verdict"] == expected["final_verdict"]
    assert result["final_output"]["confidence"] == expected["confidence_score"]
    assert "Raven returned Unverified" in result["final_output"]["explanation"]
    assert state["verification_result"] is verification
    assert state["supervisor_result"] is supervisor
