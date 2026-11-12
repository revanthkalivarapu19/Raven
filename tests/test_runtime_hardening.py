import json
import time
from types import SimpleNamespace

import pytest

from agents import graph
from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification import ollama_verifier


def _evidence(eid="e1"):
    return Evidence(evidence_id=eid, chunk_id="c", document_id="d", text="text",
                    domain="general", source="test", language="en", metadata={})


def _verifier_response(monkeypatch, payload):
    if set(payload) == {"relationship"}:
        relationship = {"SUPPORTS": "SUPPORTED", "CONTRADICTS": "CONTRADICTED"}.get(
            payload["relationship"], payload["relationship"],
        )
        output = {"results": [{"evidence_id": "e1", "relationship": relationship}]}
    else:
        output = payload
    monkeypatch.setattr(ollama_verifier._LLM_RESILIENCE, "generate",
                        lambda _call: {"response": json.dumps(output)})


def _valid_verifier():
    return {"relationship": "SUPPORTS"}


def test_verifier_complete_and_reasoning_optional(monkeypatch):
    _verifier_response(monkeypatch, _valid_verifier())
    result = ollama_verifier.verify_evidence(
        "The drug reduces blood pressure.",
        _evidence().model_copy(update={"text": "The drug reduces blood pressure."}),
    )
    assert result.relationship == "SUPPORTS"
    assert result.reasoning.startswith("Minimal item relationship; confidence is deterministic.")
    assert result.confidence == 0.9


@pytest.mark.parametrize("payload", [{}, {"relationship": "SUPPORTED", "confidence": 0.9}])
def test_verifier_requires_only_the_relationship_field(monkeypatch, payload):
    _verifier_response(monkeypatch, payload)
    result = ollama_verifier.verify_evidence("claim", _evidence())
    assert result.relationship == "INSUFFICIENT"
    assert result.verifier_failed


@pytest.mark.parametrize("payload", [
    {"relationship": "supported"}, {"relationship": 1},
    {"relationship": "SUPPORTED", "confidence": 0.99},
])
def test_verifier_invalid_output_fails_closed(monkeypatch, payload):
    _verifier_response(monkeypatch, payload)
    result = ollama_verifier.verify_evidence("claim", _evidence())
    assert result.relationship == "INSUFFICIENT"
    assert result.verifier_failed


def test_verification_failure_reflection_is_deterministic_and_retryable(monkeypatch):
    monkeypatch.setattr(graph, "get_reflection_agent", lambda: pytest.fail("LLM reflection called"))
    state = {"verification_failure": SimpleNamespace(message="verifier timed out"),
             "attempt_count": 1, "errors": [], "claim": "claim",
             "domain": {"domain": "General"}}
    result = graph.reflection_node(state)
    assert result["reflection_result"]["retry_requested"] is True
    assert result["reflection_result"]["attempt_number"] == 1
    assert graph.should_retry_after_reflection({**state, **result}) == "retry"


def test_verifier_failure_retry_respects_prerequisites_attempts_and_deadline():
    state = {"verification_failure": {"message": "timeout"}, "claim": "claim",
             "domain": {"domain": "General"}, "attempt_count": 1}
    assert graph.should_retry_after_reflection({**state, "claim": ""}) == "continue"
    assert graph.should_retry_after_reflection({**state, "attempt_count": graph.MAX_ATTEMPTS}) == "continue"
    assert graph.should_retry_after_reflection({**state, "request_deadline_at": time.monotonic() + 1}) == "continue"


def test_persona_reserve_skip_and_bad_output_are_isolated(monkeypatch):
    state = {"claim": "claim", "domain": {"domain": "General"},
             "fused_evidence": [_evidence()],
             "verification_result": SimpleNamespace(results=[]), "errors": [],
             "request_deadline_at": time.monotonic() + 1}
    result = graph.persona_analysis_node(state)
    assert "persona_analysis_skipped: insufficient_remaining_budget" in result["errors"]

    monkeypatch.setattr(graph, "_stage_budget_available", lambda *_args: True)
    monkeypatch.setattr(graph.SupervisorAgent, "build_context", lambda **_kwargs: object())
    bad = SimpleNamespace(name="journalist", analyze=lambda _ctx: {"evidence_ids": ["invented"]})
    good = SimpleNamespace(name="scientific", analyze=lambda _ctx: SimpleNamespace(evidence_ids=["e1"]))
    monkeypatch.setattr(graph, "get_persona_agents", lambda: [bad, good])
    isolated = graph.persona_analysis_node({**state, "request_deadline_at": None})
    assert isolated["persona_insights"] == [good.analyze(None)]
    assert any("persona_journalist_error" in error for error in isolated["errors"])


def test_persona_timeouts_are_isolated(monkeypatch):
    monkeypatch.setattr(graph, "_stage_budget_available", lambda *_args: True)
    monkeypatch.setattr(graph.SupervisorAgent, "build_context", lambda **_kwargs: object())
    timeout = SimpleNamespace(name="legal", analyze=lambda _ctx: (_ for _ in ()).throw(TimeoutError("slow")))
    valid = SimpleNamespace(name="journalist", analyze=lambda _ctx: SimpleNamespace(evidence_ids=["e1"]))
    monkeypatch.setattr(graph, "get_persona_agents", lambda: [timeout, valid])
    state = {"claim": "claim", "domain": {"domain": "General"}, "fused_evidence": [_evidence()],
             "verification_result": SimpleNamespace(results=[]), "errors": []}
    result = graph.persona_analysis_node(state)
    assert len(result["persona_insights"]) == 1
    assert any("persona_legal_error" in error for error in result["errors"])

