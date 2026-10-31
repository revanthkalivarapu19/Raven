import time
from types import SimpleNamespace

import pytest

from agents import graph


@pytest.mark.parametrize(
    ("node", "skip_fragment"),
    [
        (graph.reflection_node, "reflection skipped"),
        (graph.persona_analysis_node, "persona analysis skipped"),
    ],
)
def test_budget_guard_skip_without_deadline_expiry_is_not_deadline_error(
    monkeypatch, node, skip_fragment
):
    monkeypatch.setattr(graph, "_stage_budget_available", lambda *_args: False)
    state = {
        "claim": "A factual claim.",
        "request_deadline_at": time.monotonic() + 60,
        "errors": [],
    }

    result = node(state)

    assert all("request_deadline_exceeded" not in error for error in result["errors"])
    assert any("insufficient_remaining_budget" in error for error in result["errors"])
    if node is graph.reflection_node:
        assert "remaining execution budget" in result["reflection_result"]["notes"]


@pytest.mark.parametrize(
    ("node", "expected_error"),
    [
        (graph.reflection_node, "request_deadline_exceeded: reflection skipped"),
        (graph.persona_analysis_node, "request_deadline_exceeded: persona analysis skipped"),
    ],
)
def test_skip_after_actual_deadline_expiry_reports_deadline_error(
    monkeypatch, node, expected_error
):
    monkeypatch.setattr(graph, "_stage_budget_available", lambda *_args: False)
    state = {
        "claim": "A factual claim.",
        "request_deadline_at": time.monotonic() - 1,
        "errors": [],
    }

    result = node(state)

    assert expected_error in result["errors"]
    if node is graph.reflection_node:
        assert result["reflection_result"]["notes"] == "Request deadline exceeded."


def test_normal_reflection_and_persona_execution_is_unchanged(monkeypatch):
    monkeypatch.setattr(graph, "_stage_budget_available", lambda *_args: True)
    reflection_result = {"decision": "YES", "notes": "reflection ran"}
    monkeypatch.setattr(
        graph,
        "get_reflection_agent",
        lambda: SimpleNamespace(reflect=lambda **_kwargs: reflection_result),
    )
    verification = SimpleNamespace(results=[])
    persona_insight = {"persona": "journalist", "analysis": "persona ran", "evidence_ids": ["e1"]}
    persona = SimpleNamespace(name="journalist", analyze=lambda _context: persona_insight)
    monkeypatch.setattr(graph.SupervisorAgent, "build_context", lambda **_kwargs: object())
    monkeypatch.setattr(graph, "get_persona_agents", lambda: [persona])
    state = {
        "claim": "A factual claim.",
        "domain": {"domain": "General"},
        "fused_evidence": [SimpleNamespace(evidence_id="e1")],
        "verification_result": verification,
        "request_deadline_at": time.monotonic() + 60,
        "errors": [],
    }

    reflection = graph.reflection_node(state)
    personas = graph.persona_analysis_node(state)

    assert reflection["reflection_result"] is reflection_result
    assert "errors" not in reflection
    assert personas["persona_insights"] == [persona_insight]
    assert personas["errors"] == []
