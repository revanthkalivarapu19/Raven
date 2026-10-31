from agents.graph import supervisor_node
from rag_ingestion.verification.verification_schema import OverallVerification, VerificationResult


def _item(relationship, same_event=True):
    return VerificationResult(
        evidence_id="e1", claim="claim", relationship=relationship,
        confidence=0.9, reasoning="test", verifier="test", same_event=same_event,
        claim_aligned=same_event,
        directly_establishes_proposition=True,
    )


def _state(items, assessment):
    return {
        "claim": "claim", "domain": {"domain": "general"},
        "verification_result": OverallVerification(
            claim="claim", results=items, overall_assessment=assessment,
            overall_confidence=0.8,
        ),
        "reflection_result": {"decision": "NO", "is_evidence_sufficient": False},
        "fused_evidence": [], "persona_insights": [],
    }


def test_all_insufficient_uses_insufficient_safety_path():
    result = supervisor_node(_state([], "INSUFFICIENT_EVIDENCE"))
    assert result["supervisor_result"]["final_verdict"] == "Unverified"
    assert "insufficient" in result["supervisor_result"]["reasoning_summary"].lower()


def test_mismatched_contradiction_plus_support_is_not_conflicting(monkeypatch):
    monkeypatch.setattr(
        "agents.graph.get_supervisor_agent",
        lambda: lambda payload: {"final_verdict": "Real", "confidence_score": 80},
    )
    result = supervisor_node(_state(
        [_item("SUPPORTS"), _item("CONTRADICTS", False)], "SUPPORTED"
    ))
    assert result["supervisor_result"]["final_verdict"] != "Fake"


def test_different_date_opposite_event_is_not_conflicting():
    result = supervisor_node(_state([_item("CONTRADICTS", False)], "INSUFFICIENT_EVIDENCE"))
    assert result["supervisor_result"]["final_verdict"] == "Unverified"
    assert "insufficient" in result["supervisor_result"]["reasoning_summary"].lower()


def test_genuine_same_event_conflict_is_unverified():
    result = supervisor_node(_state(
        [_item("SUPPORTS"), _item("CONTRADICTS")], "CONFLICTING_EVIDENCE"
    ))
    assert result["supervisor_result"]["final_verdict"] == "Unverified"
    assert "conflicting" in result["supervisor_result"]["reasoning_summary"].lower()


def test_canonical_insufficient_cannot_be_relabelled_conflict():
    result = supervisor_node(_state(
        [_item("CONTRADICTS", False)], "INSUFFICIENT_EVIDENCE"
    ))
    assert result["supervisor_result"]["key_evidence_used"] == []
    assert result["supervisor_result"]["final_verdict"] == "Unverified"
