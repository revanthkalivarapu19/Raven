"""Deterministic aggregation and supervisor eligibility regressions."""

import pytest

from agents import graph
from agents.graph import supervisor_node
from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification import verification_manager
from rag_ingestion.verification.claim_scope import classify_claim_scope
from rag_ingestion.verification.verification_schema import OverallVerification, VerificationResult


RBI_CLAIM = "The Reserve Bank of India reduced the repo rate by 0.50% to support economic growth."
GENERAL_CLAIM = "AMPK activation increases lung fibrosis."


def _evidence(index):
    return Evidence(
        evidence_id=f"item-{index}", chunk_id=f"chunk-{index}",
        document_id=f"document-{index}", text="Controlled evidence text.",
        domain="general", source="test", language="en", metadata={},
    )


def _verify(monkeypatch, claim, scope, fields):
    responses = iter(fields)

    def fake_verify(claim, evidence, **_kwargs):
        relationship, same_event, claim_aligned, confidence = next(responses)
        return VerificationResult(
            evidence_id=evidence.evidence_id, claim=claim,
            relationship=relationship, same_event=same_event,
            claim_aligned=claim_aligned, confidence=confidence,
            directly_establishes_proposition=relationship in {"SUPPORTS", "CONTRADICTS"},
            reasoning="Controlled result.", verifier="test",
        )

    monkeypatch.setattr(verification_manager, "verify_evidence", fake_verify)
    return verification_manager.EvidenceVerificationManager().verify(
        claim, [_evidence(i) for i in range(len(fields))], claim_scope=scope,
    )


@pytest.mark.parametrize(
    "scope,relationship,same_event,claim_aligned,expected",
    [
        ("EVENT_SPECIFIC", "CONTRADICTS", False, True, "INSUFFICIENT_EVIDENCE"),
        ("EVENT_SPECIFIC", "CONTRADICTS", True, True, "CONTRADICTED"),
        ("EVENT_SPECIFIC", "SUPPORTS", True, True, "SUPPORTED"),
        ("EVENT_SPECIFIC", "SUPPORTS", False, True, "INSUFFICIENT_EVIDENCE"),
        ("GENERAL", "CONTRADICTS", False, True, "CONTRADICTED"),
        ("GENERAL", "SUPPORTS", False, True, "SUPPORTED"),
        ("EVENT_SPECIFIC", "CONTRADICTS", False, False, "INSUFFICIENT_EVIDENCE"),
        ("GENERAL", "CONTRADICTS", False, False, "INSUFFICIENT_EVIDENCE"),
    ],
)
def test_scope_controls_eligibility(
    monkeypatch, scope, relationship, same_event, claim_aligned, expected,
):
    claim = RBI_CLAIM if scope == "EVENT_SPECIFIC" else GENERAL_CLAIM
    result = _verify(
        monkeypatch, claim, scope,
        [(relationship, same_event, claim_aligned, 0.9)],
    )
    assert result.claim_scope == scope
    assert result.overall_assessment == expected
    assert result.results[0].relationship == relationship
    assert result.results[0].same_event is same_event
    assert result.results[0].claim_aligned is claim_aligned
    assert result.results[0].confidence == 0.9


def test_different_event_contradiction_does_not_conflict(monkeypatch):
    result = _verify(monkeypatch, RBI_CLAIM, "EVENT_SPECIFIC", [
        ("SUPPORTS", True, True, 0.9),
        ("CONTRADICTS", False, True, 1.0),
    ])
    assert result.overall_assessment == "SUPPORTED"
    assert result.overall_confidence == 0.9


def test_same_event_opposing_results_still_conflict(monkeypatch):
    result = _verify(monkeypatch, RBI_CLAIM, "EVENT_SPECIFIC", [
        ("SUPPORTS", True, True, 0.9),
        ("CONTRADICTS", True, True, 1.0),
    ])
    assert result.overall_assessment == "CONFLICTING_EVIDENCE"


def test_recorded_rbi_five_item_flags_are_insufficient(monkeypatch):
    result = _verify(monkeypatch, RBI_CLAIM, "EVENT_SPECIFIC", [
        ("INSUFFICIENT", False, False, 0.0),
        ("CONTRADICTS", False, True, 1.0),
        ("INSUFFICIENT", False, False, 0.0),
        ("INSUFFICIENT", False, False, 0.6),
        ("CONTRADICTS", False, False, 1.0),
    ])
    assert result.overall_assessment == "INSUFFICIENT_EVIDENCE"
    assert result.overall_confidence == 0.0
    assert [(r.relationship, r.same_event, r.claim_aligned, r.confidence)
            for r in result.results] == [
        ("INSUFFICIENT", False, False, 0.0),
        ("CONTRADICTS", False, True, 1.0),
        ("INSUFFICIENT", False, False, 0.0),
        ("INSUFFICIENT", False, False, 0.6),
        ("CONTRADICTS", False, False, 1.0),
    ]


def test_scope_classification_handles_undated_decision_and_general_proposition():
    assert classify_claim_scope(RBI_CLAIM) == "EVENT_SPECIFIC"
    assert classify_claim_scope(GENERAL_CLAIM) == "GENERAL"


def test_verification_manager_classifies_scope_when_not_supplied(monkeypatch):
    monkeypatch.setattr(
        verification_manager, "verify_evidence",
        lambda claim, evidence, **_kwargs: VerificationResult(
            evidence_id=evidence.evidence_id, claim=claim,
            relationship="CONTRADICTS", same_event=False,
            claim_aligned=True, confidence=1.0,
            directly_establishes_proposition=True,
            reasoning="Controlled result.", verifier="test",
        ),
    )
    result = verification_manager.EvidenceVerificationManager().verify(
        RBI_CLAIM, [_evidence(0)],
    )
    assert result.claim_scope == "EVENT_SPECIFIC"
    assert result.overall_assessment == "INSUFFICIENT_EVIDENCE"


def test_extraction_node_records_scope_with_claim(monkeypatch):
    monkeypatch.setattr(
        graph, "get_claim_agent",
        lambda: type("Agent", (), {"extract_claim": lambda self, text: RBI_CLAIM})(),
    )
    extracted = graph.claim_extraction_node({"processed_text": RBI_CLAIM})
    assert extracted == {"claim": RBI_CLAIM, "claim_scope": "EVENT_SPECIFIC"}


@pytest.mark.parametrize(
    "scope,same_event,aligned,expected_phrase,expected_ids",
    [
        ("EVENT_SPECIFIC", False, True, "insufficient", []),
        ("EVENT_SPECIFIC", True, True, "conflicting", ["item-0"]),
        ("GENERAL", False, True, "conflicting", ["item-0"]),
        ("GENERAL", False, False, "insufficient", []),
    ],
)
def test_supervisor_safety_uses_shared_eligibility(
    scope, same_event, aligned, expected_phrase, expected_ids,
):
    claim = RBI_CLAIM if scope == "EVENT_SPECIFIC" else GENERAL_CLAIM
    item = VerificationResult(
        evidence_id="item-0", claim=claim, relationship="CONTRADICTS",
        same_event=same_event, claim_aligned=aligned, confidence=1.0,
        reasoning="Controlled result.", verifier="test",
        directly_establishes_proposition=True,
    )
    verification = OverallVerification(
        claim=claim, claim_scope=scope, results=[item],
        overall_assessment="CONFLICTING_EVIDENCE", overall_confidence=1.0,
    )
    state = {
        "claim": claim, "claim_scope": scope, "domain": {"domain": "general"},
        "verification_result": verification, "fused_evidence": [],
        "reflection_result": {}, "persona_insights": [],
    }
    verdict = supervisor_node(state)["supervisor_result"]
    assert verdict["final_verdict"] == "Unverified"
    assert expected_phrase in verdict["reasoning_summary"].lower()
    assert verdict["key_evidence_used"] == expected_ids


@pytest.mark.parametrize(
    "relationship,assessment",
    [("SUPPORTS", "SUPPORTED"), ("CONTRADICTS", "CONTRADICTED")],
)
def test_supervisor_rejects_definitive_assessment_without_eligible_item(
    monkeypatch, relationship, assessment,
):
    monkeypatch.setattr(
        graph, "get_supervisor_agent",
        lambda: lambda _payload: pytest.fail("supervisor must not be called"),
    )
    item = VerificationResult(
        evidence_id="item-0", claim=RBI_CLAIM, relationship=relationship,
        same_event=False, claim_aligned=True, confidence=1.0,
        reasoning="Controlled result.", verifier="test",
        directly_establishes_proposition=True,
    )
    verification = OverallVerification(
        claim=RBI_CLAIM, claim_scope="EVENT_SPECIFIC", results=[item],
        overall_assessment=assessment, overall_confidence=1.0,
    )
    state = {
        "claim": RBI_CLAIM, "claim_scope": "EVENT_SPECIFIC",
        "domain": {"domain": "finance"},
        "verification_result": verification, "fused_evidence": [],
        "reflection_result": {}, "persona_insights": [],
    }
    verdict = supervisor_node(state)["supervisor_result"]
    assert verdict["final_verdict"] == "Unverified"
