"""Deterministic evidence fixtures for verifier aggregation semantics."""

import pytest

from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification import verification_manager
from rag_ingestion.verification.ollama_verifier import _build_prompt
from rag_ingestion.verification.verification_schema import VerificationResult


def _evidence(evidence_id, text):
    return Evidence(
        evidence_id=evidence_id,
        chunk_id=f"chunk-{evidence_id}",
        document_id=f"doc-{evidence_id}",
        text=text,
        domain="science",
        source="controlled-test",
        language="en",
        metadata={},
    )


def _result(claim, evidence, relationship, *, aligned, same_event=False):
    return VerificationResult(
        evidence_id=evidence.evidence_id,
        claim=claim,
        relationship=relationship,
        confidence=0.95,
        reasoning="Controlled evidence fixture.",
        verifier="test",
        same_event=same_event,
        claim_aligned=aligned,
        directly_establishes_proposition=relationship in {"SUPPORTS", "CONTRADICTS"},
    )


def _verify(monkeypatch, claim, evidence_items, classifications, claim_scope):
    calls = iter(classifications)

    def verify_evidence(claim_text, evidence, **_kwargs):
        relationship, aligned, same_event = next(calls)
        return _result(
            claim_text,
            evidence,
            relationship,
            aligned=aligned,
            same_event=same_event,
        )

    monkeypatch.setattr(verification_manager, "verify_evidence", verify_evidence)
    return verification_manager.EvidenceVerificationManager().verify(
        claim, evidence_items, claim_scope=claim_scope
    )


def test_aligned_support_is_supported(monkeypatch):
    claim = "The treatment reduces the measured outcome."
    evidence = _evidence("support", "The treatment lowered the measured outcome.")

    result = _verify(
        monkeypatch, claim, [evidence], [("SUPPORTS", True, False)], "GENERAL"
    )

    assert result.overall_assessment == "SUPPORTED"


def test_aligned_explicit_opposite_is_contradicted(monkeypatch):
    claim = "The treatment reduces the measured outcome."
    evidence = _evidence("contradiction", "The treatment increased the measured outcome.")

    result = _verify(
        monkeypatch, claim, [evidence], [("CONTRADICTS", True, False)], "GENERAL"
    )

    assert result.overall_assessment == "CONTRADICTED"


def test_unrelated_evidence_is_unverified(monkeypatch):
    claim = "The treatment reduces the measured outcome."
    evidence = _evidence("unrelated", "The agency published an annual report.")

    result = _verify(
        monkeypatch, claim, [evidence], [("INSUFFICIENT", False, False)], "GENERAL"
    )

    assert result.overall_assessment == "INSUFFICIENT_EVIDENCE"


def test_different_event_contradiction_is_ineligible(monkeypatch):
    claim = "The agency reduced the rate on Monday."
    evidence = _evidence("other-event", "The agency increased the rate on Tuesday.")

    result = _verify(
        monkeypatch,
        claim,
        [evidence],
        [("CONTRADICTS", True, False)],
        "EVENT_SPECIFIC",
    )

    assert result.overall_assessment == "INSUFFICIENT_EVIDENCE"


def test_eligible_opposing_results_remain_conflicting(monkeypatch):
    claim = "The treatment reduces the measured outcome."
    evidence = [
        _evidence("support", "The treatment lowered the measured outcome."),
        _evidence("contradiction", "The treatment increased the measured outcome."),
    ]

    result = _verify(
        monkeypatch,
        claim,
        evidence,
        [("SUPPORTS", True, False), ("CONTRADICTS", True, False)],
        "GENERAL",
    )

    assert result.overall_assessment == "CONFLICTING_EVIDENCE"


def test_event_specific_different_event_cannot_support_or_contradict(monkeypatch):
    claim = "The agency reduced the rate on Monday."
    evidence = _evidence("other-event", "The agency reduced the rate on Tuesday.")

    result = _verify(
        monkeypatch,
        claim,
        [evidence],
        [("CONTRADICTS", True, False)],
        "EVENT_SPECIFIC",
    )

    assert result.overall_assessment == "INSUFFICIENT_EVIDENCE"


def test_general_claim_uses_alignment_without_same_event(monkeypatch):
    claim = "The treatment reduces the measured outcome."
    evidence = _evidence("general-opposite", "The treatment increased the measured outcome.")

    result = _verify(
        monkeypatch, claim, [evidence], [("CONTRADICTS", True, False)], "GENERAL"
    )

    assert result.overall_assessment == "CONTRADICTED"


def test_prompt_requires_consistent_contradiction_alignment():
    prompt = _build_prompt(
        "The treatment reduces the measured outcome.",
        _evidence("prompt", "The treatment increases the measured outcome."),
    )

    assert "never return CONTRADICTS with claim_aligned=false" in prompt
    assert "identify explicit evidence of the opposite outcome" in prompt
