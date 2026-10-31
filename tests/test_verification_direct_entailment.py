"""Deterministic checks that topical relevance cannot establish a verdict."""

import json

import pytest

from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification import ollama_verifier
from rag_ingestion.verification.verification_manager import EvidenceVerificationManager


def _evidence(text: str) -> Evidence:
    return Evidence(
        evidence_id="fixture", chunk_id="chunk", document_id="document",
        text=text, domain="science", source="controlled", language="en",
        metadata={},
    )


@pytest.mark.parametrize(
    ("claim", "text", "relationship", "direct", "expected"),
    [
        (
            "The drug reduces blood pressure.",
            "A randomized trial found the drug reduced blood pressure.",
            "SUPPORTS", True, "SUPPORTED",
        ),
        (
            "The drug reduces blood pressure.",
            "The drug is commonly prescribed for cardiovascular disease.",
            "SUPPORTS", False, "INSUFFICIENT_EVIDENCE",
        ),
        (
            "The drug reduces blood pressure.",
            "A randomized trial found the drug did not reduce blood pressure.",
            "CONTRADICTS", True, "CONTRADICTED",
        ),
        (
            "The drug reduces inflammation.",
            "The drug increases inflammation.",
            "CONTRADICTS", True, "CONTRADICTED",
        ),
        (
            "The treatment prevents disease progression.",
            "The treatment accelerates disease progression.",
            "CONTRADICTS", True, "CONTRADICTED",
        ),
        (
            "Basophils counteract disease development.",
            "Basophils amplify disease activity and promote lupus nephritis.",
            "CONTRADICTS", True, "CONTRADICTED",
        ),
        (
            "The protein suppresses inflammation.",
            "The protein is associated with inflammation.",
            "INSUFFICIENT", False, "INSUFFICIENT_EVIDENCE",
        ),
        (
            "CX3CR1 suppresses airway inflammation.",
            "CX3CR1 blockade prevented airway inflammation.",
            "INSUFFICIENT", False, "INSUFFICIENT_EVIDENCE",
        ),
        (
            "Bone marrow cells contribute to adult macrophage compartments.",
            "Host macrophages retained the capacity to expand after transplantation.",
            "SUPPORTS", False, "INSUFFICIENT_EVIDENCE",
        ),
        (
            "Albendazole is used to treat lymphatic filariasis.",
            "A review discusses lymphatic filariasis as a neglected tropical disease.",
            "SUPPORTS", False, "INSUFFICIENT_EVIDENCE",
        ),
        (
            "Aspirin inhibits production of PGE2.",
            "COX inhibition affects PGE2 and tumor inflammation.",
            "SUPPORTS", False, "INSUFFICIENT_EVIDENCE",
        ),
    ],
)
def test_verdict_requires_direct_propositional_evidence(
    monkeypatch, claim, text, relationship, direct, expected,
):
    response = {
        "relationship": relationship,
        "confidence": 0.99,
        "reasoning": "Controlled deterministic relationship classification.",
        "directly_establishes_proposition": direct,
        "same_event": True,
        "claim_aligned": True,
    }
    monkeypatch.setattr(
        ollama_verifier._LLM_RESILIENCE,
        "generate",
        lambda _call: {"response": json.dumps(response)},
    )

    result = EvidenceVerificationManager(timeout=1).verify(
        claim, [_evidence(text)]
    )

    assert result.results[0].relationship == relationship
    assert result.results[0].directly_establishes_proposition is direct
    assert result.overall_assessment == expected
    if not direct:
        assert result.overall_confidence == 0.0


def test_verifier_requires_explicit_directness_field(monkeypatch):
    response = {
        "relationship": "SUPPORTS",
        "confidence": 0.99,
        "reasoning": "Mentions the same subject.",
        "same_event": True,
        "claim_aligned": True,
    }
    monkeypatch.setattr(
        ollama_verifier._LLM_RESILIENCE,
        "generate",
        lambda _call: {"response": json.dumps(response)},
    )

    with pytest.raises(ValueError, match="directly_establishes_proposition"):
        EvidenceVerificationManager(timeout=1).verify(
            "The drug reduces blood pressure.",
            [_evidence("The drug is commonly prescribed.")],
        )


def test_direct_support_and_direct_refutation_can_still_conflict(monkeypatch):
    responses = iter([
        {
            "relationship": "SUPPORTS", "confidence": 0.9,
            "reasoning": "A trial observed the claimed reduction.",
            "directly_establishes_proposition": True,
            "same_event": True, "claim_aligned": True,
        },
        {
            "relationship": "CONTRADICTS", "confidence": 0.9,
            "reasoning": "A trial explicitly found no reduction.",
            "directly_establishes_proposition": True,
            "same_event": True, "claim_aligned": True,
        },
    ])
    monkeypatch.setattr(
        ollama_verifier._LLM_RESILIENCE,
        "generate",
        lambda _call: {"response": json.dumps(next(responses))},
    )

    result = EvidenceVerificationManager(timeout=1).verify(
        "The drug reduces blood pressure.",
        [_evidence("Trial A found a reduction."), _evidence("Trial B found no reduction.")],
    )

    assert result.overall_assessment == "CONFLICTING_EVIDENCE"


def test_prompt_distinguishes_direct_evidence_from_topical_relevance():
    prompt = ollama_verifier._build_prompt(
        "The drug reduces blood pressure.",
        _evidence("The drug is commonly prescribed for cardiovascular disease."),
    )

    assert "Do not mark SUPPORTS merely because evidence discusses the same entities or subject" in prompt
    assert "directly_establishes_proposition MUST be true only when evidence" in prompt
    assert "Explicitly compare the direction of the claim's predicate" in prompt
    assert "amplifying or counteracting" in prompt
    assert "A counteracts disease development" in prompt
    assert "A medicine treats disease B" in prompt
    assert "Do not set claim_aligned=false merely because the evidence's direction opposes the claim" in prompt
