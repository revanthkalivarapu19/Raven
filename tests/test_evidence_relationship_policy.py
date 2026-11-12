"""Deterministic evidence relationship aggregation and reflection regressions."""

import json

import pytest

from agents.graph import supervisor_node
from agents.reflection.reflection_agent import ReflectionAgent
from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification import ollama_verifier, verification_manager
from rag_ingestion.verification.verification_schema import (
    OverallVerification,
    VerificationResult,
)


def _evidence(index, text="Controlled fused evidence.", source="Controlled source"):
    return Evidence(
        evidence_id=f"evidence-{index}", chunk_id=f"chunk-{index}",
        document_id=f"doc-{index}", text=text, domain="general",
        source=source, language="en", similarity_score=0.9,
        authority_score=0.9, metadata={},
    )


def _checked_result(claim, evidence, relationship, confidence=0.92):
    return VerificationResult(
        evidence_id=evidence.evidence_id, claim=claim,
        relationship=relationship, confidence=confidence,
        reasoning="Controlled item-level relationship result.", verifier="test",
        same_event=False, claim_aligned=relationship != "INSUFFICIENT",
        directly_establishes_proposition=relationship in {"SUPPORTS", "CONTRADICTS"},
    )


def _verify_and_map(monkeypatch, claim, evidence, relationships):
    responses = iter(relationships)

    def fake_verify(c, items, **_kwargs):
        results = []
        for item in items:
            response = next(responses)
            relationship, confidence = response if isinstance(response, tuple) else (response, 0.92)
            results.append(_checked_result(c, item, relationship, confidence))
        return results

    monkeypatch.setattr(
        verification_manager, "verify_evidence_batch", fake_verify,
    )
    verified = verification_manager.EvidenceVerificationManager().verify(claim, evidence)
    supervisor = supervisor_node({
        "claim": claim, "claim_scope": verified.claim_scope,
        "domain": {"domain": "general"}, "fused_evidence": evidence,
        "verification_result": verified, "verification_failure": None,
        "reflection_result": {}, "persona_insights": [],
    })["supervisor_result"]
    return verified, supervisor


@pytest.mark.parametrize(
    "claim,text,relationships,assessment,verdict",
    [
        (
            "The Earth orbits the Sun once every year.",
            ["NASA states Earth completes one orbit around the Sun in about 365 days."],
            ["SUPPORTS"], "SUPPORTED", "Real",
        ),
        (
            "Smoking increases the risk of lung cancer.",
            ["The U.S. Surgeon General concludes smoking causes lung cancer."],
            ["SUPPORTS"], "SUPPORTED", "Real",
        ),
        (
            "The Earth is the center of the solar system and the Sun orbits Earth.",
            ["Earth orbits the Sun; the Sun does not orbit Earth."],
            ["CONTRADICTS"], "CONTRADICTED", "Fake",
        ),
        (
            "A proposition is true.", ["Evidence affirms the proposition."],
            ["SUPPORTS"], "SUPPORTED", "Real",
        ),
        (
            "A proposition is true.", ["Evidence directly negates the proposition."],
            ["CONTRADICTS"], "CONTRADICTED", "Fake",
        ),
        (
            "A proposition is true.",
            ["Evidence affirms the proposition.", "Another source confirms it."],
            ["SUPPORTS", "SUPPORTS"], "SUPPORTED", "Real",
        ),
        (
            "A proposition is true.",
            ["Evidence negates the proposition.", "Another source also negates it."],
            ["CONTRADICTS", "CONTRADICTS"], "CONTRADICTED", "Fake",
        ),
        (
            "A proposition is true.",
            ["Evidence affirms the proposition.", "Evidence negates the proposition."],
            ["SUPPORTS", "CONTRADICTS"], "SUPPORTED", "Real",
        ),
        (
            "A proposition is true.", ["An unrelated article discusses weather."],
            ["INSUFFICIENT"], "INSUFFICIENT_EVIDENCE", "Unverified",
        ),
    ],
)
def test_item_relationships_aggregate_deterministically(
    monkeypatch, claim, text, relationships, assessment, verdict,
):
    evidence = [_evidence(i, value, "NASA" if i == 0 and "NASA" in value else "Controlled source")
               for i, value in enumerate(text)]
    result, mapped = _verify_and_map(monkeypatch, claim, evidence, relationships)
    assert result.overall_assessment == assessment
    assert mapped["final_verdict"] == verdict


def test_empty_evidence_is_insufficient_and_unverified(monkeypatch):
    result, mapped = _verify_and_map(monkeypatch, "A proposition is true.", [], [])
    assert result.overall_assessment == "INSUFFICIENT_EVIDENCE"
    assert mapped["final_verdict"] == "Unverified"


def test_weak_opposing_relationship_does_not_override_strong_support(monkeypatch):
    claim = "A proposition is true."
    evidence = [_evidence(0), _evidence(1)]
    result, mapped = _verify_and_map(
        monkeypatch, claim, evidence, ["SUPPORTS", ("CONTRADICTS", 0.4)],
    )
    assert result.overall_assessment == "SUPPORTED"
    assert mapped["final_verdict"] == "Real"


@pytest.mark.parametrize("relationship,assessment", [
    ("CONTRADICTS", "INSUFFICIENT_EVIDENCE"),
    ("SUPPORTS", "INSUFFICIENT_EVIDENCE"),
])
def test_reflection_does_not_erase_strong_verified_relationship(
    monkeypatch, relationship, assessment,
):
    claim = "A proposition is true."
    evidence = _evidence(0)
    item = _checked_result(claim, evidence, relationship)
    verification = OverallVerification(
        claim=claim, claim_scope="GENERAL", results=[item],
        overall_assessment=assessment, overall_confidence=0.0,
    )
    monkeypatch.setattr(
        "agents.reflection.reflection_agent._LLM_RESILIENCE.chat",
        lambda _call: {"message": {"content": json.dumps({
            "decision": "NO", "is_evidence_sufficient": False,
            "confidence": 0.0, "notes": "The reflection model was cautious.",
            "claim_supported": False, "claim_contradicted": False,
        })}},
    )
    result = ReflectionAgent().reflect(
        claim_text=claim, domain="general",
        evidence_list=[evidence], verification_result=verification,
    )
    assert result["is_evidence_sufficient"] is True
    assert result["decision"] == "YES"
    assert result["claim_supported"] is (relationship == "SUPPORTS")
    assert result["claim_contradicted"] is (relationship == "CONTRADICTS")


def test_reflection_preserves_relationship_but_keeps_existing_quality_gate(monkeypatch):
    claim = "A proposition is true."
    evidence = _evidence(0)
    item = _checked_result(claim, evidence, "CONTRADICTS")
    verification = OverallVerification(
        claim=claim, claim_scope="GENERAL", results=[item],
        overall_assessment="CONTRADICTED", overall_confidence=0.92,
    )
    monkeypatch.setattr(
        "agents.reflection.reflection_agent._LLM_RESILIENCE.chat",
        lambda _call: {"message": {"content": json.dumps({
            "decision": "NO", "is_evidence_sufficient": False,
            "confidence": 0.0, "notes": "Low quality.",
            "claim_supported": False, "claim_contradicted": False,
        })}},
    )
    result = ReflectionAgent().reflect(
        claim_text=claim, domain="general",
        evidence_list=[{"evidence_id": evidence.evidence_id, "text": evidence.text,
                        "similarity_score": 0.5, "authority_score": 0.3}],
        verification_result=verification,
    )
    assert result["claim_contradicted"] is True
    assert result["is_evidence_sufficient"] is False
    assert result["decision"] == "NO"
    assert "quality" in result["notes"].lower()


def test_verifier_prompt_classifies_each_item_instead_of_total_sufficiency():
    prompt = ollama_verifier._build_prompt(
        "The Earth orbits the Sun once every year.",
        _evidence(0, "Earth completes one orbit around the Sun in about 365 days.", "NASA"),
    )
    assert "SUPPORTED|CONTRADICTED|INSUFFICIENT" in prompt
    assert "Return only JSON" in prompt
    assert "Do not include any other fields" in prompt
    assert "directly states an incompatible fact" in prompt
