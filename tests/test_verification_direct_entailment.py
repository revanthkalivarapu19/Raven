"""Semantic verdicts remain grounded in cited evidence and generic constraints."""

import json

import pytest

from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification import ollama_verifier
from rag_ingestion.verification.directness_gate import evaluate_directness
from rag_ingestion.verification.verification_manager import EvidenceVerificationManager


def _evidence(text):
    return Evidence(evidence_id="e1", chunk_id="c1", document_id="d1", text=text,
                    domain="general", source="fixture", language="en", metadata={})


def _mock(monkeypatch, relationship, text, reason="The cited passage establishes this relation."):
    monkeypatch.setattr(
        ollama_verifier._LLM_RESILIENCE, "generate", lambda call: call(),
    )

    def post(_endpoint, *, json, timeout):
        import re
        evidence_id = re.search(r"^\[([^\]]+)\]$", json["prompt"], re.M).group(1)
        return {"response": __import__("json").dumps({
            "results": [{"evidence_id": evidence_id, "relationship": relationship}],
        })}

    monkeypatch.setattr(ollama_verifier._OLLAMA_SESSION, "post", post)


@pytest.mark.parametrize(("claim", "text", "relation", "expected"), [
    ("Smoking increases the risk of lung cancer.",
     "Scientific studies show that smoking causes lung cancer.", "SUPPORTED", "SUPPORTS"),
    ("Smoking increases the risk of lung cancer.",
     "Air pollution increases the risk of lung cancer.", "INSUFFICIENT", "INSUFFICIENT"),
    ("The Sun orbits Earth.", "The Earth orbits the Sun.", "CONTRADICTED", "CONTRADICTS"),
    ("The Earth orbits the Sun once every year.",
     "Earth travels around the Sun each year.", "SUPPORTED", "SUPPORTS"),
    ("The museum opened in 2018.", "The museum was inaugurated in 2018.", "SUPPORTED", "SUPPORTS"),
    ("The museum opened in 2018.", "The city library was inaugurated in 2018.", "INSUFFICIENT", "INSUFFICIENT"),
    ("A spacecraft landed on Mars in 2021.", "The spacecraft reached Mars in 2021 but remained in orbit.",
     "CONTRADICTED", "CONTRADICTS"),
])
def test_structured_semantic_relationship(monkeypatch, claim, text, relation, expected):
    _mock(monkeypatch, relation, text)
    result = EvidenceVerificationManager(timeout=1).verify(claim, [_evidence(text)])
    assert result.results[0].relationship == expected
    assert result.results[0].directly_establishes_proposition is (expected != "INSUFFICIENT")


def test_quote_must_be_present_in_evidence():
    decision = evaluate_directness("A company opened a factory.", "The company filed a permit.",
                                   "SUPPORTS", evidence_excerpt="The company opened a factory.",
                                   semantic_reason="Direct support")
    assert decision.relationship == "INSUFFICIENT"
    assert "not present" in decision.reason


def test_amount_and_year_conflicts_block_false_support():
    amount = evaluate_directness("The company paid a ₹20,000 fine.",
                                 "The company paid a ₹2,000 fine.", "SUPPORTS",
                                 evidence_excerpt="The company paid a ₹2,000 fine.",
                                 semantic_reason="The fine is stated.")
    year = evaluate_directness("The museum opened in 2018.",
                               "The museum opened in 2019.", "SUPPORTS",
                               evidence_excerpt="The museum opened in 2019.",
                               semantic_reason="The opening is stated.")
    assert amount.directness == year.directness == "CONSTRAINT_CONFLICT"
    assert not amount.accepted and not year.accepted


def test_quote_validation_tolerates_symbol_rendering_without_changing_facts():
    evidence = "The surface reached 464 °C, making this planet the hottest in the system."
    result = evaluate_directness("This planet is the hottest in the system.", evidence,
                                 "SUPPORTS", evidence_excerpt="The surface reached 464 C, making this planet the hottest in the system.",
                                 semantic_reason="The passage states the superlative directly.")
    assert result.relationship == "SUPPORTS"


def test_multisentence_passage_can_establish_one_claim(monkeypatch):
    text = "The spacecraft entered orbit in March. It landed on the surface on April 4."
    _mock(monkeypatch, "SUPPORTED", text)
    result = EvidenceVerificationManager(timeout=1).verify(
        "The spacecraft entered orbit and landed on April 4.", [_evidence(text)])
    assert result.overall_assessment == "SUPPORTED"
    assert result.results[0].direct_evidence_excerpt in text
    assert result.results[0].direct_evidence_excerpt == text


def test_prompt_uses_minimal_relationship_contract():
    prompt = ollama_verifier._build_prompt("The museum opened in 2018.", _evidence("The museum was inaugurated in 2018."))
    assert "directly states or unambiguously entails" in prompt
    assert "Do not include any other fields" in prompt
