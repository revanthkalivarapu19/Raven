import pytest

from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification import verification_manager
from rag_ingestion.verification.claim_scope import classify_claim_scope
from rag_ingestion.verification.verification_schema import VerificationResult


CLAIM = "The Reserve Bank of India reduced the repo rate by 0.50% on Friday to boost economic growth."


def _evidence(text: str) -> Evidence:
    return Evidence(
        evidence_id=f"e{sum(ord(char) for char in text)}", chunk_id="chunk", document_id="doc", text=text,
        domain="finance", source="test", language="en", metadata={}
    )


def _manager(
    monkeypatch,
    relationship: str,
    same_event: bool = True,
    claim_aligned: bool | None = None,
):
    def mocked_verify(claim, evidence_list, **kwargs):
        return [VerificationResult(
            evidence_id=evidence.evidence_id, claim=claim,
            relationship=relationship, confidence=0.9,
            reasoning="mocked evidence-based classification", verifier="test",
            same_event=same_event,
            claim_aligned=(
                same_event if claim_aligned is None else claim_aligned
            ),
            directly_establishes_proposition=relationship in {"SUPPORTS", "CONTRADICTS"},
        ) for evidence in evidence_list]
    monkeypatch.setattr(verification_manager, "verify_evidence_batch", mocked_verify)
    return verification_manager.EvidenceVerificationManager()


def test_unrelated_evidence_is_insufficient(monkeypatch):
    result = _manager(monkeypatch, "INSUFFICIENT").verify(
        CLAIM, [_evidence("The World Bank published unrelated development data.")]
    )
    assert result.overall_assessment == "INSUFFICIENT_EVIDENCE"


def test_different_date_event_is_insufficient(monkeypatch):
    result = _manager(monkeypatch, "INSUFFICIENT").verify(
        CLAIM, [_evidence("In October 2026, the repo rate was increased.")]
    )
    assert result.overall_assessment == "INSUFFICIENT_EVIDENCE"


def test_different_financial_instrument_is_insufficient(monkeypatch):
    result = _manager(monkeypatch, "INSUFFICIENT").verify(
        CLAIM, [_evidence("HDFC changed its MCLR.")]
    )
    assert result.overall_assessment == "INSUFFICIENT_EVIDENCE"


def test_same_event_opposite_fact_is_contradicted(monkeypatch):
    result = _manager(monkeypatch, "CONTRADICTS").verify(
        CLAIM, [_evidence("On the same Friday, RBI kept the policy repo rate unchanged.")]
    )
    assert result.overall_assessment == "CONTRADICTED"


def test_direct_support_is_supported(monkeypatch):
    result = _manager(monkeypatch, "SUPPORTS").verify(
        CLAIM, [_evidence("RBI reduced the repo rate by 50 basis points on Friday.")]
    )
    assert result.overall_assessment == "SUPPORTED"


def test_rbi_different_date_cannot_be_contradiction(monkeypatch):
    result = _manager(monkeypatch, "CONTRADICTS", same_event=False).verify(
        CLAIM, [_evidence("In October 2026, RBI increased the repo rate.")]
    )
    assert result.overall_assessment != "CONTRADICTED"


def test_multiple_temporally_mismatched_contradictions_are_insufficient(monkeypatch):
    result = _manager(monkeypatch, "CONTRADICTS", same_event=False).verify(
        CLAIM, [_evidence("A later increase occurred."), _evidence("An earlier increase occurred.")]
    )
    assert result.overall_assessment == "INSUFFICIENT_EVIDENCE"


def test_support_plus_unrelated_opposite_article_is_not_contradicted(monkeypatch):
    calls = iter([("SUPPORTS", True), ("CONTRADICTS", False)])

    def mocked_verify(claim, evidence_list, **kwargs):
        results = []
        for evidence in evidence_list:
            relationship, same_event = next(calls)
            results.append(VerificationResult(
                evidence_id=evidence.evidence_id, claim=claim,
                relationship=relationship, confidence=0.9,
                reasoning="mocked evidence-based classification", verifier="test",
                same_event=same_event,
                claim_aligned=same_event,
                directly_establishes_proposition=relationship in {"SUPPORTS", "CONTRADICTS"},
            ))
        return results

    monkeypatch.setattr(verification_manager, "verify_evidence_batch", mocked_verify)
    result = verification_manager.EvidenceVerificationManager().verify(
        CLAIM, [_evidence("RBI reduced the repo rate by 50 basis points."),
                _evidence("A later event increased a rate.")]
    )
    assert result.overall_assessment == "SUPPORTED"


def test_prompt_requires_same_event_context_for_contradiction():
    from rag_ingestion.verification.ollama_verifier import (
        _VERIFIER_OUTPUT_SCHEMA,
        _build_prompt,
    )

    evidence = _evidence("unrelated text")
    text = _build_prompt(CLAIM, evidence)
    assert f"CLAIM:\n{CLAIM}" in text
    assert f"[{evidence.evidence_id}]\n{evidence.text}" in text
    assert "SUPPORTED|CONTRADICTED|INSUFFICIENT" in text
    assert "incompatible fact" in text
    assert "Return only JSON" in text
    assert "Do not include any other fields" in text
    assert len(text) < 2000
    assert set(_VERIFIER_OUTPUT_SCHEMA["required"]) == {"results"}
    assert _VERIFIER_OUTPUT_SCHEMA["additionalProperties"] is False


@pytest.mark.parametrize(
    ("claim", "evidence_text", "relationship", "same_event", "claim_aligned", "expected"),
    [
        (
            "Ultrasound guidance significantly raises traumatic procedures.",
            "Ultrasound imaging significantly reduced traumatic procedures.",
            "CONTRADICTS",
            True,
            True,
            "INSUFFICIENT_EVIDENCE",
        ),
        (
            "AMPK activation increases inflammation-related fibrosis in the lungs.",
            "AMPK activation is associated with lower fibrotic activity and accelerated fibrosis resolution.",
            "CONTRADICTS",
            True,
            True,
            "INSUFFICIENT_EVIDENCE",
        ),
        (
            "The PRR MDA5 is a sensor of RNA virus infection.",
            "The RIG-I-like receptors RIG-I, MDA5, and LGP2 play a major role in pathogen sensing of RNA virus infection.",
            "SUPPORTS",
            True,
            True,
            "SUPPORTED",
        ),
        (
            "N348I mutations cause resistance to zidovudine (AZT).",
            "N348I provides supporting evidence for a role in zidovudine resistance.",
            "SUPPORTS",
            True,
            True,
            "INSUFFICIENT_EVIDENCE",
        ),
        (
            "Alizarin forms hydrogen bonds with PGAM1 substrate-binding residues.",
            "PGAM1 phosphorylation affects substrate binding, but Alizarin and hydrogen bonds are not mentioned.",
            "INSUFFICIENT",
            False,
            False,
            "INSUFFICIENT_EVIDENCE",
        ),
        (
            "The agency reduced its policy rate on 6 June 2025.",
            "The agency increased its policy rate on 12 October 2026.",
            "CONTRADICTS",
            False,
            False,
            "INSUFFICIENT_EVIDENCE",
        ),
    ],
)
def test_verifier_polarity_and_alignment_regressions(
    monkeypatch, claim, evidence_text, relationship, same_event, claim_aligned, expected
):
    import json
    import re

    from rag_ingestion.verification import ollama_verifier

    prompts = []

    def generate(call):
        return call()

    def post(_endpoint, *, json, timeout):
        prompts.append(json["prompt"])
        model_relationship = (
            "INSUFFICIENT" if expected == "INSUFFICIENT_EVIDENCE" else {
                "SUPPORTS": "SUPPORTED",
                "CONTRADICTS": "CONTRADICTED",
                "INSUFFICIENT": "INSUFFICIENT",
            }[relationship]
        )
        evidence_id = re.search(r"^\[([^\]]+)\]$", json["prompt"], re.M).group(1)
        return {"response": json_module.dumps({
            "results": [{"evidence_id": evidence_id, "relationship": model_relationship}],
        })}

    json_module = json
    monkeypatch.setattr(ollama_verifier._LLM_RESILIENCE, "generate", generate)
    monkeypatch.setattr(ollama_verifier._OLLAMA_SESSION, "post", post)
    result = verification_manager.EvidenceVerificationManager(timeout=1).verify(
        claim,
        [_evidence(evidence_text)],
    )

    assert result.overall_assessment == expected
    assert len(prompts) == 1
    assert f"CLAIM:\n{claim}" in prompts[0]
    assert f"]\n{evidence_text}" in prompts[0]
    assert "SUPPORTED|CONTRADICTED|INSUFFICIENT" in prompts[0]
    assert "Return only JSON" in prompts[0]
    assert "Do not include any other fields" in prompts[0]


@pytest.mark.parametrize(
    ("claim", "evidence_text", "relationship", "same_event", "claim_aligned", "expected"),
    [
        (
            "N348I mutations cause zidovudine resistance.",
            "N348I contributes to zidovudine resistance.",
            "SUPPORTS",
            False,
            True,
            "SUPPORTED",
        ),
        (
            "AMPK activation increases lung fibrosis.",
            "AMPK activation lowers fibrotic activity and accelerates fibrosis resolution.",
            "CONTRADICTS",
            False,
            True,
            "CONTRADICTED",
        ),
        (
            "RBI reduced the repo rate by 0.50% on Friday.",
            "RBI reduced the repo rate by 0.50% on the same Friday.",
            "SUPPORTS",
            True,
            True,
            "SUPPORTED",
        ),
        (
            "RBI reduced the repo rate by 0.50% on Friday.",
            "RBI increased the repo rate the following month.",
            "CONTRADICTS",
            False,
            False,
            "INSUFFICIENT_EVIDENCE",
        ),
        (
            "N348I mutations cause zidovudine resistance.",
            "The World Bank released unrelated development data.",
            "INSUFFICIENT",
            False,
            False,
            "INSUFFICIENT_EVIDENCE",
        ),
    ],
)
def test_aggregation_uses_claim_alignment_with_event_safeguards(
    monkeypatch,
    claim,
    evidence_text,
    relationship,
    same_event,
    claim_aligned,
    expected,
):
    manager = _manager(
        monkeypatch,
        relationship,
        same_event=same_event,
        claim_aligned=claim_aligned,
    )

    result = manager.verify(claim, [_evidence(evidence_text)])

    assert result.results[0].same_event is same_event
    assert result.results[0].claim_aligned is claim_aligned
    assert result.overall_assessment == expected
