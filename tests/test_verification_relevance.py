import pytest

from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification import verification_manager
from rag_ingestion.verification.verification_schema import VerificationResult


CLAIM = "The Reserve Bank of India reduced the repo rate by 0.50% on Friday to boost economic growth."


def _evidence(text: str) -> Evidence:
    return Evidence(
        evidence_id="e1", chunk_id="chunk", document_id="doc", text=text,
        domain="finance", source="test", language="en", metadata={}
    )


def _manager(
    monkeypatch,
    relationship: str,
    same_event: bool = True,
    claim_aligned: bool | None = None,
):
    def mocked_verify(claim, evidence, **kwargs):
        return VerificationResult(
            evidence_id=evidence.evidence_id, claim=claim,
            relationship=relationship, confidence=0.9,
            reasoning="mocked evidence-based classification", verifier="test",
            same_event=same_event,
            claim_aligned=(
                same_event if claim_aligned is None else claim_aligned
            ),
            directly_establishes_proposition=relationship in {"SUPPORTS", "CONTRADICTS"},
        )
    monkeypatch.setattr(verification_manager, "verify_evidence", mocked_verify)
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

    def mocked_verify(claim, evidence, **kwargs):
        relationship, same_event = next(calls)
        return VerificationResult(
            evidence_id=evidence.evidence_id, claim=claim,
            relationship=relationship, confidence=0.9,
            reasoning="mocked evidence-based classification", verifier="test",
            same_event=same_event,
            claim_aligned=same_event,
            directly_establishes_proposition=relationship in {"SUPPORTS", "CONTRADICTS"},
        )

    monkeypatch.setattr(verification_manager, "verify_evidence", mocked_verify)
    result = verification_manager.EvidenceVerificationManager().verify(
        CLAIM, [_evidence("RBI reduced the repo rate by 50 basis points."),
                _evidence("A later event increased a rate.")]
    )
    assert result.overall_assessment == "SUPPORTED"


def test_prompt_requires_same_event_context_for_contradiction():
    from rag_ingestion.verification.ollama_verifier import _build_prompt

    text = _build_prompt(CLAIM, _evidence("unrelated text"))
    assert "SUPPORTS means the evidence affirms the claim's proposition" in text
    assert "CONTRADICTS means evidence establishes the opposite proposition" in text
    assert "Claim: X increases Y. Evidence: X decreases Y. Result: CONTRADICTS." in text
    assert "Lack of support alone MUST NOT produce CONTRADICTS" in text
    assert (
        "Claim: X causes Y. Evidence mentions X but does not establish causation. "
        "Result: INSUFFICIENT."
    ) in text
    assert "INSUFFICIENT means relevant evidence establishes neither side" in text
    assert "Reason about meaning, not exact wording" in text
    assert "1/2000 = 0.0005 = 500 per million when these values describe the same quantity and context" in text
    assert "blocking X that reduces Y means active X promotes Y" in text
    assert "If the claim says \"reduced\", do not reinterpret it as \"increased\"" in text
    assert "Preserve the exact direction of the supplied claim" in text
    assert "Use ONLY the CLAIM and EVIDENCE supplied in this request." in text
    assert "Do not import entities, facts, numbers, or conclusions from instruction examples" in text
    assert "Do apply their demonstrated directness and polarity comparison" in text
    assert "Before producing the relationship, copy the exact subject and action" in text
    assert "do not invent another entity from the instructions" in text
    assert "treat the reasoning as invalid and reassess before returning the result" in text
    assert "If your reasoning contains an entity, number, event, or relationship" in text
    assert "Do NOT output chain-of-thought or step-by-step analysis" in text
    assert "Generic calibration examples" in text
    assert "CX3CR1" not in text
    assert "MDA5" not in text
    assert "PPM1D" not in text
    assert "ApoE4" not in text
    assert "For compound claims, evaluate each asserted clause separately" in text
    assert "a qualifier or independence statement about one relationship does not negate another independently supported clause" in text
    assert "claim_aligned" in text
    assert "For event/news claims" in text
    assert "same_event is separate from claim_aligned" in text
    assert "same_event=true only for the same specific event" in text
    assert "same_event=false is normal and does NOT imply claim_aligned=false" in text
    assert "claim_aligned asks whether the evidence evaluates the claim proposition" in text
    assert "a different event must not support or contradict the claim" in text
    assert "Do not infer contradiction from failure to support" in text


@pytest.mark.parametrize(
    ("claim", "evidence_text", "relationship", "same_event", "expected"),
    [
        (
            "Ultrasound guidance significantly raises traumatic procedures.",
            "Ultrasound imaging significantly reduced traumatic procedures.",
            "CONTRADICTS",
            True,
            "CONTRADICTED",
        ),
        (
            "AMPK activation increases inflammation-related fibrosis in the lungs.",
            "AMPK activation is associated with lower fibrotic activity and accelerated fibrosis resolution.",
            "CONTRADICTS",
            True,
            "CONTRADICTED",
        ),
        (
            "The PRR MDA5 is a sensor of RNA virus infection.",
            "The RIG-I-like receptors RIG-I, MDA5, and LGP2 play a major role in pathogen sensing of RNA virus infection.",
            "SUPPORTS",
            True,
            "SUPPORTED",
        ),
        (
            "N348I mutations cause resistance to zidovudine (AZT).",
            "N348I provides supporting evidence for a role in zidovudine resistance.",
            "SUPPORTS",
            True,
            "SUPPORTED",
        ),
        (
            "Alizarin forms hydrogen bonds with PGAM1 substrate-binding residues.",
            "PGAM1 phosphorylation affects substrate binding, but Alizarin and hydrogen bonds are not mentioned.",
            "INSUFFICIENT",
            False,
            "INSUFFICIENT_EVIDENCE",
        ),
        (
            "The agency reduced its policy rate on 6 June 2025.",
            "The agency increased its policy rate on 12 October 2026.",
            "CONTRADICTS",
            False,
            "INSUFFICIENT_EVIDENCE",
        ),
    ],
)
def test_verifier_polarity_and_alignment_regressions(
    monkeypatch, claim, evidence_text, relationship, same_event, expected
):
    import json

    from rag_ingestion.verification import ollama_verifier

    prompts = []

    def generate(call):
        return call()

    def post(_endpoint, *, json, timeout):
        prompts.append(json["prompt"])
        return {
            "response": json_module.dumps({
                "relationship": relationship,
                "confidence": 0.9,
                "reasoning": "Deterministic expected classification fixture.",
                "same_event": same_event,
                "claim_aligned": same_event,
                "directly_establishes_proposition": relationship in {"SUPPORTS", "CONTRADICTS"},
            })
        }

    json_module = json
    monkeypatch.setattr(ollama_verifier._LLM_RESILIENCE, "generate", generate)
    monkeypatch.setattr(ollama_verifier.requests, "post", post)
    result = verification_manager.EvidenceVerificationManager(timeout=1).verify(
        claim,
        [_evidence(evidence_text)],
    )

    assert result.overall_assessment == expected
    assert len(prompts) == 1
    assert "SUPPORTS means the evidence affirms the claim's proposition" in prompts[0]
    assert "CONTRADICTS means evidence establishes the opposite proposition" in prompts[0]
    assert "For scientific/general claims" in prompts[0]


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
