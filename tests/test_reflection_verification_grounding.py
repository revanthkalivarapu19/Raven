from agents.reflection.reflection_agent import ReflectionAgent
from rag_ingestion.verification.verification_schema import OverallVerification, VerificationResult


def _verification(claim, items, assessment):
    return OverallVerification(
        claim=claim, results=items, overall_assessment=assessment,
        overall_confidence=0.8,
    )


def _item(claim, relationship, same_event=True):
    return VerificationResult(
        evidence_id="e1", claim=claim, relationship=relationship,
        confidence=0.9, reasoning="test", verifier="test", same_event=same_event,
        claim_aligned=same_event,
    )


def _reflect(monkeypatch, verification):
    monkeypatch.setattr(
        "agents.reflection.reflection_agent.ollama.chat",
        lambda **kwargs: {"message": {"content": '{"decision":"YES","is_evidence_sufficient":true,"contradictions_found":false,"confidence":0.9,"notes":"Multiple reputable sources confirm the claim.","support_score":0.7,"source_credibility_score":0.9,"evidence_quality_score":0.9,"claim_supported":true,"claim_contradicted":false,"evidence_items_contradict_each_other":false}'}},
    )
    return ReflectionAgent().reflect(
        claim_text=verification.claim, domain="general", evidence_list=[{"text": "evidence"}],
        verification_result=verification,
    )


def test_insufficient_verification_rejects_hallucinated_support(monkeypatch):
    claim = "A claim."
    result = _reflect(monkeypatch, _verification(claim, [], "INSUFFICIENT_EVIDENCE"))
    assert result["claim_supported"] is False
    assert result["is_evidence_sufficient"] is False
    assert result["support_score"] == 0.0
    assert "insufficient" in result["notes"].lower()


def test_support_is_grounded(monkeypatch):
    claim = "A claim."
    result = _reflect(monkeypatch, _verification(claim, [_item(claim, "SUPPORTS")], "SUPPORTED"))
    assert result["claim_supported"] is True


def test_scientific_support_is_grounded_without_same_event(monkeypatch):
    claim = "N348I mutations cause zidovudine resistance."
    item = _item(claim, "SUPPORTS", same_event=False)
    item.claim_aligned = True
    result = _reflect(monkeypatch, _verification(claim, [item], "SUPPORTED"))
    assert result["claim_supported"] is True


def test_support_plus_insufficient_is_supported(monkeypatch):
    claim = "A claim."
    result = _reflect(monkeypatch, _verification(claim, [_item(claim, "SUPPORTS"), _item(claim, "INSUFFICIENT")], "SUPPORTED"))
    assert result["claim_supported"] is True


def test_same_event_contradiction_is_grounded(monkeypatch):
    claim = "A claim."
    result = _reflect(monkeypatch, _verification(claim, [_item(claim, "CONTRADICTS")], "CONTRADICTED"))
    assert result["claim_contradicted"] is True
    assert result["claim_supported"] is False


def test_mismatched_contradiction_is_insufficient(monkeypatch):
    claim = "A claim."
    result = _reflect(monkeypatch, _verification(claim, [_item(claim, "CONTRADICTS", False)], "INSUFFICIENT_EVIDENCE"))
    assert result["claim_contradicted"] is False
    assert result["claim_supported"] is False
    assert result["is_evidence_sufficient"] is False


def test_structured_support_and_contradiction_force_evidence_conflict(monkeypatch):
    claim = "The policy rate was reduced."
    support = _item(claim, "SUPPORTS")
    support.evidence_id = "support"
    contradiction = _item(claim, "CONTRADICTS")
    contradiction.evidence_id = "contradiction"
    verification = _verification(claim, [support, contradiction], "CONFLICTING_EVIDENCE")
    monkeypatch.setattr(
        "agents.reflection.reflection_agent.ollama.chat",
        lambda **kwargs: {"message": {"content": '{"decision":"YES","confidence":0.9,"notes":"Both items address the claim.","claim_supported":true,"claim_contradicted":false,"evidence_items_contradict_each_other":false}'}}
    )
    result = ReflectionAgent().reflect(
        claim_text=claim, domain="Finance",
        evidence_list=[{"evidence_id": "support", "text": "support"}, {"evidence_id": "contradiction", "text": "contradiction"}],
        verification_result=verification,
    )
    assert result["evidence_items_contradict_each_other"] is True
    assert result["contradictions_found"] is True
    assert result["decision"] == "NO"
    assert result["is_evidence_sufficient"] is False


def test_structured_different_event_does_not_force_conflict(monkeypatch):
    claim = "The policy rate was reduced."
    support = _item(claim, "SUPPORTS")
    support.evidence_id = "support"
    contradiction = _item(claim, "CONTRADICTS", same_event=False)
    contradiction.evidence_id = "other-event"
    contradiction.claim_aligned = True
    verification = _verification(claim, [support, contradiction], "CONFLICTING_EVIDENCE")
    monkeypatch.setattr(
        "agents.reflection.reflection_agent.ollama.chat",
        lambda **kwargs: {"message": {"content": '{"decision":"NO","confidence":0.4,"notes":"Different events.","evidence_items_contradict_each_other":true}'}}
    )
    result = ReflectionAgent().reflect(
        claim_text=claim, domain="Finance",
        evidence_list=[{"evidence_id": "support", "text": "support"}, {"evidence_id": "other-event", "text": "other event"}],
        verification_result=verification,
    )
    assert result["evidence_items_contradict_each_other"] is False
    assert result["contradictions_found"] is False


def test_scientific_conflict_does_not_require_a_dated_event(monkeypatch):
    claim = "The treatment reduces symptoms."
    support = _item(claim, "SUPPORTS", same_event=False)
    support.evidence_id = "support"
    support.claim_aligned = True
    contradiction = _item(claim, "CONTRADICTS", same_event=False)
    contradiction.evidence_id = "contradiction"
    contradiction.claim_aligned = True
    verification = _verification(claim, [support, contradiction], "CONFLICTING_EVIDENCE")
    monkeypatch.setattr(
        "agents.reflection.reflection_agent.ollama.chat",
        lambda **kwargs: {"message": {"content": '{"decision":"YES","confidence":0.8,"notes":"Both studies address the claim.","evidence_items_contradict_each_other":false}'}}
    )
    result = ReflectionAgent().reflect(
        claim_text=claim, domain="Medical",
        evidence_list=[{"evidence_id": "support", "text": "support"}, {"evidence_id": "contradiction", "text": "contradiction"}],
        verification_result=verification,
    )
    assert result["evidence_items_contradict_each_other"] is True
    assert result["decision"] == "NO"


def test_insufficient_assessment_does_not_force_conflict(monkeypatch):
    claim = "The policy rate was reduced."
    support = _item(claim, "SUPPORTS")
    support.evidence_id = "support"
    contradiction = _item(claim, "CONTRADICTS")
    contradiction.evidence_id = "contradiction"
    verification = _verification(claim, [support, contradiction], "INSUFFICIENT_EVIDENCE")
    monkeypatch.setattr(
        "agents.reflection.reflection_agent.ollama.chat",
        lambda **kwargs: {"message": {"content": '{"decision":"YES","confidence":0.8,"notes":"Model claims a conflict.","evidence_items_contradict_each_other":true}'}}
    )
    result = ReflectionAgent().reflect(
        claim_text=claim, domain="Finance",
        evidence_list=[{"evidence_id": "support", "text": "support"}, {"evidence_id": "contradiction", "text": "contradiction"}],
        verification_result=verification,
    )
    assert result["evidence_items_contradict_each_other"] is False
    assert result["is_evidence_sufficient"] is False
