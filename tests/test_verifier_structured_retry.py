import json

import pytest

from agents import graph
from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification import ollama_verifier
from rag_ingestion.verification.verification_manager import EvidenceVerificationManager


def _evidence():
    return Evidence(evidence_id="e1", chunk_id="c1", document_id="d1", text="Evidence text.",
                    domain="general", source="test", language="en", metadata={})


def _valid():
    return {
        "relationship": "SUPPORTS", "confidence": 0.9,
        "reasoning": "The evidence directly supports the claim.",
        "same_event": True, "claim_aligned": True,
        "directly_establishes_proposition": True,
    }


def _mock_outputs(monkeypatch, outputs):
    calls = []
    pending = iter(outputs)

    class Response:
        status_code = 200

        def __init__(self, output):
            self.output = output

        def json(self):
            return {"response": json.dumps(self.output)}

    def post(_endpoint, *, json, timeout):
        calls.append({"payload": json, "timeout": timeout})
        return Response(next(pending))

    monkeypatch.setattr(ollama_verifier.requests, "post", post)
    return calls


def test_complete_response_is_accepted_with_schema_constraint(monkeypatch):
    calls = _mock_outputs(monkeypatch, [_valid()])
    result = ollama_verifier.verify_evidence("Claim text.", _evidence())
    assert result.relationship == "SUPPORTS"
    assert len(calls) == 1
    schema = calls[0]["payload"]["format"]
    assert isinstance(schema, dict)
    assert set(schema["required"]) == {
        "relationship", "confidence", "reasoning", "same_event",
        "claim_aligned", "directly_establishes_proposition",
    }
    assert schema["properties"]["same_event"]["type"] == "boolean"
    assert schema["properties"]["claim_aligned"]["type"] == "boolean"
    assert schema["properties"]["directly_establishes_proposition"]["type"] == "boolean"


def test_missing_reasoning_remains_accepted_without_retry(monkeypatch):
    result_data = _valid()
    del result_data["reasoning"]
    calls = _mock_outputs(monkeypatch, [result_data])
    result = ollama_verifier.verify_evidence("Claim text.", _evidence())
    assert result.reasoning == "No reasoning text was returned by the verifier model."
    assert len(calls) == 1


@pytest.mark.parametrize(
    "field", ["same_event", "claim_aligned", "directly_establishes_proposition"]
)
def test_missing_decision_field_gets_one_structural_repair(monkeypatch, field):
    malformed = _valid()
    del malformed[field]
    calls = _mock_outputs(monkeypatch, [malformed, _valid()])
    result = ollama_verifier.verify_evidence("Claim text.", _evidence())
    assert getattr(result, field) is True
    assert len(calls) == 2
    repair = calls[1]["payload"]["prompt"]
    assert "STRUCTURE REPAIR ONLY" in repair
    assert "Preserve the same claim and evidence" in repair
    assert "Do not change the semantic decision" in repair
    assert calls[1]["payload"]["format"] == calls[0]["payload"]["format"]


@pytest.mark.parametrize("value", ["true", "false"])
def test_string_boolean_is_repaired_not_coerced(monkeypatch, value):
    malformed = _valid()
    malformed["same_event"] = value
    calls = _mock_outputs(monkeypatch, [malformed, _valid()])
    result = ollama_verifier.verify_evidence("Claim text.", _evidence())
    assert result.same_event is True
    assert len(calls) == 2


def test_second_malformed_result_becomes_safe_verification_failure(monkeypatch):
    malformed = _valid()
    del malformed["directly_establishes_proposition"]
    calls = _mock_outputs(monkeypatch, [malformed, malformed])

    class Pipeline:
        def process(self, **kwargs):
            EvidenceVerificationManager().verify(kwargs["claim"], [_evidence()])

    monkeypatch.setattr(graph, "get_claim_processing_pipeline", lambda: Pipeline())
    state = {
        "claim": "Claim text.", "domain": {"domain": "General"},
        "claim_scope": "GENERAL", "errors": [],
    }
    failed = graph.evidence_verification_node(state)
    assert len(calls) == 2
    assert failed["verification_result"] is None
    assert "directly_establishes_proposition" in failed["verification_failure"].message
    final = graph.supervisor_node({**state, **failed})
    assert final["supervisor_result"]["final_verdict"] == "Unverified"
    assert final["supervisor_result"]["confidence_score"] == 0


def test_repair_cannot_change_already_returned_semantic_decision(monkeypatch):
    malformed = _valid()
    del malformed["claim_aligned"]
    changed = _valid()
    changed["relationship"] = "CONTRADICTS"
    calls = _mock_outputs(monkeypatch, [malformed, changed])
    with pytest.raises(ValueError, match="changed existing decision field 'relationship'"):
        ollama_verifier.verify_evidence("Claim text.", _evidence())
    assert len(calls) == 2

