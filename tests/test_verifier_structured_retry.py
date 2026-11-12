import json
import re

import pytest

from agents import graph
from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification import ollama_verifier
from rag_ingestion.verification.verification_manager import EvidenceVerificationManager


def _evidence(text="Evidence text."):
    return Evidence(
        evidence_id="e1", chunk_id="c1", document_id="d1", text=text,
        domain="general", source="test", language="en", metadata={},
    )


class _Response:
    status_code = 200

    def __init__(self, raw):
        self.raw = raw

    def json(self):
        return {"response": self.raw}


def _mock_outputs(monkeypatch, outputs):
    calls = []
    pending = iter(outputs)

    def post(endpoint, *, json, timeout):
        calls.append({"payload": json, "timeout": timeout})
        output = next(pending)
        if isinstance(output, dict) and "relationship" in output:
            evidence_id = re.search(r"^\[([^\]]+)\]$", json["prompt"], re.M).group(1)
            output = {"results": [{"evidence_id": evidence_id, **output}]}
        return _Response(json_module.dumps(output))

    import json as json_module
    monkeypatch.setattr(ollama_verifier._OLLAMA_SESSION, "post", post)
    return calls


def test_minimal_relationship_response_uses_small_schema_and_bounds(monkeypatch):
    calls = _mock_outputs(monkeypatch, [{"relationship": "SUPPORTED"}])
    direct_evidence = _evidence("The drug reduces blood pressure.")
    result = ollama_verifier.verify_evidence("The drug reduces blood pressure.", direct_evidence)
    assert result.relationship == "SUPPORTS"
    assert result.confidence == ollama_verifier.VERIFIER_CONFIDENCE
    assert result.claim_aligned and result.directly_establishes_proposition
    assert len(calls) == 1
    payload = calls[0]["payload"]
    assert set(payload["format"]["required"]) == {"results"}
    assert payload["format"]["additionalProperties"] is False
    assert payload["keep_alive"] == "10m"
    assert payload["options"] == {"temperature": 0, "seed": 17, "num_predict": 64}
    assert len(payload["prompt"]) < 2000
    assert "Return only JSON" in payload["prompt"]
    assert "do not include any other fields" in payload["prompt"].casefold()


def test_prompt_contains_claim_and_evidence_and_protects_event_scope():
    prompt = ollama_verifier._build_prompt(
        "The agency reduced the rate on 6 June 2025.",
        _evidence("The agency reduced the rate on 6 June 2025."),
    )
    assert "CLAIM:\nThe agency reduced the rate on 6 June 2025." in prompt
    assert "[e1]\nThe agency reduced the rate on 6 June 2025." in prompt
    assert "directly states an incompatible fact" in prompt.casefold()


@pytest.mark.parametrize("relationship", ["INSUFFICIENT", "unsupported", None])
def test_invalid_or_insufficient_relationship_never_creates_binary_result(monkeypatch, relationship):
    calls = _mock_outputs(monkeypatch, [{"relationship": relationship}])
    item = _evidence()
    if relationship == "INSUFFICIENT":
        result = ollama_verifier.verify_evidence("Claim text.", item)
        assert result.relationship == "INSUFFICIENT"
        assert result.confidence == 0.0
        assert not result.claim_aligned
        assert not result.directly_establishes_proposition
    else:
        result = ollama_verifier.verify_evidence("Claim text.", item)
        assert result.relationship == "INSUFFICIENT"
        assert result.verifier_failed
    assert len(calls) == 1


def test_malformed_json_fails_without_semantic_retry(monkeypatch):
    calls = _mock_outputs(monkeypatch, [])

    def post(endpoint, *, json, timeout):
        calls.append({"payload": json, "timeout": timeout})
        return _Response('{"relationship":')

    monkeypatch.setattr(ollama_verifier._OLLAMA_SESSION, "post", post)
    result = ollama_verifier.verify_evidence("Claim text.", _evidence())
    assert result.relationship == "INSUFFICIENT"
    assert result.verifier_failed
    assert len(calls) == 1


def test_malformed_verifier_output_becomes_item_level_failed_result(monkeypatch):
    calls = _mock_outputs(monkeypatch, [{"relationship": "SUPPORTED", "extra": "unexpected"}])
    result = EvidenceVerificationManager().verify("Claim text.", [_evidence()])
    assert len(calls) == 1
    assert len(result.results) == 1
    assert result.results[0].relationship == "INSUFFICIENT"
    assert result.results[0].verifier_failed is True
    assert result.overall_assessment == "INSUFFICIENT_EVIDENCE"
