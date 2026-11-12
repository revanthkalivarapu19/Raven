import json

import pytest
import time

from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification import ollama_verifier
from rag_ingestion.verification.verification_manager import EvidenceVerificationManager


def _evidence(evidence_id, text):
    return Evidence(
        evidence_id=evidence_id,
        chunk_id=f"chunk-{evidence_id}",
        document_id=f"doc-{evidence_id}",
        text=text,
        domain="science",
        source="test source",
        language="en",
        metadata={},
    )


class _Response:
    status_code = 200

    def __init__(self, relationship, excerpt):
        self.relationship = relationship
        self.excerpt = excerpt

    def json(self):
        return {"response": json.dumps({"relationship": self.relationship,
                                        "reason": "The passage directly addresses the claim.",
                                        "evidence_excerpt": self.excerpt,
                                        "same_event": True})}


class _BatchResponse:
    status_code = 200

    def __init__(self, rows, claim_id=None):
        self.rows = rows
        self.claim_id = claim_id

    def json(self):
        payload = {"results": self.rows}
        if self.claim_id is not None:
            payload["claim_id"] = self.claim_id
        return {"response": json.dumps(payload)}


def test_manager_verifies_one_strong_item_and_stops(monkeypatch):
    evidence = [
        _evidence("support", "Earth orbits the Sun."),
        _evidence("contradict", "The Sun orbits Earth."),
    ]
    payloads = []
    def post(endpoint, **kwargs):
        payloads.append(kwargs["json"])
        return _BatchResponse([
            {"evidence_id": "support", "relationship": "SUPPORTED"},
            {"evidence_id": "contradict", "relationship": "CONTRADICTED"},
        ])

    monkeypatch.setattr(ollama_verifier._OLLAMA_SESSION, "post", post)
    result = EvidenceVerificationManager().verify("Earth orbits the Sun.", evidence)

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["keep_alive"] == "10m"
    assert payload["options"] == {"temperature": 0, "seed": 17, "num_predict": 64}
    assert set(payload["format"]["required"]) == {"results"}
    assert payload["format"]["additionalProperties"] is False
    assert "Earth orbits the Sun." in payload["prompt"]
    assert "The Sun orbits Earth." not in payload["prompt"]
    assert [r.relationship for r in result.results] == ["SUPPORTS"]
    assert [r.confidence for r in result.results] == [0.9]
    assert result.overall_assessment == "SUPPORTED"


def test_manager_advances_only_when_first_item_is_insufficient(monkeypatch):
    evidence = [
        _evidence("topical", "A passage discusses the Sun as a star."),
        _evidence("direct", "Earth orbits the Sun once each year."),
    ]
    sent_ids = []

    def post(_endpoint, *, json, timeout):
        import re
        evidence_id = re.search(r"^\[([^\]]+)\]$", json["prompt"], re.M).group(1)
        sent_ids.append(evidence_id)
        relationship = "INSUFFICIENT" if evidence_id == "topical" else "SUPPORTED"
        return _BatchResponse([{"evidence_id": evidence_id, "relationship": relationship}])

    monkeypatch.setattr(ollama_verifier._OLLAMA_SESSION, "post", post)
    result = EvidenceVerificationManager().verify("The Earth orbits the Sun once each year.", evidence)
    assert sent_ids == ["topical", "direct"]
    assert [item.evidence_id for item in result.results] == sent_ids
    assert result.overall_assessment == "SUPPORTED"


def test_incomplete_batch_fails_only_the_omitted_item_and_keeps_success(monkeypatch):
    evidence = [
        _evidence("verified", "Earth orbits the Sun."),
        _evidence("omitted", "The Sun is a star at the center of the Solar System."),
    ]

    def post(_endpoint, **_kwargs):
        return _BatchResponse([{
            "evidence_id": "verified",
            "relationship": "SUPPORTED",
        }])

    monkeypatch.setattr(ollama_verifier._OLLAMA_SESSION, "post", post)
    result = ollama_verifier.verify_evidence_batch("Earth orbits the Sun.", evidence)

    assert [item.evidence_id for item in result] == ["verified", "omitted"]
    assert result[0].relationship == "SUPPORTS"
    assert result[0].verifier_failed is False
    assert result[1].relationship == "INSUFFICIENT"
    assert result[1].verifier_failed is True


def test_duplicate_id_fails_only_that_item_and_preserves_other_result(monkeypatch):
    evidence = [
        _evidence("duplicate", "Earth orbits the Sun."),
        _evidence("valid", "Earth completes one orbit around the Sun each year."),
    ]

    def post(_endpoint, **_kwargs):
        return _BatchResponse([
            {"evidence_id": "duplicate", "relationship": "SUPPORTED"},
            {"evidence_id": "duplicate", "relationship": "CONTRADICTED"},
            {"evidence_id": "unprovided", "relationship": "SUPPORTED"},
            {"evidence_id": "valid", "relationship": "SUPPORTED"},
        ])

    monkeypatch.setattr(ollama_verifier._OLLAMA_SESSION, "post", post)
    results = ollama_verifier.verify_evidence_batch("Earth orbits the Sun.", evidence)

    assert [result.evidence_id for result in results] == ["duplicate", "valid"]
    assert results[0].verifier_failed is True
    assert results[0].relationship == "INSUFFICIENT"
    assert results[1].verifier_failed is False
    assert results[1].relationship == "SUPPORTS"


def test_unrequested_claim_id_field_fails_closed(monkeypatch):
    evidence = [_evidence("one", "Earth orbits the Sun."), _evidence("two", "Earth completes an orbit around the Sun.")]

    def post(_endpoint, **_kwargs):
        return _BatchResponse([
            {"evidence_id": item.evidence_id, "relationship": "SUPPORTED"}
            for item in evidence
        ], claim_id="different-claim")

    monkeypatch.setattr(ollama_verifier._OLLAMA_SESSION, "post", post)
    result = ollama_verifier.verify_evidence_batch("Earth orbits the Sun.", evidence[:1])
    assert len(result) == 1
    assert result[0].verifier_failed and result[0].relationship == "INSUFFICIENT"


def test_llm_timeout_returns_one_failed_result_per_evidence(monkeypatch):
    evidence = [_evidence("one", "Earth orbits the Sun."), _evidence("two", "The Earth travels around the Sun.")]

    def timeout(_call):
        raise ollama_verifier.LLMInvocationError("request timed out", "timeout", retryable=True)

    monkeypatch.setattr(ollama_verifier._LLM_RESILIENCE, "generate", timeout)
    result = ollama_verifier.verify_evidence_batch("Earth orbits the Sun.", evidence)
    assert [item.evidence_id for item in result] == ["one", "two"]
    assert all(item.relationship == "INSUFFICIENT" and item.verifier_failed for item in result)


@pytest.mark.parametrize("raw", ["{", '{"relationship":"SUPPORTED","extra":true}'])
def test_malformed_single_item_output_fails_closed_without_retry(monkeypatch, raw):
    item = _evidence("one", "Earth orbits the Sun.")
    calls = []

    class MalformedResponse:
        status_code = 200

        def json(self):
            return {"response": raw}

    def post(*_args, **_kwargs):
        calls.append(True)
        return MalformedResponse()

    monkeypatch.setattr(ollama_verifier._OLLAMA_SESSION, "post", post)
    results = ollama_verifier.verify_evidence_batch("Earth orbits the Sun.", [item])
    assert len(results) == 1
    assert results[0].evidence_id == item.evidence_id
    assert results[0].relationship == "INSUFFICIENT"
    assert results[0].verifier_failed is True
    assert len(calls) == 1


def test_verifier_http_timeout_is_bounded_by_active_request_deadline(monkeypatch):
    item = _evidence("deadline", "Earth orbits the Sun.")
    timeouts = []

    def post(_endpoint, *, json, timeout):
        timeouts.append(timeout)
        return _BatchResponse([{
            "evidence_id": item.evidence_id, "relationship": "SUPPORTED",
        }])

    monkeypatch.setattr(ollama_verifier._OLLAMA_SESSION, "post", post)
    from rag_ingestion.llm_resilience import request_deadline
    with request_deadline(time.monotonic() + 0.05):
        ollama_verifier.verify_evidence_batch("Earth orbits the Sun.", [item])
    assert len(timeouts) == 1
    assert 0 < timeouts[0] <= 0.051


def test_passage_selection_prefers_single_direct_sentence_over_adjacent_context():
    text = (
        "Diurnal motion describes the apparent daily path of celestial objects. "
        "It is caused by Earth's rotation around its axis. "
        "Because Earth orbits the Sun once a year, sidereal time changes."
    )

    passage = ollama_verifier._select_passage(
        "The Earth orbits the Sun once every year.", text,
    )

    assert passage == "Because Earth orbits the Sun once a year, sidereal time changes."
