import base64
import os
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import main
from agents import graph
from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification.verification_schema import OverallVerification, VerificationResult


client = TestClient(main.app)


def _state(verdict="Real", *, evidence=True):
    items = []
    fused = []
    if evidence:
        fused = [Evidence(
            evidence_id="e1", chunk_id="c1", document_id="d1", text="Source excerpt.",
            domain="general", source="Example source", language="en", metadata={},
            similarity_score=0.84, authority_score=8.7,
        )]
        items = [VerificationResult(
            evidence_id="e1", claim="The claim.", relationship="SUPPORTS",
            confidence=0.9, reasoning="The source directly supports the claim.",
            verifier="test", same_event=True, claim_aligned=True,
            directly_establishes_proposition=True,
        )]
    verification = OverallVerification(
        claim="The claim.", results=items,
        overall_assessment="SUPPORTED" if verdict == "Real" else "INSUFFICIENT_EVIDENCE",
        overall_confidence=0.9 if verdict == "Real" else 0,
    )
    confidence = 0.87 if verdict == "Real" else 0.8
    return {
        "claim": "The claim.", "claim_scope": "GENERAL",
        "verification_result": verification, "fused_evidence": fused,
        "supervisor_result": {
            "final_verdict": verdict, "confidence_score": confidence * 100,
            "reasoning_summary": "The available evidence was reviewed.",
        },
        "final_output": {
            "final_verdict": verdict, "confidence": confidence,
            "explanation": "The available evidence was reviewed.",
        },
    }


def test_health_endpoint(monkeypatch):
    from unittest.mock import Mock
    import requests

    response_obj = Mock()
    response_obj.json.return_value = {"models": [{"name": "qwen2.5:3b"}]}
    response_obj.raise_for_status.return_value = None
    monkeypatch.setattr(requests, "get", Mock(return_value=response_obj))
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "unavailable",
        "worker_ready": False,
        "bge_ready": False,
        "bge_fallback_active": False,
        "worker_restart_count": main._warm_worker.restarts,
        "verification_ready": False,
        "worker": {
            "ready": False, "pid": None, "startup_seconds": None,
            "restart_count": main._warm_worker.restarts,
            "last_startup_error": main._warm_worker.last_startup_error,
        },
        "semantic_ranking": {"bge_ready": False, "fallback_active": False, "error": None},
        "ollama": {
            "reachable": True, "model": "qwen2.5:3b",
            "model_available": True, "ready": True,
        },
    }


def test_health_endpoint_reports_bge_fallback_as_degraded(monkeypatch):
    from unittest.mock import Mock
    import requests

    response_obj = Mock()
    response_obj.json.return_value = {"models": [{"name": "qwen2.5:3b"}]}
    response_obj.raise_for_status.return_value = None
    monkeypatch.setattr(requests, "get", Mock(return_value=response_obj))
    monkeypatch.setattr(type(main._warm_worker), "is_ready", property(lambda _self: True))
    monkeypatch.setattr(main._warm_worker, "ready", {
        "pid": 234, "worker_startup_seconds": 9.2,
        "bge_ready": False, "bge_error": "allocation failure",
    })

    payload = client.get("/api/health").json()

    assert payload["status"] == "degraded"
    assert payload["worker_ready"] is True
    assert payload["bge_ready"] is False
    assert payload["bge_fallback_active"] is True
    assert payload["verification_ready"] is True
    assert payload["worker"]["ready"] is True
    assert payload["semantic_ranking"] == {
        "bge_ready": False, "fallback_active": True, "error": "allocation failure",
    }
    assert payload["ollama"]["ready"] is True


def test_verify_valid_text_maps_frontend_shape_and_serializes_evidence(monkeypatch):
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return _state()

    monkeypatch.setattr(main, "run_pipeline", fake_pipeline)
    response = client.post("/api/verify", json={"text": "The claim.", "image": None})
    assert response.status_code == 200
    assert captured == {"text": "The claim."}
    assert response.json() == {
        "verdict": "real", "confidence": 87,
        "verification_state": "VERIFIED",
        "status": "The available evidence was reviewed.",
        "why": "The available evidence was reviewed.",
        "evidence": [{
            "source": "Example source", "stance": "support", "relevance": 84,
            "trust": 87, "excerpt": "Source excerpt.",
            "note": "The source directly supports the claim.",
            "verifier_relationship": None, "constraint_result": None,
            "rejection_reason": None,
        }],
    }


def test_unverified_maps_to_uncertain_with_zero_confidence(monkeypatch):
    state = _state("Unverified", evidence=False)
    assert state["final_output"]["confidence"] == 0.8
    assert state["supervisor_result"]["confidence_score"] == 80
    monkeypatch.setattr(main, "run_pipeline", lambda **_kwargs: state)
    response = client.post("/api/verify", json={"text": "An uncertain claim.", "image": None})
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "uncertain"
    assert body["confidence"] == 0
    assert body["evidence"] == []


def test_conversation_new_verification_runs_graph_and_starts_fresh_context(monkeypatch):
    captured = {}
    state = _state()
    state.update({
        "claim": "The claim.", "text": "The claim.",
        "domain": {"domain": "Science", "confidence": 0.8},
        "local_evidence": [], "external_evidence": [], "fused_evidence": [],
        "provider_statuses": {"news": "zero_results"},
        "external_provider_state": "zero_results",
        "reflection_result": {"notes": "Reviewed this claim."},
    })

    def fake_graph(**kwargs):
        captured.update(kwargs)
        return state

    monkeypatch.setattr(main, "run_pipeline", fake_graph)
    response = client.post("/api/conversation", json={
        "intent": "NEW_VERIFICATION", "message": "The claim.",
        "session_state": {"current_claim": "stale claim"},
    })

    assert response.status_code == 200
    assert captured == {"text": "The claim."}
    body = response.json()
    assert body["verdict"] == "real"
    assert body["session_state"]["current_claim"] == "The claim."
    assert body["session_state"]["domain"] == "Science"
    assert body["session_state"]["reflection_notes"] == "Reviewed this claim."


def test_conversation_explain_and_source_routes_read_current_result_without_graph(monkeypatch):
    monkeypatch.setattr(main, "run_pipeline", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("graph must not run")))
    current_result = {
        "verdict": "real", "confidence": 88,
        "why": "The source directly supports the claim.",
        "status": "Reviewed.",
        "evidence": [{"source": "NASA", "url": "https://nasa.gov/example", "stance": "support"}],
    }

    explanation = client.post("/api/conversation", json={
        "intent": "EXPLAIN_RESULT", "message": "Why is the result real?",
        "current_result": current_result,
    })
    source = client.post("/api/conversation", json={
        "intent": "ASK_ABOUT_SOURCE", "message": "Which source was considered?",
        "current_result": current_result,
    })

    assert explanation.status_code == source.status_code == 200
    assert "real" in explanation.json()["answer"].lower()
    assert "NASA" in source.json()["answer"]


def test_conversation_more_evidence_preserves_context_without_full_graph(monkeypatch):
    state = _state("Unverified", evidence=False)
    state.update({
        "claim": "Stored claim.", "domain": {"domain": "Finance"},
        "local_evidence": [], "external_evidence": [], "fused_evidence": [],
        "provider_statuses": {"world_bank": "zero_results"},
        "external_provider_state": "zero_results",
    })
    calls = []

    def more_evidence(session):
        calls.append(session)
        return state

    monkeypatch.setattr(main, "_run_more_evidence", more_evidence)
    monkeypatch.setattr(main, "run_pipeline", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("full graph must not run")))
    response = client.post("/api/conversation", json={
        "intent": "MORE_EVIDENCE", "message": "Find more evidence.",
        "session_state": {"current_claim": "Stored claim.", "domain": "Finance"},
    })

    assert response.status_code == 200
    assert calls == [{"current_claim": "Stored claim.", "domain": "Finance"}]
    assert response.json()["session_state"]["current_claim"] == "Stored claim."


def test_conversation_general_query_does_not_run_verification_or_mutate_session(monkeypatch):
    before = {"current_claim": "Claim A", "domain": "Science"}
    import copy
    original = copy.deepcopy(before)
    monkeypatch.setattr(main, "run_pipeline", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("graph must not run")))
    response = client.post("/api/conversation", json={
        "intent": "GENERAL_QUERY", "message": "Hello!", "session_state": before,
    })

    assert response.status_code == 200
    assert response.json()["intent"] == "GENERAL_QUERY"
    assert response.json()["answer"]
    assert before == original


def test_verify_rejects_empty_text_without_image():
    response = client.post("/api/verify", json={"text": "  ", "image": None})
    assert response.status_code == 400
    assert response.json()["detail"] == "Provide claim text or an image."


def test_verify_uses_existing_ocr_pipeline_for_frontend_image_data_url(monkeypatch):
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        with open(kwargs["image_path"], "rb") as image_file:
            captured["image_bytes"] = image_file.read()
        return _state("Unverified", evidence=False)

    monkeypatch.setattr(main, "run_pipeline", fake_pipeline)
    image_bytes = _png_bytes()
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    response = client.post("/api/verify", json={
        "text": "", "image": {"name": "claim.png", "preview": data_url},
    })
    assert response.status_code == 200
    assert captured["text"] is None
    assert captured["image_bytes"] == image_bytes
    assert not os.path.exists(captured["image_path"])


def _png_bytes(width=2, height=2):
    buffer = BytesIO()
    Image.new("RGB", (width, height), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def _image_request(image_bytes=None, text=""):
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes or _png_bytes()).decode("ascii")
    return {"text": text, "image": {"name": "claim.png", "preview": data_url}}


def _processing_failure_state(monkeypatch, text):
    def fail_processing(**_kwargs):
        raise ValueError("private OCR detail")

    monkeypatch.setattr(graph, "process_input", fail_processing)
    return graph.input_processing_node({"text": text or None, "image_path": "temporary-upload.png"})


def test_image_processing_success_is_returned_normally(monkeypatch):
    state = _state("Unverified", evidence=False)
    state.update(
        had_image=True, had_text=False, quality_flags=[],
        ocr_data={"text": "A readable headline."}, ocr_confidence=0.9,
    )
    monkeypatch.setattr(main, "run_pipeline", lambda **_kwargs: state)
    response = client.post("/api/verify", json=_image_request())
    assert response.status_code == 200
    assert response.json()["verdict"] == "uncertain"
    assert "warning" not in response.json()


def test_image_only_processing_failure_is_unverified_processing_error(monkeypatch):
    state = _processing_failure_state(monkeypatch, "")
    state.update(_state("Real"))
    monkeypatch.setattr(main, "run_pipeline", lambda **_kwargs: state)
    response = client.post("/api/verify", json=_image_request())
    assert response.status_code == 422
    assert response.json() == {
        "code": "image_processing_failed",
        "detail": "Screenshot could not be processed. Please try again.",
    }
    assert "private OCR detail" not in response.text


def test_text_and_image_processing_failure_preserves_text_result_and_warns(monkeypatch):
    state = _processing_failure_state(monkeypatch, "The supplied text claim.")
    state.update(_state("Real"))
    state.update(had_image=True, had_text=True, quality_flags=["input_processing_error"])
    monkeypatch.setattr(main, "run_pipeline", lambda **_kwargs: state)
    response = client.post("/api/verify", json=_image_request(text="The supplied text claim."))
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "real"
    assert body["warning"]["code"] == "image_processing_failed"
    assert "text only" in body["warning"]["message"]


def test_image_rejects_invalid_file_content():
    response = client.post("/api/verify", json=_image_request(b"not an image"))
    assert response.status_code == 400
    assert response.json() == {"detail": "Image content is not a valid supported image."}


def test_oversized_image_bytes_are_rejected_before_pipeline(monkeypatch):
    monkeypatch.setattr(main, "run_pipeline", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("pipeline called")))
    max_encoded_length = ((main._MAX_IMAGE_BYTES + 2) // 3) * 4 + 4
    encoded = "A" * (max_encoded_length + 1)
    response = client.post("/api/verify", json={"text": "", "image": {"preview": f"data:image/png;base64,{encoded}"}})
    assert response.status_code == 400
    assert response.json() == {"detail": "Image exceeds the 12 MiB limit."}


def test_oversized_image_dimensions_are_rejected(monkeypatch):
    decoded = MagicMock()
    decoded.__enter__.return_value = decoded
    decoded.format = "PNG"
    decoded.size = (10_000, 10_000)
    monkeypatch.setattr(main.Image, "open", lambda _stream: decoded)
    response = client.post("/api/verify", json=_image_request())
    assert response.status_code == 400
    assert response.json() == {"detail": "Image dimensions exceed the 40 megapixel limit."}


def test_oversized_claim_text_is_rejected():
    response = client.post("/api/verify", json={"text": "x" * (main._MAX_TEXT_CHARS + 1), "image": None})
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid request data."}


def test_oversized_request_body_is_rejected_before_json_validation():
    limited_client = TestClient(main.RequestBodyLimitMiddleware(main.app, max_body_bytes=32))
    response = limited_client.post("/api/verify", content=b"{" + b" " * 64)
    assert response.status_code == 413
    assert response.json() == {"detail": "Request body exceeds the 17 MiB limit."}


def test_followup_explains_uncertainty_without_excerpts():
    response = client.post("/api/followup", json={
        "question": "Why is this result uncertain?",
        "result": {
            "verdict": "uncertain", "confidence": 0,
            "why": "The available evidence was insufficient to establish the claim.",
            "status": "Unverified.",
            "evidence": [{"source": "Archive", "excerpt": "A very long unrelated source excerpt."}],
        },
    })
    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "uncertain" in answer.lower()
    assert "0% confidence" in answer
    assert "insufficient" in answer
    assert "A very long unrelated source excerpt." not in answer
    assert len(answer) < 300


def test_followup_summarizes_evidence_metadata_without_excerpts():
    response = client.post("/api/followup", json={
        "question": "What evidence was considered?",
        "result": {
            "verdict": "uncertain", "confidence": 35,
            "evidence": [
                {"source": "National Archive", "stance": "unclear", "relevance": 82,
                 "trust": 91, "excerpt": "FULL SOURCE TEXT THAT MUST NOT BE RETURNED"},
                {"source": "Public Records", "stance": "limiting"},
            ],
        },
    })
    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "National Archive" in answer
    assert "Public Records" in answer
    assert "unclear" in answer
    assert "relevance 82%" in answer
    assert "trust 91%" in answer
    assert "FULL SOURCE TEXT THAT MUST NOT BE RETURNED" not in answer
    assert len(answer) < 500


def test_followup_general_answer_is_concise():
    response = client.post("/api/followup", json={
        "question": "Tell me more.",
        "result": {
            "verdict": "real", "confidence": 78,
            "why": "The submitted record supports the claim.",
            "status": "Verified against the supplied evidence.",
            "evidence": [{"source": "Official Register", "stance": "support",
                          "excerpt": "Long evidence excerpt omitted from follow-up."}],
        },
    })
    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "submitted record supports the claim" in answer
    assert "Official Register" in answer
    assert "Long evidence excerpt omitted from follow-up." not in answer
    assert len(answer) < 500


def test_followup_rejects_empty_question_and_invalid_result():
    empty_question = client.post("/api/followup", json={"question": "  ", "result": {"verdict": "real"}})
    invalid_result = client.post("/api/followup", json={"question": "Why?", "result": {"verdict": "maybe"}})
    assert empty_question.status_code == 400
    assert invalid_result.status_code == 400


def test_unexpected_graph_error_returns_clean_json_500(monkeypatch):
    monkeypatch.setattr(main, "run_pipeline", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("private stack detail")))
    response = client.post("/api/verify", json={"text": "A claim.", "image": None})
    assert response.status_code == 500
    assert response.json() == {"detail": "Verification backend failed."}
    assert "private stack detail" not in response.text


def test_followup_requires_result_object():
    response = client.post("/api/followup", json={"question": "Why?", "result": []})
    assert response.status_code == 400


@pytest.mark.parametrize(
    ("claim", "verdict", "evidence_text", "model_relationship"),
    [
        (
            "Smoking increases the risk of lung cancer.", "real",
            "Scientific studies show that cigarette smoking causes lung cancer.",
            "SUPPORTS",
        ),
        (
            "The Earth is the center of the solar system and the Sun orbits Earth.", "fake",
            "The Sun, not Earth, is at the center of the solar system. Earth orbits the Sun.",
            "CONTRADICTS",
        ),
        (
            "The Earth orbits the Sun once every year.", "real",
            "NASA explains that Earth revolves around the Sun once every 365.25 days.",
            "SUPPORTS",
        ),
    ],
)
def test_api_verify_runs_graph_and_directness_boundary(
    monkeypatch, claim, verdict, evidence_text, model_relationship,
):
    import json
    from types import SimpleNamespace

    from rag_ingestion.verification.ollama_verifier import _LLM_RESILIENCE
    from rag_ingestion.verification.verification_manager import EvidenceVerificationManager

    monkeypatch.setattr(graph, "_COMPILED_GRAPH", None)
    monkeypatch.setattr(graph, "process_input", lambda **kwargs: {
        "processed_text": kwargs.get("text") or claim,
        "pre_translation_text": kwargs.get("text") or claim,
        "original_language": "en", "language_confidence": 1.0,
        "was_translated": False, "had_text": True, "had_image": False,
        "segments": [], "ocr_data": {}, "ocr_confidence": None,
        "quality_flags": [], "translation_info": {}, "errors": [],
    })

    class ClaimAgent:
        def extract_claim(self, value):
            return value

    class DomainAgent:
        def detect_domain(self, _claim):
            return {"domain": "Science", "confidence": 1.0, "reason": "test"}

    monkeypatch.setattr(graph, "get_claim_agent", ClaimAgent)
    monkeypatch.setattr(graph, "get_domain_agent", DomainAgent)
    monkeypatch.setattr(graph, "get_reflection_agent", lambda: SimpleNamespace(
        reflect=lambda **_kwargs: {"decision": "NO", "is_evidence_sufficient": False,
                                   "confidence": 0.0, "notes": "controlled API fixture"}
    ))
    monkeypatch.setattr(graph, "get_persona_agents", lambda: [])

    evidence = Evidence(
        evidence_id="directness-api-fixture", chunk_id="chunk", document_id="document",
        text=evidence_text, domain="science", source="NASA / controlled test source",
        language="en", metadata={}, retrieval_method="external",
        similarity_score=0.9, authority_score=9.0,
    )

    class ControlledPipeline:
        def process(self, *, claim, domain, **_kwargs):
            verification = EvidenceVerificationManager(timeout=1).verify(claim, [evidence])
            return SimpleNamespace(
                claim=claim, domain=domain, local_evidence=[], external_evidence=[evidence],
                fused_evidence=[evidence], verification_result=verification,
                verification_failure=None, external_source_errors=[],
                provider_statuses={"controlled": "success"},
                provider_diagnostics={}, retrieval_diagnostics={},
            )

    monkeypatch.setattr(graph, "get_claim_processing_pipeline", ControlledPipeline)
    monkeypatch.setattr(
        _LLM_RESILIENCE, "generate",
        lambda _call: {"response": json.dumps({
            "results": [{
                "evidence_id": evidence.evidence_id,
                "relationship": {"SUPPORTS": "SUPPORTED", "CONTRADICTS": "CONTRADICTED"}[model_relationship],
            }],
        })},
    )

    response = client.post("/api/verify", json={"text": claim, "image": None})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["verdict"] == verdict, body
    assert body["evidence"][0]["stance"] == ("support" if verdict == "real" else "limiting")
