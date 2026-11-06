import base64
import os
from io import BytesIO
from unittest.mock import MagicMock

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


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
        "status": "The available evidence was reviewed.",
        "why": "The available evidence was reviewed.",
        "evidence": [{
            "source": "Example source", "stance": "support", "relevance": 84,
            "trust": 87, "excerpt": "Source excerpt.",
            "note": "The source directly supports the claim.",
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
    state.update(had_image=True, had_text=False, quality_flags=[])
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
