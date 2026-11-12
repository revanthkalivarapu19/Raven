from datetime import datetime
from types import SimpleNamespace
import threading
import time

import pytest
import requests
import numpy as np

from agents import graph
from agents.input_processing import pipeline as input_pipeline
from agents.input_processing.ocr import OCR_MAX_LONG_EDGE, preprocess_image
from api import main
from rag_ingestion.llm_resilience import LLMInvocationError, LLMResilience, remaining_request_budget
from rag_ingestion.pipeline.claim_processing import (
    MAX_EVIDENCE_ITEMS_TO_VERIFY,
    ClaimProcessingPipeline,
)
from rag_ingestion.retrieval.evidence_fusion import EvidenceFusionManager
from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification.verification_manager import EvidenceVerificationManager
from tests.test_p0_4_verification_bounds import _pipeline, valid_result


def _evidence(evidence_id, text, *, domain="general", similarity=0.9, year=None):
    return Evidence(
        evidence_id=evidence_id,
        chunk_id=evidence_id,
        document_id=evidence_id,
        text=text,
        title=text,
        domain=domain,
        source="fixture",
        language="en",
        metadata={},
        similarity_score=similarity,
        publication_date=datetime(year, 1, 1) if year else None,
    )


def test_claim_specific_fusion_keeps_smoking_evidence_and_drops_distractors():
    claim = "Smoking increases the risk of lung cancer."
    relevant = _evidence(
        "smoking",
        "Smoking and tobacco use increase lung cancer risk.",
        domain="medical",
        similarity=0.72,
    )
    distractors = [
        _evidence("covid", "COVID-19 vaccination lowers respiratory infection risk.", domain="medical", similarity=0.98),
        _evidence("antibiotics", "Antibiotic resistance affects bacterial infections.", domain="medical", similarity=0.97),
    ]

    result = EvidenceFusionManager().fuse(
        [relevant, *distractors], [], 5, claim=claim, domain="medical"
    )

    assert [item.evidence_id for item in result] == ["smoking"]


def test_claim_specific_fusion_keeps_federal_reserve_2022_and_drops_current_or_broad_items():
    claim = "The Federal Reserve raised interest rates in 2022."
    relevant = _evidence(
        "fed-2022",
        "In 2022, the Federal Reserve raised interest rates several times.",
        domain="finance",
        similarity=0.70,
        year=2022,
    )
    broad = _evidence(
        "world-bank",
        "World Bank Global Economic Prospects discusses productivity in 2020.",
        domain="finance",
        similarity=0.99,
        year=2020,
    )
    current = _evidence(
        "fed-2026",
        "The Federal Reserve held interest rates steady in 2026.",
        domain="finance",
        similarity=0.95,
        year=2026,
    )

    result = EvidenceFusionManager().fuse(
        [relevant, broad, current], [], 5, claim=claim, domain="finance"
    )

    assert [item.evidence_id for item in result] == ["fed-2022"]


def test_claim_specific_fusion_keeps_earth_sun_orbit_and_drops_unrelated_science():
    claim = "The Earth revolves around the Sun."
    relevant = _evidence(
        "orbit",
        "Earth orbits the Sun once every year.",
        domain="science",
        similarity=0.72,
    )
    unrelated = _evidence(
        "anthropogenesis",
        "Probable physical factors of anthropogenesis and human origins on Earth.",
        domain="science",
        similarity=0.99,
    )

    result = EvidenceFusionManager().fuse(
        [relevant, unrelated], [], 5, claim=claim, domain="science"
    )

    assert [item.evidence_id for item in result] == ["orbit"]


def test_unrelated_general_distractor_is_filtered_without_claim_specific_terms():
    claim = "The Eiffel Tower opened in 1889."
    unrelated = _evidence(
        "distractor", "COVID-19 guidance and antibiotic resistance updates.", similarity=0.99
    )
    assert EvidenceFusionManager().fuse(
        [unrelated], [], 3, claim=claim, domain="general"
    ) == []


def test_unknown_domain_routes_to_general_even_without_local_index(monkeypatch, tmp_path):
    calls = []

    class Pipeline:
        def process(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                local_evidence=[], external_evidence=[], external_source_errors=[],
                fused_evidence=[], verification_result=None,
                verification_failure=None, local_retrieval_error=None,
            )

    monkeypatch.setattr(graph, "get_claim_processing_pipeline", lambda: Pipeline())
    monkeypatch.setattr(graph, "_build_retrieval_queries", lambda **_kwargs: ["Earth solar system orbit"])
    monkeypatch.setattr(graph._config, "data_dir", tmp_path)
    state = {
        "claim": "The Earth is the center of the solar system.",
        "domain": {"domain": "Unknown", "confidence": 0.0},
        "claim_scope": "GENERAL",
        "errors": [],
    }

    graph.evidence_verification_node(state)

    assert calls[0]["domain"] == "General"
    assert state["domain"] == {"domain": "Unknown", "confidence": 0.0}
    assert calls[0]["claim"] == "The Earth is the center of the solar system."


def test_unknown_domain_uses_general_only_when_general_index_exists(monkeypatch, tmp_path):
    calls = []

    class Pipeline:
        def process(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                local_evidence=[], external_evidence=[], external_source_errors=[],
                fused_evidence=[], verification_result=None,
                verification_failure=None, local_retrieval_error=None,
            )

    index = tmp_path / "vector_store" / "general" / "index.faiss"
    index.parent.mkdir(parents=True)
    index.write_bytes(b"existing index fixture")
    monkeypatch.setattr(graph, "get_claim_processing_pipeline", lambda: Pipeline())
    monkeypatch.setattr(graph, "_build_retrieval_queries", lambda **_kwargs: ["query"])
    monkeypatch.setattr(graph._config, "data_dir", tmp_path)
    state = {
        "claim": "A factual claim.",
        "domain": {"domain": "Unknown", "confidence": 0.0},
        "claim_scope": "GENERAL", "errors": [],
    }

    graph.evidence_verification_node(state)

    assert calls[0]["domain"] == "General"
    assert state["domain"]["domain"] == "Unknown"


def test_politics_alias_uses_existing_political_index(monkeypatch):
    calls = []

    class Pipeline:
        def process(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                local_evidence=[], external_evidence=[], external_source_errors=[],
                fused_evidence=[], verification_result=None,
                verification_failure=None, local_retrieval_error=None,
            )

    monkeypatch.setattr(graph, "get_claim_processing_pipeline", lambda: Pipeline())
    monkeypatch.setattr(graph, "_build_retrieval_queries", lambda **_kwargs: ["Prime Minister policy"])
    state = {
        "claim": "The Prime Minister announced a policy.",
        "domain": {"domain": "Politics", "confidence": 0.95},
        "claim_scope": "GENERAL",
        "errors": [],
    }

    graph.evidence_verification_node(state)

    assert calls[0]["domain"] == "political"
    assert state["domain"] == {"domain": "Politics", "confidence": 0.95}


def test_verification_input_and_model_timeout_are_bounded():
    assert EvidenceVerificationManager().timeout <= 35
    claim = "The original claim."
    verifier = SimpleNamespace(calls=[])

    def record_verify(claim, evidence_list, **_kwargs):
        verifier.calls.append((claim, evidence_list))
        return valid_result(claim)

    verifier.verify = record_verify
    pipeline = _pipeline(verifier)
    result = pipeline.process(
        claim=claim,
        domain="general",
        retrieval_queries=["one", "two", "three"],
        top_k_fused=5,
    )

    assert result.verification_failure is None
    assert len(verifier.calls[0][1]) <= MAX_EVIDENCE_ITEMS_TO_VERIFY == 2


def test_persona_is_bounded_and_xai_does_not_block(monkeypatch):
    observed = {}
    monkeypatch.setattr(graph, "PERSONA_MAX_RUNTIME_SECONDS", 2)
    monkeypatch.setattr(graph.SupervisorAgent, "build_context", lambda **_kwargs: object())
    persona_state = {
        "claim": "A claim.", "domain": {"domain": "General"},
        "fused_evidence": [SimpleNamespace(evidence_id="e1")],
        "verification_result": SimpleNamespace(results=[]),
        "request_deadline_at": time.monotonic() + 90, "errors": [],
    }
    # Return valid evidence IDs while capturing the bounded remaining duration.
    monkeypatch.setattr(
        graph,
        "get_persona_agents",
        lambda: [SimpleNamespace(
            name="test",
            analyze=lambda _context: (
                observed.setdefault("persona_budget", remaining_request_budget()),
                {"evidence_ids": ["e1"]},
            )[1],
        )],
    )
    graph.persona_analysis_node(persona_state)
    assert 0 < observed["persona_budget"] <= 2

    monkeypatch.setattr(graph, "get_xai_agent", lambda: pytest.fail("XAI must not block the response"))
    xai_state = {
        "claim": "A claim.", "domain": {"domain": "General"},
        "request_deadline_at": time.monotonic() + 300,
        "supervisor_result": {"final_verdict": "Unverified", "confidence_score": 0},
        "verification_result": SimpleNamespace(overall_assessment="INSUFFICIENT_EVIDENCE"),
    }
    result = graph.xai_node(xai_state)
    assert result["final_output"]["final_verdict"] == "Unverified"
    assert "xai_skipped: non_blocking_deterministic_explanation" in result["errors"]


def test_missing_supervisor_result_returns_deterministic_unverified_explanation():
    result = graph.xai_node({"claim": "A claim.", "supervisor_result": {}, "errors": []})
    assert result["final_output"]["final_verdict"] == "Unverified"
    assert result["final_output"]["confidence"] == 0
    assert "final safety review" in result["final_output"]["explanation"]


def test_missing_ocr_claim_still_returns_valid_deterministic_output(monkeypatch):
    monkeypatch.setattr(graph, "get_xai_agent", lambda: pytest.fail("XAI must not run"))
    result = graph.xai_node({
        "claim": "", "had_image": True, "quality_flags": ["low_ocr_confidence"],
        "supervisor_result": {"final_verdict": "Unverified", "confidence_score": 0},
        "errors": [],
    })
    assert result["final_output"]["final_verdict"] == "Unverified"
    assert "screenshot text could not be interpreted" in result["final_output"]["explanation"]
    assert "xai_skipped: no claim available" in result["errors"]


def _input_pipeline(monkeypatch, confidence, text=""):
    monkeypatch.setattr(input_pipeline.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(input_pipeline, "extract_ocr_data", lambda _path: {
        "text": "The headline claim from the screenshot.",
        "blocks": [], "average_confidence": confidence, "quality_flags": [],
    })
    monkeypatch.setattr(input_pipeline, "detect_language_with_confidence", lambda _text: ("en", 1.0))
    monkeypatch.setattr(input_pipeline, "normalize_text", lambda value, _lang: value)
    monkeypatch.setattr(input_pipeline, "clean_content_with_flags", lambda value: (value, []))
    return input_pipeline.process_input(text=text or None, image_path="fixture.png")


def test_good_ocr_is_preserved_and_used(monkeypatch):
    result = _input_pipeline(monkeypatch, 0.9)
    assert "headline claim" in result["processed_text"]
    assert result["ocr_data"]["text"] == "The headline claim from the screenshot."
    assert result["ocr_confidence"] == 0.9


def test_low_confidence_ocr_is_preserved_but_not_mixed_into_supplied_text(monkeypatch):
    result = _input_pipeline(monkeypatch, 0.4455, text="The user supplied claim.")
    assert result["processed_text"] == "The user supplied claim."
    assert result["ocr_data"]["text"] == "The headline claim from the screenshot."
    assert "low_ocr_confidence" in result["quality_flags"]


def test_image_only_low_confidence_ocr_is_unverified_and_warned(monkeypatch):
    state = {
        "claim": "A claim extracted from the image.",
        "claim_scope": "GENERAL",
        "had_image": True,
        "had_text": False,
        "ocr_data": {"text": "Actual low-confidence OCR text."},
        "ocr_confidence": 0.4455,
        "quality_flags": ["low_ocr_confidence"],
        "fused_evidence": [SimpleNamespace(
            evidence_id="e1", source="retrieved source", text="excerpt",
            similarity_score=0.9, authority_score=0.8,
        )],
        "verification_result": None,
        "supervisor_result": {"final_verdict": "Real", "confidence_score": 95},
        "final_output": {"final_verdict": "Real", "confidence": 0.95, "explanation": "model output"},
    }

    result = main.adapt_graph_result(state)

    assert result["verdict"] == "uncertain"
    assert result["confidence"] == 0
    assert result["evidence"] == []
    assert result["ocr"]["text"] == "Actual low-confidence OCR text."
    assert result["warning"]["code"] == "low_ocr_confidence"


def test_low_confidence_ocr_with_text_keeps_text_verdict_and_warns():
    state = {
        "had_image": True, "had_text": True,
        "ocr_data": {"text": "Noisy OCR"}, "ocr_confidence": 0.4455,
        "quality_flags": ["low_ocr_confidence"],
        "final_output": {"final_verdict": "Real", "confidence": 0.8, "explanation": "Text result."},
        "supervisor_result": {}, "fused_evidence": [],
    }

    result = main.adapt_graph_result(state)

    assert result["verdict"] == "real"
    assert result["confidence"] == 80
    assert result["warning"]["code"] == "low_ocr_confidence"
    assert "supplied text only" in result["warning"]["message"]


def test_ocr_failure_remains_explicit_for_image_only(monkeypatch):
    monkeypatch.setattr(input_pipeline, "extract_ocr_data", lambda _path: (_ for _ in ()).throw(ValueError("OCR failed")))
    result = graph.input_processing_node({"text": None, "image_path": "fixture.png"})
    assert "input_processing_error" in result["quality_flags"]
    assert result["ocr_confidence"] is None


def test_ocr_timeout_returns_empty_low_quality_result_without_rewriting_text(monkeypatch):
    release = threading.Event()

    def blocked(_path):
        release.wait(1)
        return {"text": "", "blocks": [], "average_confidence": 0.0, "quality_flags": []}

    monkeypatch.setattr(input_pipeline.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(input_pipeline, "extract_ocr_data", blocked)
    monkeypatch.setattr(input_pipeline, "OCR_TIMEOUT_SECONDS", 0.01)
    result = input_pipeline.process_input(image_path="fixture.png")
    assert result["ocr_data"]["text"] == ""
    assert result["ocr_confidence"] == 0.0
    assert "ocr_timeout" in result["quality_flags"]
    release.set()
    for _ in range(100):
        if not input_pipeline._OCR_WORK_IN_PROGRESS:
            break
        time.sleep(0.01)


def test_ocr_preprocessing_caps_long_edge_at_900px():
    processed = preprocess_image(np.zeros((747, 1324, 3), dtype=np.uint8))
    assert OCR_MAX_LONG_EDGE == 900
    assert max(processed.shape) <= OCR_MAX_LONG_EDGE


def test_ollama_verification_failure_does_not_trigger_second_attempt():
    state = {
        "verification_failure": SimpleNamespace(
            error_type="OllamaUnavailableError", message="Ollama request failed (transport)"
        ),
        "attempt_count": 1,
        "claim": "A claim.",
        "errors": [],
    }
    assert graph.should_retry_after_reflection(state) == "continue"


def test_ollama_transport_diagnostic_keeps_exact_exception_in_internal_logs(caplog):
    def fail_transport():
        raise requests.exceptions.ReadTimeout("localhost:11434 read timed out")

    with pytest.raises(LLMInvocationError):
        LLMResilience(timeout_seconds=1).generate(fail_transport)

    assert "exception_type=ReadTimeout" in caplog.text
    assert "localhost:11434 read timed out" in caplog.text
