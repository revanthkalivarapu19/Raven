from __future__ import annotations

import base64
import binascii
import asyncio
import logging
import math
import multiprocessing
import os
import pickle
import re
import sys
import tempfile
import time
import warnings
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from PIL import Image, UnidentifiedImageError

from agents.input_processing.pipeline import MIN_RELIABLE_OCR_CONFIDENCE
from rag_ingestion.verification.claim_scope import verification_can_establish_claim
from rag_ingestion.request_budget import MAX_REQUEST_BUDGET_SECONDS
from api.warm_worker import (
    _THREAD_ENV_DEFAULTS,
    warm_pipeline_worker_entry as _warm_pipeline_worker_entry,
)

logger = logging.getLogger(__name__)


def run_pipeline(*args, **kwargs):
    """Load the graph only inside the warm worker, not in the API parent."""
    from agents.graph import run_pipeline as _run
    return _run(*args, **kwargs)


def supervisor_node(*args, **kwargs):
    from agents.graph import supervisor_node as _supervisor
    return _supervisor(*args, **kwargs)


def get_claim_processing_pipeline(*args, **kwargs):
    from agents.graph import get_claim_processing_pipeline as _get_pipeline
    return _get_pipeline(*args, **kwargs)

_MAX_IMAGE_BYTES = 12 * 1024 * 1024
_MAX_REQUEST_BYTES = 17 * 1024 * 1024
_MAX_TEXT_CHARS = 20_000
_MAX_IMAGE_PIXELS = 40_000_000
_IMAGE_TYPES = {
    "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
    "image/bmp": ".bmp", "image/tiff": ".tif",
}
_DATA_URL_RE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.*)$", re.DOTALL)

@asynccontextmanager
async def _application_lifespan(_app):
    try:
        await _warm_worker.start()
    except Exception:
        logger.exception("verification_worker event=startup_failed")
    try:
        yield
    finally:
        await _warm_worker.stop()


app = FastAPI(title="Raven API", version="1.0.0", lifespan=_application_lifespan)


class RequestBodyLimitMiddleware:
    """Bound request JSON before FastAPI parses it."""

    def __init__(self, app, max_body_bytes: int = _MAX_REQUEST_BYTES):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or scope.get("path") not in {"/api/verify", "/api/followup", "/api/conversation"}):
            await self.app(scope, receive, send)
            return

        content_length = next(
            (value for key, value in scope.get("headers", []) if key.lower() == b"content-length"),
            None,
        )
        if content_length is not None:
            try:
                if int(content_length) > self.max_body_bytes:
                    await self._too_large(send)
                    return
            except ValueError:
                await self._too_large(send)
                return

        chunks = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.max_body_bytes:
                await self._too_large(send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)
        sent = False

        async def replay_receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)

    async def _too_large(self, send):
        body = b'{"detail":"Request body exceeds the 17 MiB limit."}'
        await send({
            "type": "http.response.start", "status": 413,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        })
        await send({"type": "http.response.body", "body": body})


app.add_middleware(RequestBodyLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(RequestValidationError)
async def request_validation_error(_request: Request, _exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": "Invalid request data."})


class VerifyRequest(BaseModel):
    text: str = Field(default="", max_length=_MAX_TEXT_CHARS)
    image: Any | None = None

    @field_validator("text", mode="before")
    @classmethod
    def text_must_be_string(cls, value):
        if not isinstance(value, str):
            raise ValueError("text must be a string")
        return value


class FollowupRequest(BaseModel):
    question: str
    result: dict[str, Any]

    @field_validator("question", mode="before")
    @classmethod
    def question_must_be_string(cls, value):
        if not isinstance(value, str):
            raise ValueError("question must be a string")
        return value


class ConversationRequest(BaseModel):
    intent: Literal[
        "NEW_VERIFICATION", "EXPLAIN_RESULT", "MORE_EVIDENCE",
        "ASK_ABOUT_SOURCE", "GENERAL_QUERY",
    ]
    message: str = Field(default="", max_length=_MAX_TEXT_CHARS)
    image: Any | None = None
    session_state: dict[str, Any] = Field(default_factory=dict)
    current_result: dict[str, Any] = Field(default_factory=dict)


def _get(value: Any, key: str, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _score_percent(value: Any, *, scale: float) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    if not math.isfinite(score):
        return None
    if scale == 1.0 and 0.0 <= score <= 1.0:
        score *= 100.0
    elif scale == 10.0 and 0.0 <= score <= 10.0:
        score *= 10.0
    return int(round(max(0.0, min(100.0, score))))


def _frontend_confidence(final_output: Any, supervisor: Any) -> int:
    confidence = _get(final_output, "confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and math.isfinite(float(confidence)):
        score = float(confidence)
        return int(round(max(0.0, min(100.0, score * 100.0 if 0.0 <= score <= 1.0 else score))))
    score = _get(supervisor, "confidence_score")
    if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(float(score)):
        return int(round(max(0.0, min(100.0, float(score)))))
    return 0


def _verification_state(
    verification: Any, failure: Any, evidence_count: int,
    provider_statuses: dict[str, str] | None = None,
) -> str:
    """Keep retrieval, verification, and evidence outcomes distinct."""
    statuses = list((provider_statuses or {}).values())
    if failure is not None:
        return "VERIFICATION_FAILED"
    provider_failed = any(value in {
        "unavailable", "timed_out", "rate_limited", "request_failed", "partial_failure",
    } for value in statuses)
    if not evidence_count:
        return "RETRIEVAL_FAILED" if provider_failed else "NO_RELEVANT_EVIDENCE"
    assessment = str(_get(verification, "overall_assessment", "")).split(".")[-1].upper()
    if assessment == "CONFLICTING_EVIDENCE":
        return "CONFLICTING_EVIDENCE"
    if assessment in {"SUPPORTED", "CONTRADICTED"}:
        return "VERIFIED"
    return "EVIDENCE_FOUND"


_STATE_MESSAGES = {
    "RETRIEVAL_FAILED": "Source retrieval failed for this claim.",
    "NO_RELEVANT_EVIDENCE": "No reliable evidence was found for this claim.",
    "EVIDENCE_FOUND": "Retrieved evidence did not directly establish or contradict this claim.",
    "VERIFICATION_FAILED": "Evidence was retrieved, but verification could not complete.",
    "CONFLICTING_EVIDENCE": "Available reliable sources conflict.",
}


def _serialize_component(record: dict[str, Any]) -> dict[str, Any]:
    """Return isolated, evidence-attributed output for one atomic claim."""
    component = record.get("claim")
    verification = record.get("verification_result")
    failure = record.get("verification_failure")
    claim_id = str(_get(component, "id", "C1"))
    claim_text = str(_get(component, "text", ""))
    assessment = str(_get(verification, "overall_assessment", "INSUFFICIENT_EVIDENCE")).upper()
    confidence = _get(verification, "overall_confidence", 0.0)
    scope = _get(verification, "claim_scope", "GENERAL")
    by_id = {
        str(_get(item, "evidence_id")): item
        for item in (_get(verification, "results", []) or [])
        if _get(item, "evidence_id") is not None
    }
    expected = {"SUPPORTED": "SUPPORTS", "CONTRADICTED": "CONTRADICTS"}.get(assessment)
    establishes = bool(expected and any(
        str(_get(item, "relationship", "")).upper() == expected
        and verification_can_establish_claim(item, scope)
        for item in (_get(verification, "results", []) or [])
    ))
    verdict = {"SUPPORTED": "real", "CONTRADICTED": "fake"}.get(assessment, "uncertain")
    if failure is not None or not establishes:
        verdict, confidence = "uncertain", 0
    else:
        confidence = _score_percent(confidence, scale=1.0) or 0
    serialized = []
    for evidence in record.get("fused_evidence") or []:
        evidence_id = str(_get(evidence, "evidence_id", ""))
        verification_item = by_id.get(evidence_id)
        stance = "unclear"
        if verification_item is not None and verification_can_establish_claim(verification_item, scope):
            relationship = str(_get(verification_item, "relationship", "")).upper()
            stance = "support" if relationship == "SUPPORTS" else "limiting" if relationship == "CONTRADICTS" else "unclear"
        item = {
            "claim_id": claim_id,
            "source": str(_get(evidence, "source") or _get(evidence, "title") or ""),
            "stance": stance,
            "relevance": _score_percent(_get(evidence, "similarity_score"), scale=1.0),
            "trust": _score_percent(_get(evidence, "authority_score"), scale=10.0),
            "excerpt": str(_get(verification_item, "direct_evidence_excerpt") or _get(evidence, "text") or ""),
            "verifier_relationship": _get(verification_item, "verifier_relationship"),
            "constraint_result": _get(verification_item, "constraint_result"),
            "rejection_reason": _get(verification_item, "rejection_reason"),
        }
        url = _get(evidence, "url")
        if isinstance(url, str) and url.strip():
            item["url"] = url.strip()
        serialized.append(item)
    component_obj = component.model_dump() if hasattr(component, "model_dump") else component or {}
    reason = (
        _STATE_MESSAGES.get(_verification_state(
            verification, failure, len(serialized), record.get("provider_statuses")
        ), "Evidence was insufficient for this component.")
        if verdict == "uncertain" else
        f"The evidence relationship for this component was {assessment.lower()}."
    )
    return {
        "id": claim_id,
        "text": claim_text,
        "entities": component_obj.get("entities", []),
        "domain": component_obj.get("domain"),
        "temporal_context": component_obj.get("temporal_context", {}),
        "search_queries": component_obj.get("search_queries", []),
        "verdict": verdict,
        "confidence": confidence,
        "relationship": assessment,
        "verification_state": _verification_state(
            verification, failure, len(serialized), record.get("provider_statuses")
        ),
        "reason": reason,
        "supporting_evidence": [item for item in serialized if item["stance"] == "support"],
        "contradicting_evidence": [item for item in serialized if item["stance"] == "limiting"],
        "evidence": serialized,
        "retrieval_diagnostics": record.get("retrieval_diagnostics", {}),
    }


def adapt_graph_result(state: dict[str, Any]) -> dict[str, Any]:
    """Convert the existing LangGraph state to the frontend's result shape."""
    final_output = state.get("final_output") or {}
    supervisor = state.get("supervisor_result") or {}
    raw_verdict = str(
        _get(final_output, "final_verdict") or _get(supervisor, "final_verdict") or "Unverified"
    ).strip().lower()
    verdict = {"real": "real", "fake": "fake", "unverified": "uncertain"}.get(raw_verdict, "uncertain")
    confidence = 0 if verdict == "uncertain" else _frontend_confidence(final_output, supervisor)

    explanation = _get(final_output, "explanation")
    summary = _get(supervisor, "reasoning_summary")
    status = summary if isinstance(summary, str) and summary.strip() else explanation
    why = explanation if isinstance(explanation, str) and explanation.strip() else status

    verification = state.get("verification_result")
    verification_results = _get(verification, "results", []) or []
    by_evidence_id = {
        str(_get(item, "evidence_id")): item
        for item in verification_results if _get(item, "evidence_id") is not None
    }
    scope = state.get("claim_scope") or _get(verification, "claim_scope") or "GENERAL"
    serialized_evidence = []
    for item in state.get("fused_evidence") or []:
        evidence_id = _get(item, "evidence_id")
        verification_item = by_evidence_id.get(str(evidence_id)) if evidence_id is not None else None
        stance = "unclear"
        if verification_item is not None and verification_can_establish_claim(verification_item, scope):
            relationship = str(_get(verification_item, "relationship", "")).upper()
            if relationship == "SUPPORTS":
                stance = "support"
            elif relationship == "CONTRADICTS":
                stance = "limiting"

        source = _get(item, "source") or _get(item, "title") or ""
        excerpt = (
            _get(verification_item, "direct_evidence_excerpt")
            if verification_item is not None
            and verification_can_establish_claim(verification_item, scope)
            else None
        ) or _get(item, "text") or ""
        if not source and not excerpt:
            continue
        reasoning = _get(verification_item, "reasoning") if verification_item is not None else None
        serialized_item = {
            "source": str(source),
            "stance": stance,
            "relevance": _score_percent(_get(item, "similarity_score"), scale=1.0),
            "trust": _score_percent(_get(item, "authority_score"), scale=10.0),
            "excerpt": str(excerpt),
            "note": reasoning if isinstance(reasoning, str) else None,
            "verifier_relationship": _get(verification_item, "verifier_relationship"),
            "constraint_result": _get(verification_item, "constraint_result"),
            "rejection_reason": _get(verification_item, "rejection_reason"),
        }
        source_url = _get(item, "url")
        if isinstance(source_url, str) and source_url.strip():
            serialized_item["url"] = source_url.strip()
        metadata = _get(item, "metadata") or {}
        if isinstance(metadata, dict):
            retrieved_date = metadata.get("retrieved_date")
            if isinstance(retrieved_date, str) and retrieved_date.strip():
                serialized_item["retrieved_date"] = retrieved_date.strip()
        serialized_evidence.append(serialized_item)

    result = {
        "verdict": verdict,
        "confidence": confidence,
        "status": status if isinstance(status, str) else "",
        "why": why if isinstance(why, str) else "",
        "evidence": serialized_evidence,
    }
    verification_state = _verification_state(
        verification, state.get("verification_failure"), len(serialized_evidence),
        state.get("provider_statuses"),
    )
    result["verification_state"] = verification_state
    if result["verdict"] == "uncertain":
        message = _STATE_MESSAGES.get(verification_state)
        if message:
            result["status"] = message
            result["why"] = message
    component_records = state.get("claim_results") or []
    if component_records:
        result["claim_components"] = [
            _serialize_component(record) for record in component_records
        ]
    retrieval_diagnostics = state.get("retrieval_diagnostics")
    if retrieval_diagnostics:
        result["diagnostics"] = {
            "retrieval": retrieval_diagnostics,
            "decomposition": state.get("decomposition_diagnostics", {}),
            "request": state.get("request_diagnostics", {}),
        }
    elif state.get("request_diagnostics"):
        result["diagnostics"] = {"request": state["request_diagnostics"]}
    warnings_out = []
    provider_state = str(state.get("external_provider_state") or "")
    if provider_state == "provider_unavailable":
        warnings_out.append("External sources were unavailable for this check.")
    elif provider_state == "partial":
        warnings_out.append("Some external sources were unavailable for this check.")
    if state.get("verification_failure") is not None:
        warnings_out.append("Verification did not complete successfully.")
    if warnings_out:
        result["warnings"] = warnings_out
    verification_assessment = _get(verification, "overall_assessment")
    verification_confidence = _get(verification, "overall_confidence")
    provider_statuses = state.get("provider_statuses") or {}
    logger.info(
        "raven_trace original_input=%r claim=%r detected_domain=%r external_provider_state=%s provider_statuses=%s local_count=%d "
        "external_count=%d fused_count=%d verifier_model_invoked=%s "
        "verification_relationship=%s verification_confidence=%s supervisor_verdict=%s "
        "api_verdict=%s api_confidence=%d",
        str(state.get("text") or "")[:240],
        str(state.get("claim") or "")[:240],
        state.get("domain"),
        state.get("external_provider_state", "not_applicable"),
        provider_statuses,
        len(state.get("local_evidence") or []),
        len(state.get("external_evidence") or []),
        len(state.get("fused_evidence") or []),
        bool(state.get("fused_evidence")),
        getattr(verification_assessment, "value", verification_assessment),
        verification_confidence,
        _get(supervisor, "final_verdict") or "Unverified",
        verdict,
        confidence,
    )
    if state.get("had_image"):
        ocr_data = state.get("ocr_data") or {}
        ocr_confidence = state.get("ocr_confidence")
        quality_flags = list(state.get("quality_flags") or [])
        result["ocr"] = {
            "text": str(ocr_data.get("text") or ""),
            "confidence": ocr_confidence,
            "quality_flags": quality_flags,
        }
    image_failed = bool(
        state.get("had_image")
        and "input_processing_error" in (state.get("quality_flags") or [])
    )
    if image_failed and state.get("had_text"):
        result["warning"] = {
            "code": "image_processing_failed",
            "message": "The screenshot could not be processed. This result uses the supplied text only.",
        }
    elif state.get("had_image") and (
        not isinstance(ocr_confidence, (int, float))
        or ocr_confidence < MIN_RELIABLE_OCR_CONFIDENCE
        or "low_ocr_confidence" in quality_flags
    ):
        result["warning"] = {
            "code": "low_ocr_confidence",
            "message": (
                "Screenshot text confidence is low. "
                "The result is Unverified for image-only input."
                if not state.get("had_text")
                else "Screenshot text confidence is low. The result uses the supplied text only."
            ),
        }
        if not state.get("had_text"):
            result["verdict"] = "uncertain"
            result["confidence"] = 0
            result["status"] = "Screenshot text could not be interpreted with sufficient confidence."
            result["why"] = "Raven returned Unverified because OCR confidence was below the minimum quality threshold."
            result["evidence"] = []
    return result


def _session_state_from_graph(state: dict[str, Any], result: dict[str, Any], intent: str) -> dict[str, Any]:
    """Serialize the current graph context into the existing browser session."""
    domain = state.get("domain") or {}
    domain_name = _get(domain, "domain") or (domain if isinstance(domain, str) else "Unknown")
    verification = state.get("verification_result")
    supervisor = state.get("supervisor_result") or {}
    return {
        "current_intent": intent,
        "current_claim": str(state.get("claim") or ""),
        "current_input": str(state.get("text") or state.get("processed_text") or ""),
        "domain": str(domain_name),
        "evidence": result.get("evidence", []),
        "reflection_notes": str(_get(state.get("reflection_result"), "notes", "") or ""),
        "verification_result": {
            "relationship": str(_get(verification, "overall_assessment", "INSUFFICIENT_EVIDENCE")),
            "confidence": _score_percent(_get(verification, "overall_confidence"), scale=1.0) or 0,
            "failed": state.get("verification_failure") is not None,
        },
        "supervisor_result": {
            "verdict": str(_get(supervisor, "final_verdict", "Unverified")),
            "confidence": _score_percent(_get(supervisor, "confidence_score"), scale=1.0) or 0,
            "explanation": str(_get(supervisor, "reasoning_summary", "") or ""),
        },
        "generated_explanation": str(result.get("why") or ""),
        "provider_diagnostics": {
            "state": str(state.get("external_provider_state", "not_applicable")),
            "providers": dict(state.get("provider_statuses") or {}),
        },
    }


def _run_more_evidence(session_state: dict[str, Any]) -> dict[str, Any]:
    claim = str(session_state.get("current_claim") or "").strip()
    if not claim:
        raise HTTPException(status_code=409, detail="There is no current claim to search.")
    domain = session_state.get("domain") or "Unknown"
    if isinstance(domain, dict):
        domain = domain.get("domain") or "Unknown"
    domain = str(domain).strip() or "Unknown"
    pipeline = get_claim_processing_pipeline()
    processed = pipeline.process(
        claim=claim,
        domain=domain,
        retrieval_queries=[claim],
        top_k_local=5,
        top_k_external=5,
        top_k_fused=3,
    )
    supervisor = supervisor_node({
        "claim": claim,
        "claim_scope": getattr(processed.verification_result, "claim_scope", None),
        "domain": {"domain": domain},
        "fused_evidence": processed.fused_evidence,
        "verification_result": processed.verification_result,
        "verification_failure": processed.verification_failure,
    }).get("supervisor_result", {})
    return {
        "claim": claim,
        "domain": {"domain": domain},
        "local_evidence": processed.local_evidence,
        "external_evidence": processed.external_evidence,
        "fused_evidence": processed.fused_evidence,
        "verification_result": processed.verification_result,
        "verification_failure": processed.verification_failure,
        "supervisor_result": supervisor,
        "provider_statuses": processed.provider_statuses,
        "external_provider_state": processed.external_provider_state,
    }


def _image_parts(image: Any) -> tuple[bytes, str] | None:
    if image is None:
        return None
    data_url = image if isinstance(image, str) else image.get("preview") if isinstance(image, dict) else None
    if not isinstance(data_url, str):
        raise HTTPException(status_code=400, detail="Image must be a base64 image data URL.")
    match = _DATA_URL_RE.fullmatch(data_url)
    if not match:
        raise HTTPException(status_code=400, detail="Image must be a base64 image data URL.")
    mime_type, encoded = match.groups()
    suffix = _IMAGE_TYPES.get(mime_type.lower())
    if suffix is None:
        raise HTTPException(status_code=400, detail="Image type is not supported; use JPEG, PNG, WebP, BMP, or TIFF.")
    if len(encoded) > ((_MAX_IMAGE_BYTES + 2) // 3) * 4 + 4:
        raise HTTPException(status_code=400, detail="Image exceeds the 12 MiB limit.")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Image data is not valid base64.") from exc
    if not image_bytes or len(image_bytes) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image is empty or exceeds the 12 MiB limit.")
    expected_formats = {
        "image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP",
        "image/bmp": "BMP", "image/tiff": "TIFF",
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as decoded_image:
                if decoded_image.format != expected_formats[mime_type.lower()]:
                    raise HTTPException(status_code=400, detail="Image content does not match its declared type.")
                width, height = decoded_image.size
                if width < 1 or height < 1 or width * height > _MAX_IMAGE_PIXELS:
                    raise HTTPException(status_code=400, detail="Image dimensions exceed the 40 megapixel limit.")
                decoded_image.verify()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=400, detail="Image content is not a valid supported image.") from exc
    return image_bytes, suffix


def _run_pipeline(text: str, image: Any, deadline_at: float | None = None):
    image_parts = _image_parts(image)
    if image_parts is None:
        if deadline_at is None:
            return run_pipeline(text=text)
        return run_pipeline(text=text, deadline_at=deadline_at)
    image_bytes, suffix = image_parts
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="raven-upload-", suffix=suffix, delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_path = Path(temp_file.name)
        if deadline_at is None:
            return run_pipeline(text=text or None, image_path=str(temp_path))
        return run_pipeline(text=text or None, image_path=str(temp_path), deadline_at=deadline_at)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _atomic_pickle(path: Path, value: Any) -> None:
    temporary = Path(f"{path}.part")
    with temporary.open("wb") as output:
        pickle.dump(value, output, protocol=pickle.HIGHEST_PROTOCOL)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


class WarmPipelineWorker:
    """Single-slot persistent process with hard-deadline termination."""

    def __init__(self) -> None:
        self.work_dir = Path(tempfile.gettempdir()) / f"raven-warm-worker-{uuid.uuid4().hex}"
        self.process = None
        self.ready: dict[str, Any] | None = None
        self.busy = False
        self.restarts = 0
        self.timeout_count = 0
        self._restart_task: asyncio.Task | None = None
        self._next_request_restarted = False
        self.last_startup_error: str | None = None
        self.limit_worker_threads = True

    @property
    def is_ready(self) -> bool:
        return bool(self.process and self.process.is_alive() and self.ready)

    async def start(self) -> None:
        if self.is_ready:
            return
        # A prior startup can fail while the child is still alive (for
        # example, Windows can briefly deny the atomic readiness-file open).
        # Never overwrite that process handle and orphan a second worker.
        if self.process and self.process.is_alive():
            self._terminate()
        self.last_startup_error = None
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._clear_job_files()
        context = multiprocessing.get_context("spawn")
        graph_module = sys.modules.get("agents.graph")
        parent_pipeline = getattr(graph_module, "_CLAIM_PROCESSING_PIPELINE", None)
        logger.info(
            "verification_worker event=parent_runtime_before_spawn start_method=%s "
            "target=%s.%s python=%s torch_preloaded=%s sentence_transformers_preloaded=%s "
            "graph_preloaded=%s retrieval_pipeline_initialized=%s thread_environment=%s",
            getattr(context, "get_start_method", multiprocessing.get_start_method)(),
            _warm_pipeline_worker_entry.__module__,
            _warm_pipeline_worker_entry.__name__,
            sys.executable,
            "torch" in sys.modules,
            "sentence_transformers" in sys.modules,
            graph_module is not None,
            parent_pipeline is not None,
            {key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM"
            )},
        )
        self.process = context.Process(
            target=_warm_pipeline_worker_entry,
            args=(str(self.work_dir), self.limit_worker_threads),
            name="raven-warm-verification-worker",
        )
        started = time.monotonic()
        process_created_wall = time.time()
        self.ready = None
        inherited_environment = {
            key: os.environ.get(key) for key in _THREAD_ENV_DEFAULTS
        }
        try:
            if self.limit_worker_threads:
                # Windows spawn re-imports the main module before invoking its
                # target. Set these at process creation so even that bootstrap
                # happens before Torch/SentenceTransformers can initialize.
                for key, value in _THREAD_ENV_DEFAULTS.items():
                    os.environ[key] = value
            self.process.start()
        finally:
            for key, value in inherited_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        startup_limit = 180.0
        while time.monotonic() - started < startup_limit:
            ready_path = self.work_dir / "worker-ready.pickle"
            if ready_path.exists():
                try:
                    with ready_path.open("rb") as ready_file:
                        payload = pickle.load(ready_file)
                except PermissionError:
                    # Atomic replacement is used by the child, but Windows
                    # security scanners can transiently hold the new file.
                    await asyncio.sleep(0.05)
                    continue
                except Exception as exc:
                    self.last_startup_error = f"invalid worker readiness artifact: {type(exc).__name__}: {exc}"
                    self._terminate()
                    raise RuntimeError(self.last_startup_error) from exc
                if payload.get("status") == "ready":
                    child_initialized_wall = float(payload.get("worker_started_at", process_created_wall))
                    worker_ready_wall = float(payload.get("worker_ready_at", time.time()))
                    payload["worker_process_created_at"] = process_created_wall
                    payload["worker_import_complete_at"] = child_initialized_wall
                    payload["worker_started_at"] = process_created_wall
                    payload["worker_ready_at"] = worker_ready_wall
                    payload["worker_startup_seconds"] = max(0.0, worker_ready_wall - process_created_wall)
                    payload["worker_post_import_init_seconds"] = max(
                        0.0, worker_ready_wall - child_initialized_wall
                    )
                    self.ready = payload
                    self.last_startup_error = None
                    logger.info(
                        "verification_worker event=startup_complete pid=%s worker_started_at=%.6f "
                        "worker_ready_at=%.6f worker_startup_seconds=%.3f "
                        "worker_post_import_init_seconds=%.3f worker_restarts=%d "
                        "bge_ready=%s bge_fallback_active=%s start_method=%s "
                        "thread_environment=%s",
                        payload.get("pid"), process_created_wall, worker_ready_wall,
                        payload["worker_startup_seconds"],
                        payload["worker_post_import_init_seconds"], self.restarts,
                        payload.get("bge_ready"), payload.get("bge_fallback_active"),
                        payload.get("start_method"), payload.get("thread_environment"),
                    )
                    if self.restarts:
                        self._next_request_restarted = True
                    return
                self._terminate()
                self.last_startup_error = str(payload.get("error"))
                logger.error(
                    "verification_worker event=child_initialization_failed "
                    "last_successful_checkpoint=%s checkpoints=%s error=%s",
                    payload.get("last_successful_checkpoint"),
                    payload.get("checkpoint_details", payload.get("checkpoints")),
                    payload.get("error"),
                )
                raise RuntimeError(f"Warm verification worker initialization failed: {payload.get('error')}")
            if not self.process.is_alive():
                checkpoint_path = self.work_dir / "worker-checkpoints.pickle"
                checkpoint_data = None
                if checkpoint_path.exists():
                    try:
                        with checkpoint_path.open("rb") as checkpoint_file:
                            checkpoint_data = pickle.load(checkpoint_file)
                    except Exception:
                        logger.exception("verification_worker failed to read child checkpoints")
                logger.error(
                    "verification_worker event=child_exited_during_startup exit_code=%s "
                    "last_successful_checkpoint=%s child_checkpoints=%s",
                    self.process.exitcode,
                    (checkpoint_data or {}).get("last_successful_checkpoint"),
                    (checkpoint_data or {}).get("checkpoint_details", (checkpoint_data or {}).get("checkpoints")),
                )
                self.last_startup_error = (
                    f"worker exited with code {self.process.exitcode}; last checkpoint="
                    f"{(checkpoint_data or {}).get('last_successful_checkpoint')}"
                )
                raise RuntimeError(
                    "Warm verification worker exited during startup "
                    f"(exit_code={self.process.exitcode}, last_checkpoint="
                    f"{(checkpoint_data or {}).get('last_successful_checkpoint')})"
                )
            await asyncio.sleep(0.05)
        self._terminate()
        self.last_startup_error = "worker startup exceeded 180 seconds"
        raise TimeoutError("Warm verification worker did not become ready within 180 seconds")

    async def stop(self) -> None:
        self._terminate()
        try:
            for item in self.work_dir.iterdir():
                item.unlink(missing_ok=True)
            self.work_dir.rmdir()
        except OSError:
            logger.debug("Could not remove warm worker directory %s", self.work_dir)

    def _terminate(self) -> None:
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=0.2)
        self.process = None
        self.ready = None

    def _clear_job_files(self) -> None:
        for pattern in (
            "job-*.pickle", "running-*.pickle", "result-*.pickle",
            "started-*.pickle", "worker-ready.pickle", "worker-checkpoints.pickle",
        ):
            for item in self.work_dir.glob(pattern):
                item.unlink(missing_ok=True)

    async def _restart_in_background(self) -> None:
        if self._restart_task and not self._restart_task.done():
            return

        async def restart():
            try:
                await self.start()
            except Exception:
                logger.exception("verification_worker event=replacement_start_failed")

        self._restart_task = asyncio.create_task(restart())

    def _add_lifecycle_diagnostics(
        self, state: dict[str, Any], *, queue_wait: float, graph_seconds: float,
        total_seconds: float, restarted: bool,
    ) -> dict[str, Any]:
        diagnostics = dict(state.get("request_diagnostics") or {})
        diagnostics.update({
            "worker_started_at": self.ready.get("worker_started_at") if self.ready else None,
            "worker_ready_at": self.ready.get("worker_ready_at") if self.ready else None,
            # This is startup charged to this request. A warm request pays zero;
            # the one-time process startup remains visible separately below.
            "worker_startup_seconds": 0.0,
            "worker_lifetime_startup_seconds": (
                self.ready.get("worker_startup_seconds") if self.ready else None
            ),
            "worker_warm": True,
            "worker_restarts": self.restarts,
            "worker_timeout_count": self.timeout_count,
            "worker_restarted": restarted,
            "bge_ready": bool(self.ready and self.ready.get("bge_ready")),
            "bge_fallback_active": bool(self.ready and self.ready.get("bge_fallback_active")),
            "bge_error": self.ready.get("bge_error") if self.ready else None,
            "queue_wait_seconds": max(0.0, queue_wait),
            "graph_execution_seconds": max(0.0, graph_seconds),
            "verification_seconds": self._verification_seconds(state),
            "total_request_seconds": max(0.0, total_seconds),
            "deadline_seconds": MAX_REQUEST_BUDGET_SECONDS,
        })
        state["request_diagnostics"] = diagnostics
        return state

    @staticmethod
    def _verification_seconds(state: dict[str, Any]) -> float:
        claims = (state.get("retrieval_diagnostics") or {}).get("claims") or []
        if not claims:
            claims = [record.get("retrieval_diagnostics", {}) for record in state.get("claim_results", [])]
        return sum(float(item.get("verification_seconds", 0.0) or 0.0) for item in claims)

    async def run(self, text: str, image: Any) -> dict[str, Any]:
        if self.busy:
            raise HTTPException(status_code=503, detail="Raven verification worker is busy. Retry shortly.")
        if not self.is_ready:
            if not self._restart_task or self._restart_task.done():
                await self._restart_in_background()
            raise HTTPException(status_code=503, detail="Raven verification worker is warming or restarting. Retry shortly.")
        self.busy = True
        request_started = time.monotonic()
        wall_deadline_at = time.time() + MAX_REQUEST_BUDGET_SECONDS
        job_id = uuid.uuid4().hex
        job_path = self.work_dir / f"job-{job_id}.pickle"
        result_path = self.work_dir / f"result-{job_id}.pickle"
        started_path = self.work_dir / f"started-{job_id}.pickle"
        restarted = self._next_request_restarted
        self._next_request_restarted = False
        try:
            _atomic_pickle(job_path, {
                "text": text, "image": image,
                "wall_deadline_at": wall_deadline_at,
            })
            submitted_wall = time.time()
            deadline_at = request_started + MAX_REQUEST_BUDGET_SECONDS
            while time.monotonic() < deadline_at - 0.75:
                if result_path.exists():
                    with result_path.open("rb") as result_file:
                        status, payload = pickle.load(result_file)
                    if status != "ok":
                        raise RuntimeError(f"Warm verification worker failed: {payload}")
                    claimed_wall = submitted_wall
                    if started_path.exists():
                        with started_path.open("rb") as started_file:
                            claimed_wall = float(pickle.load(started_file).get("worker_claimed_at", submitted_wall))
                    diagnostics = payload.get("request_diagnostics") or {}
                    graph_seconds = float(diagnostics.get("total_duration_seconds", 0.0) or 0.0)
                    return self._add_lifecycle_diagnostics(
                        payload,
                        queue_wait=claimed_wall - submitted_wall,
                        graph_seconds=graph_seconds,
                        total_seconds=time.monotonic() - request_started,
                        restarted=restarted,
                    )
                if not self.process or not self.process.is_alive():
                    raise RuntimeError("Warm verification worker exited during request execution")
                await asyncio.sleep(0.025)

            prior_ready = self.ready
            self.timeout_count += 1
            self.restarts += 1
            self._terminate()
            await self._restart_in_background()
            state = _deadline_failure_state(text, time.monotonic() - request_started)
            diagnostics = dict(state.get("request_diagnostics") or {})
            diagnostics.update({
                "worker_started_at": prior_ready.get("worker_started_at") if prior_ready else None,
                "worker_ready_at": prior_ready.get("worker_ready_at") if prior_ready else None,
                "worker_startup_seconds": 0.0,
                "worker_lifetime_startup_seconds": (
                    prior_ready.get("worker_startup_seconds") if prior_ready else None
                ),
                "worker_warm": False,
                "worker_restarts": self.restarts,
                "worker_timeout_count": self.timeout_count,
                "worker_restarted": True,
                "queue_wait_seconds": 0.0,
                "graph_execution_seconds": 0.0,
                "verification_seconds": 0.0,
                "total_request_seconds": time.monotonic() - request_started,
                "deadline_seconds": MAX_REQUEST_BUDGET_SECONDS,
            })
            state["request_diagnostics"] = diagnostics
            return state
        except asyncio.CancelledError:
            self.restarts += 1
            self._terminate()
            await self._restart_in_background()
            logger.warning("verification_worker event=request_cancelled worker_restarted=true")
            raise
        except HTTPException:
            raise
        except Exception:
            prior_ready = self.ready
            self.restarts += 1
            self._terminate()
            await self._restart_in_background()
            logger.exception("verification_worker event=request_failed worker_restarted=true")
            state = _deadline_failure_state(text, time.monotonic() - request_started)
            diagnostics = dict(state.get("request_diagnostics") or {})
            diagnostics.update({
                "worker_started_at": prior_ready.get("worker_started_at") if prior_ready else None,
                "worker_ready_at": prior_ready.get("worker_ready_at") if prior_ready else None,
                "worker_startup_seconds": 0.0,
                "worker_lifetime_startup_seconds": (
                    prior_ready.get("worker_startup_seconds") if prior_ready else None
                ),
                "worker_warm": False,
                "worker_restarts": self.restarts,
                "worker_timeout_count": self.timeout_count,
                "worker_restarted": True,
                "queue_wait_seconds": 0.0,
                "graph_execution_seconds": 0.0,
                "verification_seconds": 0.0,
                "total_request_seconds": time.monotonic() - request_started,
                "deadline_seconds": MAX_REQUEST_BUDGET_SECONDS,
            })
            state["request_diagnostics"] = diagnostics
            return state
        finally:
            result_path.unlink(missing_ok=True)
            started_path.unlink(missing_ok=True)
            job_path.unlink(missing_ok=True)
            self.busy = False


_warm_worker = WarmPipelineWorker()


def _deadline_failure_state(text: str, elapsed: float) -> dict[str, Any]:
    from rag_ingestion.verification.verification_schema import VerificationFailure

    message = "The server-side request deadline was reached; processing was stopped."
    return {
        "text": text,
        "claim": text,
        "fused_evidence": [],
        "verification_result": None,
        "verification_failure": VerificationFailure(
            claim=text, evidence_count=0, error_type="RequestDeadlineExceeded",
            message=message,
        ),
        "supervisor_result": {
            "final_verdict": "Unverified", "confidence_score": 0,
            "reasoning_summary": message, "key_evidence_used": [],
        },
        "final_output": {
            "final_verdict": "Unverified", "confidence": 0,
            "explanation": message,
        },
        "request_diagnostics": {
            "request_deadline_seconds": MAX_REQUEST_BUDGET_SECONDS,
            "total_duration_seconds": elapsed,
            "deadline_respected": False,
            "deadline_reached": True,
            "graph_terminated": True,
        },
        "errors": ["request_deadline_exceeded: graph worker terminated"],
    }


async def _run_pipeline_with_deadline(text: str, image: Any):
    """Submit to the warm single-slot worker under a request-scoped deadline."""
    return await _warm_worker.run(text, image)


@app.get("/api/health")
async def health():
    from rag_ingestion.verification.ollama_verifier import DEFAULT_ENDPOINT, DEFAULT_MODEL
    import requests

    tags_endpoint = DEFAULT_ENDPOINT.rsplit("/", 1)[0] + "/tags"
    try:
        response = requests.get(tags_endpoint, timeout=1.0)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", []) if isinstance(payload, dict) else []
        reachable = True
        model_available = any(
            isinstance(item, dict) and item.get("name") == DEFAULT_MODEL
            for item in models
        )
    except Exception:
        reachable = False
        model_available = False
    worker_ready = _warm_worker.is_ready
    worker_state = _warm_worker.ready or {}
    bge_ready = bool(worker_ready and worker_state.get("bge_ready"))
    bge_fallback_active = bool(worker_ready and not bge_ready)
    verification_ready = bool(worker_ready and reachable and model_available)
    if not worker_ready:
        status = "unavailable"
    elif not reachable or not model_available or bge_fallback_active:
        status = "degraded"
    else:
        status = "ok"
    return {
        "status": status,
        "worker_ready": worker_ready,
        "bge_ready": bge_ready,
        "bge_fallback_active": bge_fallback_active,
        "worker_restart_count": _warm_worker.restarts,
        "verification_ready": verification_ready,
        "worker": {
            "ready": worker_ready,
            "pid": worker_state.get("pid"),
            "startup_seconds": worker_state.get("worker_startup_seconds"),
            "restart_count": _warm_worker.restarts,
            "last_startup_error": _warm_worker.last_startup_error,
        },
        "semantic_ranking": {
            "bge_ready": bge_ready,
            "fallback_active": bge_fallback_active,
            "error": worker_state.get("bge_error"),
        },
        "ollama": {
            "reachable": reachable,
            "model": DEFAULT_MODEL,
            "model_available": model_available,
            "ready": reachable and model_available,
        },
    }


@app.post("/api/verify")
async def verify(request: VerifyRequest):
    text = request.text.strip()
    if not text and request.image is None:
        raise HTTPException(status_code=400, detail="Provide claim text or an image.")
    try:
        state = await _run_pipeline_with_deadline(text, request.image)
        if (state.get("had_image")
                and "input_processing_error" in (state.get("quality_flags") or [])
                and not state.get("had_text")):
            return JSONResponse(status_code=422, content={
                "code": "image_processing_failed",
                "detail": "Screenshot could not be processed. Please try again.",
            })
        return adapt_graph_result(state)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Verification request failed")
        raise HTTPException(status_code=500, detail="Verification backend failed.") from None


@app.post("/api/conversation")
async def route_conversation(request: ConversationRequest):
    """Dispatch a conversation turn using the caller's existing session state."""
    intent = request.intent
    if intent == "NEW_VERIFICATION":
        if not request.message.strip() and request.image is None:
            raise HTTPException(status_code=400, detail="Provide claim text or an image.")
        try:
            state = await _run_pipeline_with_deadline(request.message.strip(), request.image)
            if (state.get("had_image")
                    and "input_processing_error" in (state.get("quality_flags") or [])
                    and not state.get("had_text")):
                raise HTTPException(status_code=422, detail="Screenshot could not be processed. Please try again.")
            result = adapt_graph_result(state)
            result["session_state"] = _session_state_from_graph(state, result, intent)
            return result
        except HTTPException:
            raise
        except Exception:
            logger.exception("Conversation verification failed")
            raise HTTPException(status_code=500, detail="Verification backend failed.") from None

    if intent == "GENERAL_QUERY":
        return {
            "intent": intent,
            "answer": (
                "I can explain a Raven result, show its sources, look for more evidence, "
                "or verify a new factual claim."
            ),
        }

    if intent in {"EXPLAIN_RESULT", "ASK_ABOUT_SOURCE"}:
        current_result = request.current_result
        if not current_result:
            return {
                "intent": intent,
                "answer": "There is no current verification result in this conversation yet.",
            }
        answer = await run_in_threadpool(_answer_followup, request.message.strip(), current_result)
        return {"intent": intent, "answer": answer}

    if intent == "MORE_EVIDENCE":
        try:
            state = await run_in_threadpool(_run_more_evidence, request.session_state)
            result = adapt_graph_result(state)
            result["session_state"] = _session_state_from_graph(state, result, intent)
            return result
        except HTTPException:
            raise
        except Exception:
            logger.exception("Additional evidence retrieval failed")
            raise HTTPException(status_code=500, detail="Additional evidence retrieval failed.") from None

    raise HTTPException(status_code=400, detail="Unsupported conversation intent.")


def _answer_followup(question: str, result: dict[str, Any]) -> str:
    verdict = str(result.get("verdict", "uncertain")).strip().capitalize()
    confidence = result.get("confidence")
    why = str(result.get("why") or "").strip()
    status = str(result.get("status") or "").strip()
    explanation = why or status
    evidence = result.get("evidence") or []
    question_text = question.casefold()

    def concise(value: str, limit: int = 260) -> str:
        value = " ".join(value.split())
        if len(value) > limit:
            return value[: limit - 1].rsplit(" ", 1)[0] + "…"
        return value

    source_summaries = []
    for item in evidence:
        source = str(item.get("source") or "").strip()
        if not source:
            continue
        details = []
        stance = str(item.get("stance") or "").strip()
        if stance:
            details.append(stance)
        relevance = item.get("relevance")
        trust = item.get("trust")
        if isinstance(relevance, (int, float)) and not isinstance(relevance, bool):
            details.append(f"relevance {relevance:g}%")
        if isinstance(trust, (int, float)) and not isinstance(trust, bool):
            details.append(f"trust {trust:g}%")
        source_summaries.append(source + (f" ({', '.join(details)})" if details else ""))

    evidence_question = any(word in question_text for word in (
        "evidence", "source", "sources", "support", "considered",
    ))
    if evidence_question:
        if source_summaries:
            return concise("Sources considered: " + "; ".join(source_summaries[:4]) + ".", 420)
        return "No source metadata was included with this result."

    why_question = any(word in question_text for word in (
        "why", "uncertain", "confidence", "confident", "result", "low",
    ))
    if why_question:
        parts = [f"Raven classified this result as {verdict}"]
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            parts[-1] += f" with {confidence:g}% confidence"
        if explanation:
            parts.append(concise(explanation))
        return concise(". ".join(parts).rstrip(".") + ".")

    parts = []
    if explanation:
        parts.append(concise(explanation))
    elif status:
        parts.append(concise(status))
    if source_summaries:
        parts.append("Sources: " + "; ".join(source_summaries[:3]) + ".")
    return concise(" ".join(parts)) if parts else f"Raven classified this result as {verdict}; no further explanation or source metadata was provided."


@app.post("/api/followup")
async def followup(request: FollowupRequest):
    question = request.question.strip()
    result = request.result
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if not isinstance(result.get("verdict"), str) or result.get("verdict") not in {"real", "fake", "uncertain"}:
        raise HTTPException(status_code=400, detail="Result must contain a supported verdict.")
    confidence = result.get("confidence")
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0 <= float(confidence) <= 100
    ):
        raise HTTPException(status_code=400, detail="Result confidence must be between 0 and 100.")
    if "evidence" in result and (not isinstance(result["evidence"], list) or any(not isinstance(item, dict) for item in result["evidence"])):
        raise HTTPException(status_code=400, detail="Result evidence must be a list of objects.")
    try:
        return {"answer": _answer_followup(question, result)}
    except Exception:
        logger.exception("Follow-up request failed")
        raise HTTPException(status_code=500, detail="Follow-up response failed.") from None
