from __future__ import annotations

import base64
import binascii
import logging
import math
import re
import tempfile
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from PIL import Image, UnidentifiedImageError

from agents.graph import run_pipeline
from rag_ingestion.verification.claim_scope import verification_can_establish_claim

logger = logging.getLogger(__name__)

_MAX_IMAGE_BYTES = 12 * 1024 * 1024
_MAX_REQUEST_BYTES = 17 * 1024 * 1024
_MAX_TEXT_CHARS = 20_000
_MAX_IMAGE_PIXELS = 40_000_000
_IMAGE_TYPES = {
    "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
    "image/bmp": ".bmp", "image/tiff": ".tif",
}
_DATA_URL_RE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.*)$", re.DOTALL)

app = FastAPI(title="Raven API", version="1.0.0")


class RequestBodyLimitMiddleware:
    """Bound request JSON before FastAPI parses it."""

    def __init__(self, app, max_body_bytes: int = _MAX_REQUEST_BYTES):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or scope.get("path") not in {"/api/verify", "/api/followup"}):
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
    text: str = Field(max_length=_MAX_TEXT_CHARS)
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
        excerpt = _get(item, "text") or ""
        if not source and not excerpt:
            continue
        reasoning = _get(verification_item, "reasoning") if verification_item is not None else None
        serialized_evidence.append({
            "source": str(source),
            "stance": stance,
            "relevance": _score_percent(_get(item, "similarity_score"), scale=1.0),
            "trust": _score_percent(_get(item, "authority_score"), scale=10.0),
            "excerpt": str(excerpt),
            "note": reasoning if isinstance(reasoning, str) else None,
        })

    result = {
        "verdict": verdict,
        "confidence": confidence,
        "status": status if isinstance(status, str) else "",
        "why": why if isinstance(why, str) else "",
        "evidence": serialized_evidence,
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
    return result


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


def _run_pipeline(text: str, image: Any):
    image_parts = _image_parts(image)
    if image_parts is None:
        return run_pipeline(text=text)
    image_bytes, suffix = image_parts
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="raven-upload-", suffix=suffix, delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_path = Path(temp_file.name)
        return run_pipeline(text=text or None, image_path=str(temp_path))
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/verify")
async def verify(request: VerifyRequest):
    text = request.text.strip()
    if not text and request.image is None:
        raise HTTPException(status_code=400, detail="Provide claim text or an image.")
    try:
        state = await run_in_threadpool(_run_pipeline, text, request.image)
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
