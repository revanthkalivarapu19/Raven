"""
graph.py
========
RAVEN Project - Active LangGraph Workflow Orchestration

Active pipeline:
Input Processing -> Claim Extraction -> Domain Detection -> Evidence Verification -> Reflection -> Persona Analysis -> Supervisor -> XAI -> END

Preserves full segment provenance, language confidence, OCR metadata,
and quality flags in the shared state without any intermediate NLP pipeline.
"""

import logging
import json
import difflib
import os
import re
import sys
import time
from typing import Optional, Dict, Any

from langgraph.graph import StateGraph, START, END

from agents.state import RavenState
from agents.input_processing.pipeline import process_input
from agents.claim_extraction.claim_extraction_agent import ClaimExtractionAgent
from agents.domain_detection.domain_agent import DomainDetectionAgent
from rag_ingestion.pipeline.claim_processing import ClaimProcessingPipeline
from agents.reflection.reflection_agent import ReflectionAgent
from agents.supervisior.supervisor_agent import run_supervisor
from agents.xai.xai_agent import XAIAgent
from agents.claim_extraction.services.llm_service import LLMService
from rag_ingestion.utils.temporal_context import build_temporal_context
from rag_ingestion.supervisor.supervisor_agent import SupervisorAgent
from rag_ingestion.personas.journalist_persona import JournalistPersona
from rag_ingestion.personas.legal_persona import LegalPersona
from rag_ingestion.personas.scientific_persona import ScientificPersona
from rag_ingestion.verification.verification_schema import (
    OverallVerification,
    VerificationFailure,
)
from rag_ingestion.config.config import config as _config
from rag_ingestion.llm_resilience import request_deadline

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3


def _request_deadline_exceeded(state: RavenState) -> bool:
    deadline = state.get("request_deadline_at")
    return bool(deadline and time.monotonic() >= deadline)


def _stage_budget_available(state: RavenState, reserve_seconds: float = 0.0) -> bool:
    deadline = state.get("request_deadline_at")
    return not deadline or (deadline - time.monotonic()) > reserve_seconds


def _instrument_node(name, node):
    def wrapped(state):
        started = time.monotonic()
        result = node(state)
        finished = time.monotonic()
        diagnostics = dict(state.get("request_diagnostics") or {})
        node_times = dict(diagnostics.get("nodes") or {})
        node_times[name] = {
            "start": started,
            "end": finished,
            "duration_seconds": finished - started,
            "remaining_before": max(0.0, state.get("request_deadline_at", finished) - started),
            "duration_ms": (finished - started) * 1000.0,
        }
        diagnostics["nodes"] = node_times
        diagnostics.setdefault("stage_timings_ms", {})[name] = (finished - started) * 1000.0
        logger.info("Stage %s completed in %.1f ms", name, (finished - started) * 1000.0)
        diagnostics["attempt_count"] = result.get("attempt_count", state.get("attempt_count", 1))
        result["request_diagnostics"] = diagnostics
        return result
    return wrapped
HIGH_CONFIDENCE_THRESHOLD = 0.70


def _build_retrieval_queries(
    claim: str, retrieval_context: Optional[str] = None, domain: Optional[str] = None
) -> list[str]:
    """Ask the model for focused retrieval queries, with a safe claim fallback."""
    claim_text = (claim or "").strip()
    context_text = (retrieval_context or "").strip()
    if not claim_text:
        return []

    prompt = f"""Generate exactly 3 or 4 concise search queries for fact-checking.
Return search phrases only: keyword or phrase searches, never questions or sentences.
Do not use question marks. Preserve names, organizations, locations, numbers, dates,
and multi-word phrases exactly as copied from the claim or context, including spelling
and whitespace. Never join, split, rewrite, or invent words or entities. Keep spaces
between every word and entity; never concatenate separate words. Use only information
present in the claim and context. If the claim contains a year, month, explicit date,
or relative temporal phrase, preserve that exact temporal phrase in at least one query;
never invent an absolute date from a relative phrase.
Return ONLY valid JSON in the form {{\"queries\": [\"query 1\", \"query 2\", \"query 3\"]}}.
Do not answer the claim or explain your reasoning.

Detected domain: {domain or "unknown"}
Claim: {claim_text}
Retrieval context: {context_text}
"""
    source = f"{claim_text} {context_text}"
    source_tokens = {
        token.casefold()
        for token in re.findall(r"[^\W_]+(?:[.%][^\W_]+)*", source)
    }

    def normalized_token(token: str) -> str:
        return re.sub(r"[^\w]", "", token.casefold())

    source_normalized_tokens = {
        normalized_token(token) for token in source_tokens if normalized_token(token)
    }

    temporal_terms = re.findall(
        r"\b(?:19|20)\d{2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:19|20)\d{2}\b|"
        r"\b(?:recently|yesterday|today|tomorrow|currently|now|earlier|later)\b|"
        r"\b(?:last|this|next)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|week|month|year|day|quarter)\b|"
        r"\b(?:on\s+)?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b|"
        r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[/-]\d{1,2}[/-](?:19|20)?\d{2}\b",
        claim_text,
        flags=re.IGNORECASE,
    )

    def is_grounded_token(token: str) -> bool:
        """Allow formatting/inflection variants without accepting new concepts."""
        normalized = normalized_token(token)
        if not normalized:
            return False
        if normalized in source_normalized_tokens:
            return True
        if any(character.isdigit() for character in normalized):
            return False
        for source_token in source_normalized_tokens:
            if len(normalized) >= 4 and len(source_token) >= 4:
                if normalized.startswith(source_token) or source_token.startswith(normalized):
                    return True
                if difflib.SequenceMatcher(None, normalized, source_token).ratio() >= 0.70:
                    return True
        return False

    def split_source_token(token: str):
        """Return source-word parts when a token is a glued source phrase."""
        token = token.casefold()
        if token in source_tokens:
            return [token]
        possible = [False] * (len(token) + 1)
        paths = [[] for _ in range(len(token) + 1)]
        possible[0] = True
        for end in range(1, len(token) + 1):
            for start in range(end):
                piece = token[start:end]
                if possible[start] and piece in source_tokens:
                    possible[end] = True
                    paths[end] = paths[start] + [piece]
                    break
        return paths[-1] if possible[-1] and len(paths[-1]) >= 2 else None

    failed_queries = []
    for attempt in range(2):
        try:
            retry_feedback = ""
            if failed_queries:
                retry_feedback = (
                    "\nThe previous attempt failed validation for these generated "
                    "queries:\n"
                    + "\n".join(
                        f"- {query}: {reason}" for query, reason in failed_queries
                    )
                    + "\nRegenerate all queries using only grounded source terms.\n"
                )
            raw = LLMService().generate(prompt + retry_feedback)
            parsed = json.loads(raw)
            candidates = parsed.get("queries", []) if isinstance(parsed, dict) else parsed
            if not isinstance(candidates, list) or not 3 <= len(candidates) <= 4:
                raise ValueError("query planner did not return 3-4 queries")
            queries = []
            seen = set()
            failed_queries = []
            for candidate in candidates:
                if not isinstance(candidate, str):
                    failed_queries.append((str(candidate), "query is not text"))
                    continue
                query = re.sub(r"\s+", " ", candidate).strip().strip("` ")
                query_tokens = re.findall(
                    r"[^\W_]+(?:[.%][^\W_]+)*", query
                )
                if not query or "?" in query or not query_tokens:
                    failed_queries.append((query, "not a concise search phrase"))
                    continue
                ungrounded_numbers = [
                    token
                    for token in query_tokens
                    if any(character.isdigit() for character in token)
                    and not is_grounded_token(token)
                ]
                if ungrounded_numbers:
                    failed_queries.append(
                        (
                            query,
                            "contains ungrounded numeric/date value(s): "
                            + ", ".join(ungrounded_numbers),
                        )
                    )
                    continue
                repaired_query = query
                grounded_count = 0
                for token in query_tokens:
                    parts = split_source_token(token)
                    grounded = parts is not None or is_grounded_token(token)
                    if grounded:
                        grounded_count += 1
                    if parts is not None and len(parts) > 1:
                        repaired_query = re.sub(
                            re.escape(token), " ".join(parts), repaired_query,
                            count=1, flags=re.IGNORECASE,
                        )
                overlap = grounded_count / len(query_tokens)
                if overlap < 0.60:
                    failed_queries.append(
                        (query, "insufficient grounded term overlap")
                    )
                    continue
                query = re.sub(r"\s+", " ", repaired_query).strip()
                key = query.casefold()
                if key not in seen:
                    seen.add(key)
                    queries.append(query)
                if len(queries) == 4:
                    break
            if 3 <= len(queries) <= 4:
                if temporal_terms:
                    combined_queries = " ".join(queries).casefold()
                    missing_temporal_terms = [
                        term for term in temporal_terms
                        if term.casefold() not in combined_queries
                    ]
                    if missing_temporal_terms:
                        failed_queries.append(
                            ("; ".join(queries),
                             "claim temporal information was omitted: "
                             + ", ".join(missing_temporal_terms))
                        )
                        raise ValueError("query planner omitted claim temporal information")
                return queries
            raise ValueError("query planner returned malformed or ungrounded queries")
        except Exception as error:
            logger.warning(
                "Retrieval query planning attempt %d failed: %s", attempt + 1, error
            )
    return [claim_text]


# Singletons for downstream agents to avoid re-instantiation overhead
_CLAIM_AGENT = None
_DOMAIN_AGENT = None
_CLAIM_PROCESSING_PIPELINE = None
_REFLECTION_AGENT = None
_PERSONA_AGENTS = None
_SUPERVISOR_AGENT = run_supervisor
_XAI_AGENT = None


def get_claim_agent() -> ClaimExtractionAgent:
    global _CLAIM_AGENT
    if _CLAIM_AGENT is None:
        _CLAIM_AGENT = ClaimExtractionAgent()
    return _CLAIM_AGENT


def get_domain_agent() -> DomainDetectionAgent:
    global _DOMAIN_AGENT
    if _DOMAIN_AGENT is None:
        _DOMAIN_AGENT = DomainDetectionAgent()
    return _DOMAIN_AGENT


def get_claim_processing_pipeline() -> ClaimProcessingPipeline:
    global _CLAIM_PROCESSING_PIPELINE
    if _CLAIM_PROCESSING_PIPELINE is None:
        _CLAIM_PROCESSING_PIPELINE = ClaimProcessingPipeline()
    return _CLAIM_PROCESSING_PIPELINE


def get_reflection_agent() -> ReflectionAgent:
    global _REFLECTION_AGENT
    if _REFLECTION_AGENT is None:
        _REFLECTION_AGENT = ReflectionAgent()
    return _REFLECTION_AGENT


def get_persona_agents():
    global _PERSONA_AGENTS

    if _PERSONA_AGENTS is None:
        _PERSONA_AGENTS = [
            JournalistPersona(),
            LegalPersona(),
            ScientificPersona(),
        ]

    return _PERSONA_AGENTS


def get_supervisor_agent():
    return _SUPERVISOR_AGENT


def get_xai_agent() -> XAIAgent:
    global _XAI_AGENT
    if _XAI_AGENT is None:
        _XAI_AGENT = XAIAgent()
    return _XAI_AGENT


# -----------------------------------------------------------------------------
# LangGraph Nodes
# -----------------------------------------------------------------------------

def input_processing_node(state: RavenState) -> Dict[str, Any]:
    """
    Node 1: Input Processing
    Processes raw text and/or image into tagged segments, normalized text,
    and translated claim-ready content.
    """
    logger.info("Executing Graph Node: [input_processing]")
    text = state.get("text")
    image_path = state.get("image_path")

    try:
        result = process_input(text=text, image_path=image_path)
        return result
    except Exception as e:
        logger.exception(f"Error in input_processing_node: {e}")
        return {
            "processed_text": (text or "").strip(),
            "pre_translation_text": (text or "").strip(),
            "original_language": "en",
            "language_confidence": 0.0,
            "was_translated": False,
            "had_text": bool(text),
            "had_image": bool(image_path),
            "segments": [],
            "ocr_data": {},
            "ocr_confidence": None,
            "quality_flags": ["input_processing_error"],
            "errors": [str(e)]
        }


def claim_extraction_node(state: RavenState) -> Dict[str, Any]:
    """
    Node 2: Claim Extraction
    Consumes processed_text directly from state and extracts the concise main claim.
    """
    logger.info("Executing Graph Node: [claim_extraction]")
    if state.get("request_deadline_at") and not _stage_budget_available(state, _config.timeout_seconds):
        return {"claim": "", "errors": state.get("errors", []) + ["request_deadline_exceeded: claim extraction skipped"]}
    claim_input = state.get("processed_text") or state.get("text") or ""

    if not claim_input.strip():
        logger.warning("No text available for claim extraction.")
        return {"claim": ""}

    try:
        agent = get_claim_agent()
        claim = agent.extract_claim(claim_input)
        from rag_ingestion.verification.claim_scope import classify_claim_scope
        return {
            "claim": claim,
            "claim_scope": state.get("claim_scope") or classify_claim_scope(claim),
        }
    except Exception as e:
        logger.exception(f"Error in claim_extraction_node: {e}")
        from rag_ingestion.verification.claim_scope import classify_claim_scope
        return {
            "claim": claim_input.strip(),
            "claim_scope": state.get("claim_scope") or classify_claim_scope(claim_input.strip()),
            "errors": (state.get("errors", []) + [f"claim_extraction_error: {e}"])
        }


def domain_detection_node(state: RavenState) -> Dict[str, Any]:
    """
    Node 3: Domain Detection
    Consumes extracted claim from state and detects the subject domain.
    """
    logger.info("Executing Graph Node: [domain_detection]")
    if state.get("request_deadline_at") and not _stage_budget_available(state, _config.timeout_seconds):
        return {"domain": {"domain": "Unknown", "confidence": 0.0, "reason": "request deadline exceeded"}, "errors": state.get("errors", []) + ["request_deadline_exceeded: domain detection skipped"]}
    claim = state.get("claim") or state.get("processed_text") or ""

    if not claim.strip():
        logger.warning("No claim available for domain detection.")
        return {
            "domain": {
                "domain": "Unknown",
                "confidence": 0.0,
                "reason": "No claim available for domain detection."
            }
        }

    try:
        agent = get_domain_agent()
        domain_result = agent.detect_domain(claim)
        return {"domain": domain_result}
    except Exception as e:
        logger.exception(f"Error in domain_detection_node: {e}")
        return {
            "domain": {
                "domain": "Unknown",
                "confidence": 0.0,
                "reason": f"domain_detection_error: {e}"
            },
            "errors": (state.get("errors", []) + [f"domain_detection_error: {e}"])
        }


def _legacy_evidence_verification_node(state: RavenState) -> Dict[str, Any]:
    """Node 4: Evidence retrieval, fusion, and verification."""
    logger.info("Executing Graph Node: [evidence_verification]")

    claim = state.get("claim") or ""
    domain_state = state.get("domain") or {}

    if isinstance(domain_state, dict):
        domain = domain_state.get("domain") or "Unknown"
    else:
        domain = str(domain_state)

    if not claim.strip():
        logger.warning("No claim available for evidence verification.")
        return {
            "local_evidence": [],
            "external_evidence": [],
            "fused_evidence": [],
            "verification_result": None,
            "verification_failure": None,
            "errors": state.get("errors", []) + [
                "evidence_verification_error: no claim available"
            ],
        }

    if not domain.strip() or domain.lower() == "unknown":
        logger.warning("No valid domain available for evidence verification.")
        return {
            "local_evidence": [],
            "external_evidence": [],
            "fused_evidence": [],
            "verification_result": None,
            "verification_failure": None,
            "errors": state.get("errors", []) + [
                "evidence_verification_error: no valid domain available"
            ],
        }

    try:
        pipeline = get_claim_processing_pipeline()

        retrieval_context = (
            state.get("processed_text")
            or state.get("text")
            or claim
        ).strip()

        retrieval_queries = _build_retrieval_queries(
            claim=claim,
            retrieval_context=retrieval_context,
            domain=domain,
        )

        logger.info(
            "Running %d focused evidence retrieval queries.", len(retrieval_queries)
        )

        errors = list(state.get("errors", []))
        local_evidence_by_id = {}
        external_evidence_by_id = {}
        local_retrieval_error = None

        for retrieval_query in retrieval_queries:
            try:
                local_evidence = pipeline.retrieval_manager.search(
                    claim=retrieval_query,
                    domain=domain,
                    top_k=5,
                )
            except FileNotFoundError as retrieval_error:
                logger.warning(
                    "Local retrieval unavailable for domain %s: %s",
                    domain,
                    retrieval_error,
                )
                local_evidence = []
                if local_retrieval_error is None:
                    local_retrieval_error = retrieval_error

            external_result = pipeline.external_manager.search(
                query=retrieval_query,
                domain=domain,
                top_k=5,
            )

            for item in local_evidence:
                local_evidence_by_id[item.evidence_id] = item

            for item in external_result.evidence:
                try:
                    normalized_item = pipeline.normalizer.normalize(item)
                except Exception as normalization_error:
                    logger.warning(
                        "Normalization failed for external evidence %s: %s",
                        item.evidence_id,
                        normalization_error,
                    )
                    errors.append(
                        f"external_retrieval_error: {normalization_error}"
                    )
                    continue
                external_evidence_by_id[normalized_item.evidence_id] = normalized_item

            for error in external_result.errors:
                errors.append(f"external_retrieval_error: {error}")

        local_evidence = list(local_evidence_by_id.values())
        external_evidence = list(external_evidence_by_id.values())
        if local_retrieval_error is not None:
            errors.append(
                f"local_retrieval_error: {local_retrieval_error}"
            )

        fused_evidence = pipeline.fusion_manager.fuse(
            local_evidence=local_evidence,
            external_evidence=external_evidence,
            top_k=5,
        )
        verification_failure = None
        try:
            verification_result = pipeline.verification_manager.verify(
                claim=claim,
                evidence_list=fused_evidence,
            )
        except Exception as verification_error:
            logger.error("Verification failed: %s", verification_error)
            verification_result = None
            verification_failure = VerificationFailure.from_exception(
                claim=claim,
                evidence_count=len(fused_evidence),
                exc=verification_error,
            )
            errors.append(
                f"verification_error: {verification_failure}"
            )

        return {
            "local_evidence": local_evidence,
            "external_evidence": external_evidence,
            "fused_evidence": fused_evidence,
            "verification_result": verification_result,
            "verification_failure": verification_failure,
            "errors": errors,
        }

    except Exception as e:
        logger.exception(
            f"Error in evidence_verification_node: {e}"
        )

        return {
            "local_evidence": [],
            "external_evidence": [],
            "fused_evidence": [],
            "verification_result": None,
            "verification_failure": None,
            "errors": state.get("errors", []) + [
                f"evidence_verification_error: {e}"
            ],
        }



def evidence_verification_node(state: RavenState) -> Dict[str, Any]:
    """Build retrieval queries and delegate retrieval/verification to the pipeline."""
    logger.info("Executing Graph Node: [evidence_verification]")
    if state.get("request_deadline_at") and not _stage_budget_available(state, 2 * _config.timeout_seconds):
        remaining = state["request_deadline_at"] - time.monotonic()
        logger.info(
            "evidence_diag graph_stage_skip stage=evidence_verification "
            "reason=insufficient_reserved_budget remaining_seconds=%.2f "
            "required_reserve_seconds=%.2f deadline_reached=%s",
            max(0.0, remaining),
            2 * _config.timeout_seconds,
            remaining <= 0,
        )
        deadline_exceeded = _request_deadline_exceeded(state)
        failure = VerificationFailure(
            claim=state.get("claim") or "",
            evidence_count=len(state.get("fused_evidence") or []),
            error_type=(
                "RequestDeadlineExceeded"
                if deadline_exceeded
                else "InsufficientRemainingBudget"
            ),
            message=(
                "Request deadline was reached before evidence verification."
                if deadline_exceeded
                else "Evidence verification was skipped because the remaining execution budget was insufficient."
            ),
        )
        return {
            "local_evidence": [], "external_evidence": [], "fused_evidence": [],
            "verification_result": None,
            "verification_failure": failure,
            "errors": state.get("errors", []) + [
                "request_deadline_exceeded: verification skipped"
                if deadline_exceeded
                else "verification_skipped: insufficient_remaining_budget"
            ],
        }
    claim = state.get("claim") or ""
    domain_state = state.get("domain") or {}
    domain = (
        domain_state.get("domain") or "Unknown"
        if isinstance(domain_state, dict)
        else str(domain_state)
    )
    if not claim.strip() or not domain.strip() or domain.lower() == "unknown":
        reason = "missing claim" if not claim.strip() else "missing or unsupported domain"
        failure = VerificationFailure(
            claim=claim,
            evidence_count=0,
            error_type="UpstreamPrerequisiteError",
            message=f"Evidence verification could not start: {reason}.",
        )
        return {
            "local_evidence": [], "external_evidence": [], "fused_evidence": [],
            "verification_result": None, "verification_failure": failure,
            "errors": state.get("errors", []) + [
                f"evidence_verification_error: {reason}"
            ],
        }
    try:
        retrieval_context = (
            state.get("processed_text") or state.get("text") or claim
        ).strip()
        queries = _build_retrieval_queries(
            claim=claim, retrieval_context=retrieval_context, domain=domain
        )
        result = get_claim_processing_pipeline().process(
            claim=claim, claim_scope=state.get("claim_scope"),
            domain=domain, retrieval_queries=queries,
            top_k_local=5, top_k_external=5, top_k_fused=5,
        )
        errors = list(state.get("errors", []))
        errors.extend(
            "external_retrieval_error: "
            f"provider={error.provider_name} category={error.category}"
            for error in result.external_source_errors
        )
        if result.local_retrieval_error is not None:
            errors.append(f"local_retrieval_error: {result.local_retrieval_error}")
        if result.verification_failure is not None:
            errors.append(f"verification_error: {result.verification_failure}")

        local_evidence = result.local_evidence
        external_evidence = result.external_evidence
        fused_evidence = result.fused_evidence
        verification_result = result.verification_result
        verification_failure = result.verification_failure
        if not fused_evidence and state.get("fused_evidence"):
            logger.info(
                "Preserving previous evidence and verification after an empty retry result."
            )
            local_evidence = state.get("local_evidence") or local_evidence
            external_evidence = state.get("external_evidence") or external_evidence
            fused_evidence = state["fused_evidence"]
            verification_result = state.get("verification_result") or verification_result
            verification_failure = state.get("verification_failure") or verification_failure

        if verification_failure is not None:
            verification_result = None
        elif not isinstance(verification_result, OverallVerification):
            invalid_result = TypeError(
                "claim-processing pipeline returned no valid verification result or failure"
            )
            verification_failure = VerificationFailure.from_exception(
                claim=claim,
                evidence_count=len(fused_evidence),
                exc=invalid_result,
            )
            verification_result = None
            errors.append(f"verification_error: {verification_failure}")

        return {
            "local_evidence": local_evidence,
            "external_evidence": external_evidence,
            "fused_evidence": fused_evidence,
            "verification_result": verification_result,
            "verification_failure": verification_failure,
            "errors": errors,
        }
    except Exception as e:
        logger.exception("Error in evidence_verification_node: %s", e)
        failure = VerificationFailure.from_exception(
            claim=claim,
            evidence_count=len(state.get("fused_evidence") or []),
            exc=e,
        )
        return {
            "local_evidence": [], "external_evidence": [], "fused_evidence": [],
            "verification_result": None, "verification_failure": failure,
            "errors": state.get("errors", []) + [
                f"evidence_verification_error: {failure}"
            ],
        }


def reflection_node(state: RavenState) -> Dict[str, Any]:
    """Node 5: Reflect on evidence quality, credibility, and sufficiency."""
    logger.info("Executing Graph Node: [reflection]")
    verification_failure = state.get("verification_failure")
    verification_result = state.get("verification_result")
    explicit_failure_state = (
        isinstance(verification_result, dict)
        and str(verification_result.get("status", "")).upper() == "VERIFICATION_FAILURE"
    )
    if verification_failure is not None or explicit_failure_state:
        failure = verification_failure
        reason = (
            getattr(failure, "message", None)
            or (failure.get("message") if isinstance(failure, dict) else None)
            or "Verifier returned no result."
        )
        attempt = state.get("attempt_count", 1)
        return {
            "reflection_result": {
                "decision": "NO", "is_evidence_sufficient": False,
                "confidence": 0.0, "claim_supported": False,
                "claim_contradicted": False, "notes": f"Verification failed: {reason}",
                "support_score": 0.0, "verification_failed": True,
                "reason": str(reason), "retry_requested": True,
                "attempt_number": attempt,
            },
            "errors": state.get("errors", []) + ["reflection_deterministic: verification failed"],
        }
    if state.get("request_deadline_at") and not _stage_budget_available(state, 6 * _config.timeout_seconds):
        remaining = state["request_deadline_at"] - time.monotonic()
        deadline_exceeded = _request_deadline_exceeded(state)
        logger.info(
            "evidence_diag graph_stage_skip stage=reflection "
            "reason=insufficient_reserved_budget remaining_seconds=%.2f "
            "required_reserve_seconds=%.2f deadline_reached=%s",
            max(0.0, remaining),
            6 * _config.timeout_seconds,
            remaining <= 0,
        )
        skip_reason = (
            "request_deadline_exceeded: reflection skipped"
            if deadline_exceeded
            else "reflection_skipped: insufficient_remaining_budget"
        )
        note = (
            "Request deadline exceeded."
            if deadline_exceeded
            else "Reflection skipped because the remaining execution budget was considered insufficient for the reflection step."
        )
        return {"reflection_result": {"decision": "NO", "is_evidence_sufficient": False, "confidence": 0.0, "claim_supported": False, "claim_contradicted": False, "notes": note, "support_score": 0.0}, "errors": state.get("errors", []) + [skip_reason]}

    claim = state.get("claim") or ""
    domain_state = state.get("domain") or {}

    if isinstance(domain_state, dict):
        domain = domain_state.get("domain") or "General"
    else:
        domain = str(domain_state) if domain_state else "General"

    evidence_list = state.get("fused_evidence") or []

    if not claim.strip():
        logger.warning("No claim available for reflection.")
        return {
            "reflection_result": {
                "decision": "NO",
                "is_evidence_sufficient": False,
                "contradictions_found": False,
                "confidence": 0.0,
                "notes": "No claim was provided.",
                "support_score": 0.0,
                "source_credibility_score": 0.0,
                "evidence_quality_score": 0.0,
                "claim_supported": False,
                "claim_contradicted": False,
                "evidence_items_contradict_each_other": False,
            },
            "errors": state.get("errors", []) + [
                "reflection_error: no claim available"
            ],
        }

    try:
        agent = get_reflection_agent()

        reflection_result = agent.reflect(
            claim_text=claim,
            domain=domain,
            evidence_list=evidence_list,
            verification_result=state.get("verification_result"),
        )

        return {
            "reflection_result": reflection_result,
        }

    except Exception as e:
        logger.exception(f"Error in reflection_node: {e}")

        return {
            "reflection_result": {
                "decision": "NO",
                "is_evidence_sufficient": False,
                "contradictions_found": False,
                "confidence": 0.0,
                "notes": f"Reflection failed: {e}",
                "support_score": 0.0,
                "source_credibility_score": 0.0,
                "evidence_quality_score": 0.0,
                "claim_supported": False,
                "claim_contradicted": False,
                "evidence_items_contradict_each_other": False,
            },
            "errors": state.get("errors", []) + [
                f"reflection_error: {e}"
            ],
        }



def persona_analysis_node(state: RavenState) -> Dict[str, Any]:
    """Node 6: Run journalist, legal, and scientific persona analyses."""
    logger.info("Executing Graph Node: [persona_analysis]")
    persona_and_downstream_reserve = 8 * _config.timeout_seconds
    if state.get("request_deadline_at") and not _stage_budget_available(state, persona_and_downstream_reserve):
        remaining = state["request_deadline_at"] - time.monotonic()
        deadline_exceeded = _request_deadline_exceeded(state)
        logger.info(
            "evidence_diag graph_stage_skip stage=persona_analysis "
            "reason=insufficient_reserved_budget remaining_seconds=%.2f "
            "required_reserve_seconds=%.2f deadline_reached=%s",
            max(0.0, remaining),
            persona_and_downstream_reserve,
            remaining <= 0,
        )
        skip_reason = (
            "request_deadline_exceeded: persona analysis skipped"
            if deadline_exceeded
            else "persona_analysis_skipped: insufficient_remaining_budget"
        )
        return {"persona_insights": [], "errors": state.get("errors", []) + [skip_reason]}

    claim = state.get("claim") or ""
    domain_state = state.get("domain") or {}
    evidence = state.get("fused_evidence") or []
    verification = state.get("verification_result")

    if isinstance(domain_state, dict):
        domain = domain_state.get("domain") or "General"
    else:
        domain = str(domain_state) if domain_state else "General"

    if not claim.strip():
        return {
            "persona_insights": [],
            "errors": state.get("errors", []) + [
                "persona_analysis_error: no claim available"
            ],
        }

    if not evidence:
        return {
            "persona_insights": [],
            "errors": state.get("errors", []) + [
                "persona_analysis_error: no fused evidence available"
            ],
        }

    if verification is None:
        return {
            "persona_insights": [],
            "errors": state.get("errors", []) + [
                "persona_analysis_error: no verification result available"
            ],
        }

    allowed_evidence_ids = {getattr(item, "evidence_id", None) for item in evidence}
    allowed_evidence_ids.discard(None)

    try:
        context = SupervisorAgent.build_context(
            claim=claim,
            domain=domain,
            evidence=evidence,
            verification_results=verification.results,
        )

        insights = []
        errors = list(state.get("errors", []))

        for persona in get_persona_agents():
            try:
                insight = persona.analyze(context)
                raw_ids = (
                    insight.get("evidence_ids") if isinstance(insight, dict)
                    else getattr(insight, "evidence_ids", None)
                )
                if (not isinstance(raw_ids, list) or not raw_ids
                        or any(not isinstance(eid, str) or eid not in allowed_evidence_ids for eid in raw_ids)):
                    raise ValueError("persona output contains invalid evidence_ids")
                insights.append(insight)

                logger.info(
                    "Persona [%s] completed successfully.",
                    persona.name,
                )

            except Exception as persona_error:
                logger.exception(
                    "Persona [%s] failed: %s",
                    persona.name,
                    persona_error,
                )
                errors.append(
                    f"persona_{persona.name}_error: {persona_error}"
                )

        return {
            "persona_insights": insights,
            "errors": errors,
        }

    except Exception as e:
        logger.exception(
            f"Error building PersonaContext: {e}"
        )

        return {
            "persona_insights": [],
            "errors": state.get("errors", []) + [
                f"persona_analysis_error: {e}"
            ],
        }



def supervisor_node(state: RavenState) -> Dict[str, Any]:
    """Node 7: Produce the final fact-checking verdict using evidence,
    reflection, and persona analyses.
    """
    logger.info("Executing Graph Node: [supervisor]")
    from rag_ingestion.verification.claim_scope import (
        classify_claim_scope, verification_can_establish_claim,
    )
    if state.get("request_deadline_at") and not _stage_budget_available(state, _config.timeout_seconds):
        deadline_exceeded = _request_deadline_exceeded(state)
        if deadline_exceeded:
            reasoning_summary = "Request deadline exceeded before supervisor completion."
            skip_reason = "request_deadline_exceeded: supervisor fallback"
        else:
            reasoning_summary = (
                "Supervisor was skipped because the remaining execution budget "
                "was insufficient for the supervisor step."
            )
            skip_reason = "supervisor_skipped: insufficient_remaining_budget"
        return {
            "supervisor_result": {
                "final_verdict": "Unverified",
                "confidence_score": 0,
                "trust_score": 0,
                "reasoning_summary": reasoning_summary,
                "key_evidence_used": [],
            },
            "errors": state.get("errors", []) + [skip_reason],
        }

    claim = state.get("claim") or ""
    domain_state = state.get("domain") or {}
    evidence_list = state.get("fused_evidence") or []
    reflection_result = state.get("reflection_result")
    persona_insights = state.get("persona_insights") or []
    verification_result = state.get("verification_result")
    verification_failure = state.get("verification_failure")
    attempt_count = state.get("attempt_count", 1)
    maximum_attempts_reached = state.get("maximum_attempts_reached", False)

    if isinstance(domain_state, dict):
        domain = domain_state.get("domain") or "General"
    else:
        domain = str(domain_state) if domain_state else "General"

    try:
        def unverified_result(reason: str) -> Dict[str, Any]:
            return {
                "final_verdict": "Unverified",
                "confidence_score": 0,
                "trust_score": 0,
                "reasoning_summary": reason,
                "key_evidence_used": [],
            }

        # Safety boundary: the supervisor may not produce a verdict without a
        # successful, non-conflicting verification result.
        if verification_failure is not None:
            logger.warning(
                "Supervisor safety gate triggered: verification failure -> Unverified"
            )
            return {
                "supervisor_result": unverified_result(
                    "Evidence verification failed; no definitive verdict is available."
                ),
                "verification_failure": verification_failure,
            }

        if verification_result is None:
            logger.warning(
                "Supervisor safety gate triggered: missing verification -> Unverified"
            )
            return {
                "supervisor_result": unverified_result(
                    "Evidence verification was not completed; no definitive verdict is available."
                ),
            }

        if not claim.strip():
            logger.warning(
                "Supervisor safety gate triggered: missing claim -> Unverified"
            )
            return {
                "supervisor_result": unverified_result(
                    "The claim was unavailable; evidence verification could not be completed."
                ),
            }

        # Safety gate: unresolved conflicting evidence must not be converted
        # directly into a definitive Real/Fake verdict by the LLM.
        verification_assessment = None

        if verification_result is not None:
            if isinstance(verification_result, dict):
                verification_assessment = verification_result.get(
                    "overall_assessment"
                )
            else:
                verification_assessment = getattr(
                    verification_result,
                    "overall_assessment",
                    None,
                )

        claim_scope = state.get("claim_scope") or (
            verification_result.get("claim_scope")
            if isinstance(verification_result, dict)
            else getattr(verification_result, "claim_scope", None)
        ) or classify_claim_scope(claim)
        verification_results = (
            verification_result.get("results", [])
            if isinstance(verification_result, dict)
            else getattr(verification_result, "results", [])
        )

        def relationship_name(item):
            relationship = (
                item.get("relationship") if isinstance(item, dict)
                else getattr(item, "relationship", None)
            )
            return str(getattr(relationship, "value", relationship)).split(".")[-1]

        # Normalize Enum / string representations such as:
        # "CONFLICTING_EVIDENCE" or "VerificationAssessment.CONFLICTING_EVIDENCE"
        assessment_name = str(
            getattr(verification_assessment, "value", verification_assessment)
        ).split(".")[-1]

        if assessment_name in {"SUPPORTED", "CONTRADICTED"} and verification_results:
            required_relationship = (
                "SUPPORTS" if assessment_name == "SUPPORTED" else "CONTRADICTS"
            )
            has_eligible_result = any(
                relationship_name(item) == required_relationship
                and verification_can_establish_claim(item, claim_scope)
                for item in verification_results
            )
            if not has_eligible_result:
                return {
                    "supervisor_result": unverified_result(
                        "Verification found insufficient eligible evidence for a definitive verdict."
                    ),
                }

        if assessment_name in {
            "INSUFFICIENT_EVIDENCE",
            "CONFLICTING_EVIDENCE",
        }:
            conflict_evidence_ids = []

            for verification in verification_results:
                if relationship_name(verification) == "CONTRADICTS" and verification_can_establish_claim(verification, claim_scope):
                    evidence_id = getattr(
                        verification,
                        "evidence_id",
                        None,
                    )

                    if evidence_id is None and isinstance(verification, dict):
                        evidence_id = verification.get("evidence_id")

                    if evidence_id:
                        conflict_evidence_ids.append(str(evidence_id))

            valid_conflict = bool(conflict_evidence_ids)
            if assessment_name == "CONFLICTING_EVIDENCE" and valid_conflict:
                safety_reason = (
                    "Verification detected conflicting evidence. The system "
                    "will not issue a definitive Real/Fake verdict until the "
                    "conflict is explicitly resolved."
                )
            else:
                safety_reason = (
                    "Verification found insufficient evidence; the system "
                    "will not issue a definitive Real/Fake verdict."
                )

            logger.warning(
                "Supervisor safety gate triggered: %s -> Unverified",
                "CONFLICTING_EVIDENCE" if valid_conflict and assessment_name == "CONFLICTING_EVIDENCE"
                else "INSUFFICIENT_EVIDENCE",
            )

            return {
                "supervisor_result": {
                    "final_verdict": "Unverified",
                    "confidence_score": 0,
                    "trust_score": 0,
                    "reasoning_summary": safety_reason,
                    "key_evidence_used": conflict_evidence_ids,
                },
            }

        supervisor_input = {
            "claim_text": claim,
            "domain": domain,
            "evidence_list": evidence_list,
            "reflection_notes": reflection_result,
            "persona_insights": persona_insights,
            "verification_result": verification_result,
            "attempt_count": attempt_count,
            "maximum_attempts_reached": maximum_attempts_reached,
        }

        result = get_supervisor_agent()(supervisor_input)

        logger.info(
            "Supervisor completed with verdict: %s",
            result.get("final_verdict"),
        )

        return {
            "supervisor_result": result,
        }

    except Exception as e:
        logger.exception(
            f"Error in supervisor_node: {e}"
        )

        return {
            "supervisor_result": {},
            "errors": state.get("errors", []) + [
                f"supervisor_error: {e}"
            ],
        }



def xai_node(state: RavenState) -> Dict[str, Any]:
    """Node 8: Generate the final explainable output."""
    logger.info("Executing Graph Node: [xai]")

    claim = state.get("claim") or ""
    domain_state = state.get("domain") or {}
    evidence_list = state.get("fused_evidence") or []
    reflection_result = state.get("reflection_result") or {}
    supervisor_result = state.get("supervisor_result") or {}
    if state.get("request_deadline_at") and not _stage_budget_available(state, 4 * _config.timeout_seconds) and supervisor_result:
        remaining = state["request_deadline_at"] - time.monotonic()
        deadline_reached = _request_deadline_exceeded(state)
        logger.info(
            "evidence_diag graph_stage_skip stage=xai "
            "reason=insufficient_reserved_budget remaining_seconds=%.2f "
            "required_reserve_seconds=%.2f deadline_reached=%s",
            max(0.0, remaining),
            4 * _config.timeout_seconds,
            remaining <= 0,
        )
        verdict = str(supervisor_result.get("final_verdict") or "Unverified")
        verification = state.get("verification_result")
        assessment = (
            verification.get("overall_assessment")
            if isinstance(verification, dict)
            else getattr(verification, "overall_assessment", None)
        )
        assessment = str(getattr(assessment, "value", assessment)).split(".")[-1]
        if deadline_reached:
            explanation = (
                f"Raven determined the claim as {verdict}. The request deadline "
                "was reached before explanation generation."
            )
            skip_reason = "request_deadline_exceeded: XAI skipped"
        else:
            if assessment == "INSUFFICIENT_EVIDENCE":
                explanation = (
                    "Raven determined the claim as Unverified because the available "
                    "evidence was insufficient. A detailed explanation was not "
                    "generated because the remaining execution budget was considered "
                    "insufficient for the XAI step."
                )
            elif assessment == "CONFLICTING_EVIDENCE":
                explanation = (
                    "Raven determined the claim as Unverified because verification "
                    "found conflicting evidence. A detailed explanation was not "
                    "generated because the remaining execution budget was considered "
                    "insufficient for the XAI step."
                )
            else:
                explanation = (
                    f"Raven determined the claim as {verdict}. A detailed explanation "
                    "was not generated because the remaining execution budget was "
                    "considered insufficient for the XAI step."
                )
            skip_reason = "xai_skipped: insufficient_remaining_budget"
        return {
            "final_output": {
                "final_verdict": verdict,
                "confidence": supervisor_result.get("confidence_score", 0),
                "explanation": explanation,
            },
            "errors": state.get("errors", []) + [skip_reason],
        }
    attempt_count = state.get("attempt_count", 1)
    maximum_attempts_reached = state.get("maximum_attempts_reached", False)

    if isinstance(domain_state, dict):
        domain = domain_state.get("domain") or "General"
    else:
        domain = str(domain_state) if domain_state else "General"

    if not claim.strip():
        return {
            "final_output": {},
            "errors": state.get("errors", []) + [
                "xai_error: no claim available"
            ],
        }

    if not supervisor_result:
        return {
            "final_output": {},
            "errors": state.get("errors", []) + [
                "xai_error: no supervisor result available"
            ],
        }

    try:
        agent = get_xai_agent()

        final_output = agent.explain(
            claim_text=claim,
            domain=domain,
            evidence_list=evidence_list,
            reflection_result=reflection_result,
            supervisor_result=supervisor_result,
            attempt_count=attempt_count,
            maximum_attempts_reached=maximum_attempts_reached,
        )

        temporal_context = build_temporal_context(claim, evidence_list)
        final_output["temporal_context"] = temporal_context

        logger.info(
            "XAI completed with verdict: %s",
            final_output.get("final_verdict"),
        )

        return {
            "final_output": final_output,
            "temporal_context": temporal_context,
        }

    except Exception as e:
        logger.exception(f"Error in xai_node: {e}")

        verdict = str(supervisor_result.get("final_verdict") or "Unverified")
        if verdict.lower() == "supported":
            explanation = "Raven determined the claim as Supported. The explanation model was unavailable."
        elif verdict.lower() == "contradicted":
            explanation = "Raven determined the claim as Contradicted. The explanation model was unavailable."
        else:
            explanation = "Raven determined the claim as Unverified because the available evidence was insufficient. The explanation model was unavailable."

        return {
            "final_output": {
                "final_verdict": verdict,
                "confidence": supervisor_result.get("confidence_score", 0),
                "explanation": explanation,
            },
            "errors": state.get("errors", []) + [
                f"xai_error: {e}"
            ],
        }


def retry_evidence_node(state: RavenState) -> Dict[str, Any]:
    """Advance to the next bounded evidence-retrieval attempt."""
    current_attempt = state.get("attempt_count", 1)
    next_attempt = min(current_attempt + 1, MAX_ATTEMPTS)

    logger.info(
        "Retrying evidence retrieval: attempt %s/%s",
        next_attempt,
        MAX_ATTEMPTS,
    )

    return {
        "attempt_count": next_attempt,
        "maximum_attempts_reached": next_attempt >= MAX_ATTEMPTS,
    }


def should_retry_after_reflection(state: RavenState) -> str:
    """Route to another evidence attempt or continue to persona analysis."""
    current_attempt = state.get("attempt_count", 1)

    # Never allow an attempt beyond the configured maximum.
    if current_attempt >= MAX_ATTEMPTS:
        logger.info(
            "Maximum retrieval attempts reached: %s/%s. Continuing.",
            current_attempt,
            MAX_ATTEMPTS,
        )
        return "continue"

    deadline = state.get("request_deadline_at")
    if deadline:
        remaining = deadline - time.monotonic()
        # Leave enough budget for a complete bounded retrieval attempt. Starting
        # one with less than three normal LLM timeouts remaining only guarantees
        # another deadline failure and amplifies latency.
        if remaining <= max(0.0, 3 * _config.timeout_seconds):
            return "continue"

    reflection = state.get("reflection_result") or {}
    verification = state.get("verification_result")
    verification_failure = state.get("verification_failure")
    errors = [str(error).lower() for error in (state.get("errors") or [])]
    deterministic_provider_failure = any(
        "category=rate_limited" in error
        or "category=authentication/configuration" in error
        for error in errors
    )

    # Retrieval cannot repair an unavailable upstream claim/domain prerequisite.
    # Preserve the failure and continue to the existing safety path instead of
    # repeatedly invoking evidence retrieval with invalid inputs.
    domain_state = state.get("domain") or {}
    domain_name = (
        domain_state.get("domain") if isinstance(domain_state, dict) else str(domain_state)
    ) or ""
    if (
        not str(state.get("claim") or "").strip()
        or not domain_name.strip()
        or domain_name.strip().lower() == "unknown"
        or any(
            "claim_extraction_error" in error
            or "domain_detection_error" in error
            or "missing claim or valid domain" in error
            for error in errors
        )
    ):
        logger.info("No retrieval retry: upstream claim/domain prerequisite failed.")
        return "continue"

    if deterministic_provider_failure:
        logger.info(
            "No retrieval retry: a provider failure is not expected to change "
            "within the current request."
        )
        return "continue"

    # Verification failure means we should try retrieving evidence again.
    verification_failed_state = (
        verification_failure is not None
        or verification is None
        or (isinstance(verification, dict)
            and str(verification.get("status", "")).upper() == "VERIFICATION_FAILURE")
    )
    if verification_failed_state:
        logger.info("Retry requested: verification failure detected.")
        return "retry"

    # Verification assessment is authoritative for retrieval quality.
    assessment = None

    if verification is not None:
        if isinstance(verification, dict):
            assessment = verification.get("overall_assessment")
        else:
            assessment = getattr(
                verification,
                "overall_assessment",
                None,
            )

    assessment_name = str(
        getattr(assessment, "value", assessment)
    ).split(".")[-1]

    if assessment_name in {
        "CONFLICTING_EVIDENCE",
        "INSUFFICIENT_EVIDENCE",
    }:
        if (
            assessment_name == "INSUFFICIENT_EVIDENCE"
            and not state.get("fused_evidence")
            and any("external_retrieval_error" in error for error in errors)
        ):
            logger.info("No retrieval retry: no evidence and external providers failed.")
            return "continue"
        logger.info(
            "Retry requested: verification assessment=%s",
            assessment_name,
        )
        return "retry"

    # Preserve the existing 3-attempt reflection rule.
    decision = str(
        reflection.get("decision", "NO")
    ).upper()

    try:
        confidence = float(
            reflection.get("confidence", 0.0)
        )
    except (TypeError, ValueError):
        confidence = 0.0

    if decision != "YES" or confidence < HIGH_CONFIDENCE_THRESHOLD:
        logger.info(
            "Retry requested: reflection decision=%s confidence=%.2f",
            decision,
            confidence,
        )
        return "retry"

    logger.info(
        "Evidence accepted: reflection decision=%s confidence=%.2f",
        decision,
        confidence,
    )
    return "continue"


# -----------------------------------------------------------------------------
# Graph Builder
# -----------------------------------------------------------------------------

def build_raven_graph():
    """
    Assembles and compiles the active LangGraph workflow:
    Input Processing -> Claim Extraction -> Domain Detection -> Evidence Verification -> Reflection -> Persona Analysis -> Supervisor -> XAI -> END
    """
    workflow = StateGraph(RavenState)

    # Add Nodes
    workflow.add_node("input_processing", _instrument_node("input_processing", input_processing_node))
    workflow.add_node("claim_extraction", _instrument_node("claim_extraction", claim_extraction_node))
    workflow.add_node("domain_detection", _instrument_node("domain_detection", domain_detection_node))
    workflow.add_node("evidence_verification", _instrument_node("evidence_verification", evidence_verification_node))
    workflow.add_node("reflection", _instrument_node("reflection", reflection_node))
    workflow.add_node("persona_analysis", _instrument_node("persona_analysis", persona_analysis_node))
    workflow.add_node("supervisor", _instrument_node("supervisor", supervisor_node))
    workflow.add_node("xai", _instrument_node("xai", xai_node))

    # Add Linear Edges
    workflow.add_edge(START, "input_processing")
    workflow.add_edge("input_processing", "claim_extraction")
    workflow.add_edge("claim_extraction", "domain_detection")
    workflow.add_edge("domain_detection", "evidence_verification")
    workflow.add_edge("evidence_verification", "reflection")

    workflow.add_node("retry_evidence", _instrument_node("retry_evidence", retry_evidence_node))

    workflow.add_conditional_edges(
        "reflection",
        should_retry_after_reflection,
        {
            "retry": "retry_evidence",
            "continue": "persona_analysis",
        },
    )

    workflow.add_edge("retry_evidence", "evidence_verification")
    workflow.add_edge("persona_analysis", "supervisor")
    workflow.add_edge("supervisor", "xai")
    workflow.add_edge("xai", END)

    # Compile Graph
    app = workflow.compile()
    return app


# Cached compiled graph singleton
_COMPILED_GRAPH = None


def get_raven_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_raven_graph()
    return _COMPILED_GRAPH


def run_pipeline(text: Optional[str] = None, image_path: Optional[str] = None) -> RavenState:
    """
    Convenience executor running the complete active graph from START to END.
    """
    graph = get_raven_graph()
    started = time.monotonic()
    initial_state: RavenState = {
        "text": text,
        "image_path": image_path,
        "errors": [],
        "attempt_count": 1,
        "maximum_attempts_reached": False,
        "request_started_at": started,
        "request_deadline_at": started + _config.request_deadline_seconds,
        "request_diagnostics": {"request_deadline_seconds": _config.request_deadline_seconds},
    }
    with request_deadline(initial_state["request_deadline_at"]):
        final_state = graph.invoke(initial_state)
    finished = time.monotonic()
    diagnostics = dict(final_state.get("request_diagnostics") or {})
    diagnostics["total_duration_seconds"] = finished - started
    diagnostics["deadline_respected"] = finished <= initial_state["request_deadline_at"]
    diagnostics["deadline_reached"] = not diagnostics["deadline_respected"]
    final_state["request_diagnostics"] = diagnostics
    return final_state


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 70)
    print("  RAVEN LangGraph Pipeline: Input -> Claim -> Domain -> Evidence -> Reflection -> Persona -> Supervisor -> XAI")
    print("=" * 70)

    test_input = "The Reserve Bank of India reduced the repo rate by 0.50% on Friday to boost economic growth."
    print(f"\nRunning test pipeline with English input: {test_input}")

    state = run_pipeline(text=test_input)

    print("\n--- Final State ---")
    print("Original Language :", state.get("original_language"), f"(conf: {state.get('language_confidence')})")
    print("Was Translated    :", state.get("was_translated"))
    print("Segments Count    :", len(state.get("segments", [])))
    print("Processed Text    :", state.get("processed_text"))
    print("Extracted Claim   :", state.get("claim"))
    print("Detected Domain   :", state.get("domain"))
    print("Quality Flags     :", state.get("quality_flags"))
    print("Errors            :", state.get("errors"))
    print("=" * 70)
