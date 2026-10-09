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
import os
import sys
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
from rag_ingestion.supervisor.supervisor_agent import SupervisorAgent
from rag_ingestion.personas.journalist_persona import JournalistPersona
from rag_ingestion.personas.legal_persona import LegalPersona
from rag_ingestion.personas.scientific_persona import ScientificPersona

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
HIGH_CONFIDENCE_THRESHOLD = 0.70

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
    claim_input = state.get("processed_text") or state.get("text") or ""

    if not claim_input.strip():
        logger.warning("No text available for claim extraction.")
        return {"claim": ""}

    try:
        agent = get_claim_agent()
        claim = agent.extract_claim(claim_input)
        return {"claim": claim}
    except Exception as e:
        logger.exception(f"Error in claim_extraction_node: {e}")
        return {
            "claim": claim_input.strip(),
            "errors": (state.get("errors", []) + [f"claim_extraction_error: {e}"])
        }


def domain_detection_node(state: RavenState) -> Dict[str, Any]:
    """
    Node 3: Domain Detection
    Consumes extracted claim from state and detects the subject domain.
    """
    logger.info("Executing Graph Node: [domain_detection]")
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


def evidence_verification_node(state: RavenState) -> Dict[str, Any]:
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

        retrieval_query = claim.strip()

        if (
            retrieval_context
            and retrieval_context.lower() != claim.strip().lower()
        ):
            retrieval_query = (
                f"{claim.strip()} "
                f"Context: {retrieval_context}"
            )

        logger.info(
            "Building evidence retrieval query from claim + input context."
        )

        result = pipeline.process(
            claim=claim,
            domain=domain,
            top_k_local=5,
            top_k_external=5,
            top_k_fused=5,
            retrieval_query=retrieval_query,
        )

        errors = list(state.get("errors", []))

        if result.local_retrieval_error is not None:
            errors.append(
                f"local_retrieval_error: {result.local_retrieval_error}"
            )

        for error in result.external_source_errors:
            errors.append(
                f"external_retrieval_error: {error}"
            )

        if result.verification_failure is not None:
            errors.append(
                f"verification_error: {result.verification_failure}"
            )

        return {
            "local_evidence": result.local_evidence,
            "external_evidence": result.external_evidence,
            "fused_evidence": result.fused_evidence,
            "verification_result": result.verification_result,
            "verification_failure": result.verification_failure,
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



def reflection_node(state: RavenState) -> Dict[str, Any]:
    """Node 5: Reflect on evidence quality, credibility, and sufficiency."""
    logger.info("Executing Graph Node: [reflection]")

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

    claim = state.get("claim") or ""
    domain_state = state.get("domain") or {}
    evidence_list = state.get("fused_evidence") or []
    reflection_result = state.get("reflection_result")
    persona_insights = state.get("persona_insights") or []
    verification_result = state.get("verification_result")
    attempt_count = state.get("attempt_count", 1)
    maximum_attempts_reached = state.get("maximum_attempts_reached", False)

    if isinstance(domain_state, dict):
        domain = domain_state.get("domain") or "General"
    else:
        domain = str(domain_state) if domain_state else "General"

    if not claim.strip():
        return {
            "supervisor_result": {},
            "errors": state.get("errors", []) + [
                "supervisor_error: no claim available"
            ],
        }

    try:
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

        # Normalize Enum / string representations such as:
        # "CONFLICTING_EVIDENCE" or "VerificationAssessment.CONFLICTING_EVIDENCE"
        assessment_name = str(
            getattr(verification_assessment, "value", verification_assessment)
        ).split(".")[-1]

        if assessment_name == "CONFLICTING_EVIDENCE":
            overall_confidence = 0.0

            if isinstance(verification_result, dict):
                raw_confidence = verification_result.get(
                    "overall_confidence",
                    0.0,
                )
            else:
                raw_confidence = getattr(
                    verification_result,
                    "overall_confidence",
                    0.0,
                )

            try:
                overall_confidence = float(raw_confidence)
            except (TypeError, ValueError):
                overall_confidence = 0.0

            conflict_evidence_ids = []

            verification_results = (
                verification_result.get("results", [])
                if isinstance(verification_result, dict)
                else getattr(verification_result, "results", [])
            )

            for verification in verification_results:
                relationship = getattr(
                    verification,
                    "relationship",
                    None,
                )

                if relationship is None and isinstance(verification, dict):
                    relationship = verification.get("relationship")

                relationship_name = str(
                    getattr(relationship, "value", relationship)
                ).split(".")[-1]

                if relationship_name == "CONTRADICTS":
                    evidence_id = getattr(
                        verification,
                        "evidence_id",
                        None,
                    )

                    if evidence_id is None and isinstance(verification, dict):
                        evidence_id = verification.get("evidence_id")

                    if evidence_id:
                        conflict_evidence_ids.append(str(evidence_id))

            conflict_reason = (
                "Verification detected conflicting evidence. "
                "The evidence set contains contradictory relationships, "
                "so the system will not issue a definitive Real/Fake verdict "
                "until the conflict is explicitly resolved."
            )

            logger.warning(
                "Supervisor safety gate triggered: "
                "CONFLICTING_EVIDENCE -> Unverified"
            )

            return {
                "supervisor_result": {
                    "final_verdict": "Unverified",
                    "confidence_score": round(
                        max(0.0, min(1.0, overall_confidence)) * 100
                    ),
                    "trust_score": round(
                        max(0.0, min(1.0, overall_confidence)) * 100
                    ),
                    "reasoning_summary": conflict_reason,
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

        logger.info(
            "XAI completed with verdict: %s",
            final_output.get("final_verdict"),
        )

        return {
            "final_output": final_output,
        }

    except Exception as e:
        logger.exception(f"Error in xai_node: {e}")

        return {
            "final_output": {},
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

    reflection = state.get("reflection_result") or {}
    verification = state.get("verification_result")
    verification_failure = state.get("verification_failure")

    # Verification failure means we should try retrieving evidence again.
    if verification_failure is not None:
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
    workflow.add_node("input_processing", input_processing_node)
    workflow.add_node("claim_extraction", claim_extraction_node)
    workflow.add_node("domain_detection", domain_detection_node)
    workflow.add_node("evidence_verification", evidence_verification_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("persona_analysis", persona_analysis_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("xai", xai_node)

    # Add Linear Edges
    workflow.add_edge(START, "input_processing")
    workflow.add_edge("input_processing", "claim_extraction")
    workflow.add_edge("claim_extraction", "domain_detection")
    workflow.add_edge("domain_detection", "evidence_verification")
    workflow.add_edge("evidence_verification", "reflection")

    workflow.add_node("retry_evidence", retry_evidence_node)

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
    initial_state: RavenState = {
        "text": text,
        "image_path": image_path,
        "errors": [],
        "attempt_count": 1,
        "maximum_attempts_reached": False,
    }
    final_state = graph.invoke(initial_state)
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
