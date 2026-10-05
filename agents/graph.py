"""
graph.py
========
RAVEN Project - Active LangGraph Workflow Orchestration

Strict active pipeline boundary:
Input Processing -> Claim Extraction -> Domain Detection -> STOP

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

logger = logging.getLogger(__name__)

# Singletons for downstream agents to avoid re-instantiation overhead
_CLAIM_AGENT = None
_DOMAIN_AGENT = None


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


# -----------------------------------------------------------------------------
# Graph Builder
# -----------------------------------------------------------------------------

def build_raven_graph():
    """
    Assembles and compiles the active LangGraph workflow:
    Input Processing -> Claim Extraction -> Domain Detection -> END
    """
    workflow = StateGraph(RavenState)

    # Add Nodes
    workflow.add_node("input_processing", input_processing_node)
    workflow.add_node("claim_extraction", claim_extraction_node)
    workflow.add_node("domain_detection", domain_detection_node)

    # Add Linear Edges
    workflow.add_edge(START, "input_processing")
    workflow.add_edge("input_processing", "claim_extraction")
    workflow.add_edge("claim_extraction", "domain_detection")
    workflow.add_edge("domain_detection", END)

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
        "errors": []
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
    print("  RAVEN LangGraph Pipeline: Input Processing -> Claim -> Domain")
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
