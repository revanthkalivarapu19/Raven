"""
supervisor_agent.py
==================
RAVEN Project - Supervisor Agent

This module is intentionally standalone (no LangGraph dependency)
so it can be developed and tested independently. Once the team's
LangGraph graph is assembled, this module's run_supervisor()
function will be wrapped as a single graph node - it already
accepts a plain dict and returns a plain dict, matching the shape
LangGraph nodes expect, so no internal logic should need to change
at integration time.
"""

import logging
from typing import Dict, Any

from .prompt_builder import build_supervisor_prompt
from .llm_client import call_supervisor_llm
from .output_parser import parse_supervisor_output

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Expected supervisor_input shape:
#   {
#       "claim_text": str,              (required)
#       "domain": str,                  (required)
#       "evidence_list": List[Evidence], (required)
#       "reflection_notes": dict | None, (optional)
#       "original_language": str,        (optional)
#       "was_translated": bool           (optional)
#   }
def run_supervisor(supervisor_input: dict, max_retries: int = 2) -> dict:
    """
    Run the full Supervisor pipeline:
        1. Build the prompt using build_supervisor_prompt().
        2. Call the LLM using call_supervisor_llm().
        3. Parse the response using parse_supervisor_output().
        4. If parsing fails, retry up to max_retries times.
        5. Return SupervisorOutput as a dict on success.
    
    Raises:
        ValueError: If required keys are missing from supervisor_input.
        RuntimeError: If all retries fail.
    """
    if "claim_text" not in supervisor_input:
        raise ValueError("Missing required key in supervisor_input: 'claim_text'")
    if "domain" not in supervisor_input:
        raise ValueError("Missing required key in supervisor_input: 'domain'")
    if "evidence_list" not in supervisor_input:
        raise ValueError("Missing required key in supervisor_input: 'evidence_list'")

    base_prompt = build_supervisor_prompt(supervisor_input)
    prompt = base_prompt
    
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries + 1} for Supervisor Agent.")
            raw_response = call_supervisor_llm(prompt)
            output = parse_supervisor_output(raw_response)
            logger.info("Successfully parsed LLM response.")
            if hasattr(output, 'model_dump'):
                return output.model_dump()
            else:
                return output.dict()
        except ValueError as e:
            logger.warning(f"Parse error on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                logger.info("Retrying with stricter prompt...")
                retry_reminder = (
                    "\n\nYour previous response was not valid JSON. "
                    "Respond with ONLY the JSON object, nothing else."
                )
                prompt = base_prompt + retry_reminder
            else:
                logger.error("All retries exhausted.")
                raise RuntimeError(f"Supervisor Agent could not produce a valid verdict after {max_retries + 1} attempts.") from e

if __name__ == "__main__":
    from .mock_data import get_mock_input
    
    print("=== Testing Supervisor Agent ===")
    
    print("\n--- Scenario: Supporting ---")
    try:
        supp_input = get_mock_input("supporting")
        supp_result = run_supervisor(supp_input)
        print("Result (Supporting):", supp_result)
    except Exception as e:
        print(f"Test failed (Supporting): {e}")

    print("\n--- Scenario: Contradicting ---")
    try:
        contra_input = get_mock_input("contradicting")
        contra_result = run_supervisor(contra_input)
        print("Result (Contradicting):", contra_result)
    except Exception as e:
        print(f"Test failed (Contradicting): {e}")

    print("\n--- Scenario: No Reflection ---")
    try:
        no_refl_input = get_mock_input("no_reflection")
        no_refl_result = run_supervisor(no_refl_input)
        print("Result (No Reflection):", no_refl_result)
    except Exception as e:
        print(f"Test failed (No Reflection): {e}")
