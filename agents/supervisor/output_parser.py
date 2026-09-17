"""
Output Parser for Supervisor Agent.
Validates and parses the LLM's raw text response into a strict schema.
"""

import json
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, List

class SupervisorOutput(BaseModel):
    final_verdict: Literal["Real", "Fake", "Unverified"]
    confidence_score: int = Field(ge=0, le=100)
    trust_score: int = Field(ge=0, le=100)
    reasoning_summary: str
    key_evidence_used: List[str]

def parse_supervisor_output(raw_response: str) -> SupervisorOutput:
    """
    Parse raw_response (the LLM's text output) into a SupervisorOutput.
    
    Steps:
        1. Strip whitespace and any markdown code fences.
        2. Parse the text as JSON.
        3. Validate against SupervisorOutput.
    
    Raises:
        ValueError: If JSON parsing or validation fails, with raw_response included.
    """
    cleaned_response = raw_response.strip()
    
    # Strip markdown code fences if present
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[len("```json"):].strip()
    elif cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[len("```"):].strip()
        
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3].strip()
        
    try:
        data = json.loads(cleaned_response)
        # Using model_validate for Pydantic v2 if dict is available, or unpacking as fallback
        if hasattr(SupervisorOutput, 'model_validate'):
            return SupervisorOutput.model_validate(data)
        else:
            return SupervisorOutput(**data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON. Raw response: {raw_response}\nError: {e}") from e
    except ValidationError as e:
        raise ValueError(f"Failed to validate schema. Raw response: {raw_response}\nError: {e}") from e
