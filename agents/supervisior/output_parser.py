"""Output Parser for Supervisor Agent.
Validates and parses the LLM's raw text response into a strict schema,
with deterministic validation and anti-hallucination safeguards for key_evidence_used.
"""

import json
import re
from typing import Literal, List, Any, Optional, Dict
from pydantic import BaseModel, Field, ValidationError


class SupervisorOutput(BaseModel):
    final_verdict: Literal["Real", "Fake", "Unverified"]
    confidence_score: int = Field(ge=0, le=100)
    trust_score: int = Field(ge=0, le=100)
    reasoning_summary: str
    key_evidence_used: List[str]


def _extract_evidence_dict(evidence: Any) -> Dict[str, Any]:
    if hasattr(evidence, "model_dump"):
        return evidence.model_dump()
    elif hasattr(evidence, "dict"):
        return evidence.dict()
    elif isinstance(evidence, dict):
        return dict(evidence)
    return {"text": str(evidence)}


def _normalize_score(val: Any, default: int = 50) -> int:
    try:
        f = float(val)
        if 0.0 <= f <= 1.0 and f > 0.0 and isinstance(val, float):
            return int(round(f * 100))
        return max(0, min(100, int(round(f))))
    except (ValueError, TypeError):
        return default


def _validate_and_fix_key_evidence(
    key_evidence_used: List[str],
    evidence_list: Optional[List[Any]]
) -> List[str]:
    """Ensure key_evidence_used references ONLY actual evidence from evidence_list."""
    if not evidence_list:
        return []

    normalized_ev = [_extract_evidence_dict(ev) for ev in evidence_list]
    valid_ids = set()
    valid_sources = set()

    for idx, ev in enumerate(normalized_ev, 1):
        if ev.get("evidence_id"):
            valid_ids.add(str(ev["evidence_id"]).lower())
        if ev.get("chunk_id"):
            valid_ids.add(str(ev["chunk_id"]).lower())
        if ev.get("document_id"):
            valid_ids.add(str(ev["document_id"]).lower())
        valid_ids.add(f"evidence_{idx}")
        if ev.get("source"):
            valid_sources.add(str(ev["source"]).strip().lower())

    sanitized = []
    generic_placeholders = {"item 1", "item 2", "source 1", "source 2", "evidence item 1", "evidence item 2"}

    for item in key_evidence_used:
        item_str = str(item).strip()
        if not item_str or item_str.lower() in generic_placeholders:
            continue

        item_lower = item_str.lower()
        # Check if item mentions any valid ID or valid source
        has_id = any(vid in item_lower for vid in valid_ids)
        has_source = any(vsrc in item_lower for vsrc in valid_sources if len(vsrc) > 2)

        if has_id or has_source:
            sanitized.append(item_str)

    # If all items were invalid/hallucinated or empty, populate with actual evidence
    if not sanitized:
        for idx, ev in enumerate(normalized_ev, 1):
            source = ev.get("source", "Unknown Source")
            ev_id = ev.get("evidence_id") or ev.get("chunk_id", f"evidence_{idx}")
            sanitized.append(f"{source} ({ev_id})")

    return sanitized


def _clean_reasoning(text: str, evidence_count: int) -> str:
    """Strip plural/hallucinatory claims if only one evidence item exists."""
    if evidence_count <= 1:
        text = re.sub(r"\bmultiple authoritative sources\b", "an authoritative source", text, flags=re.IGNORECASE)
        text = re.sub(r"\bmultiple sources\b", "the single provided source", text, flags=re.IGNORECASE)
        text = re.sub(r"\bvarious sources\b", "the provided source", text, flags=re.IGNORECASE)
        text = re.sub(r"\bsources confirm\b", "the source confirms", text, flags=re.IGNORECASE)
        text = re.sub(r"\bacross sources\b", "in the source", text, flags=re.IGNORECASE)
    return text


def parse_supervisor_output(
    raw_response: str,
    evidence_list: Optional[List[Any]] = None
) -> SupervisorOutput:
    """
    Parse raw_response into a SupervisorOutput, with robust normalization and anti-hallucination.
    """
    cleaned_response = raw_response.strip()

    # Strip markdown code fences if present
    cleaned_response = re.sub(r"^```json\s*", "", cleaned_response, flags=re.IGNORECASE)
    cleaned_response = re.sub(r"^```\s*", "", cleaned_response)
    cleaned_response = re.sub(r"```\s*$", "", cleaned_response)
    cleaned_response = cleaned_response.strip()

    start = cleaned_response.find("{")
    end = cleaned_response.rfind("}")
    if start != -1 and end != -1:
        json_str = cleaned_response[start:end + 1]
    else:
        json_str = cleaned_response

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON. Raw response: {raw_response}\nError: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Parsed JSON is not a dictionary. Raw response: {raw_response}")

    # Normalize verdict
    raw_verdict = str(data.get("final_verdict", "Unverified")).strip()
    verdict_map = {
        "real": "Real",
        "fake": "Fake",
        "unverified": "Unverified"
    }
    final_verdict = verdict_map.get(raw_verdict.lower(), "Unverified")
    data["final_verdict"] = final_verdict

    # Normalize scores
    data["confidence_score"] = _normalize_score(data.get("confidence_score"), default=50)
    data["trust_score"] = _normalize_score(data.get("trust_score"), default=50)

    # Sanitize and validate key_evidence_used
    raw_key_evidence = data.get("key_evidence_used", [])
    if isinstance(raw_key_evidence, str):
        raw_key_evidence = [raw_key_evidence]
    elif not isinstance(raw_key_evidence, list):
        raw_key_evidence = []
    data["key_evidence_used"] = _validate_and_fix_key_evidence(raw_key_evidence, evidence_list)

    # Clean reasoning summary
    evidence_count = len(evidence_list) if evidence_list else 0
    raw_summary = str(data.get("reasoning_summary", "Evaluation completed."))
    data["reasoning_summary"] = _clean_reasoning(raw_summary, evidence_count)

    try:
        if hasattr(SupervisorOutput, 'model_validate'):
            return SupervisorOutput.model_validate(data)
        else:
            return SupervisorOutput(**data)
    except ValidationError as e:
        raise ValueError(f"Failed to validate schema. Data: {data}\nError: {e}") from e
