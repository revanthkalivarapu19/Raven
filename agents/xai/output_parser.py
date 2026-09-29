import json
import re
from typing import Any, Dict, List, Optional


def _clean_hallucinations(text: str, evidence_count: int) -> str:
    if evidence_count <= 1:
        text = re.sub(r"\bmultiple authoritative sources\b", "an authoritative source", text, flags=re.IGNORECASE)
        text = re.sub(r"\bmultiple sources\b", "the single provided source", text, flags=re.IGNORECASE)
        text = re.sub(r"\bvarious sources\b", "the provided source", text, flags=re.IGNORECASE)
        text = re.sub(r"\bsources confirm\b", "the source confirms", text, flags=re.IGNORECASE)
        text = re.sub(r"\bacross sources\b", "in the source", text, flags=re.IGNORECASE)
    return text


def parse_xai_output(
    raw_output: str,
    supervisor_result: Optional[Dict[str, Any]] = None,
    evidence_list: Optional[List[Any]] = None
) -> Dict[str, Any]:
    raw_output = raw_output.strip()

    # Remove markdown code fences if Llama adds them
    raw_output = re.sub(r"^```json\s*", "", raw_output, flags=re.IGNORECASE)
    raw_output = re.sub(r"^```\s*", "", raw_output)
    raw_output = re.sub(r"```\s*$", "", raw_output)
    raw_output = raw_output.strip()

    try:
        result = json.loads(raw_output)
    except json.JSONDecodeError:
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("XAI Agent did not return valid JSON.")
        result = json.loads(raw_output[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError("XAI output is not a JSON object.")

    # 1. Enforce final_verdict matches Supervisor strictly
    if supervisor_result and "final_verdict" in supervisor_result:
        expected_verdict = str(supervisor_result["final_verdict"]).strip()
        result["final_verdict"] = expected_verdict
    else:
        raw_v = str(result.get("final_verdict", "Unverified")).strip()
        v_map = {"real": "Real", "fake": "Fake", "unverified": "Unverified"}
        result["final_verdict"] = v_map.get(raw_v.lower(), "Unverified")

    # 2. Fix XAI Confidence Bug:
    # - Must be a valid float between 0.0 and 1.0
    # - If returned as > 1.0 (e.g. 95), normalize to 0.95
    # - If returned as 0.0 or missing while supervisor has valid confidence, derive from supervisor
    raw_conf = result.get("confidence")
    parsed_conf = None
    try:
        if raw_conf is not None:
            f = float(raw_conf)
            if f > 1.0 and f <= 100.0:
                parsed_conf = round(f / 100.0, 2)
            elif 0.0 <= f <= 1.0:
                parsed_conf = round(f, 2)
    except (ValueError, TypeError):
        parsed_conf = None

    # Grounding check: if parsed_conf is 0.0 or None, derive from supervisor if available
    sup_conf = None
    if supervisor_result and "confidence_score" in supervisor_result:
        try:
            sc = float(supervisor_result["confidence_score"])
            sup_conf = round(sc / 100.0 if sc > 1.0 else sc, 2)
        except (ValueError, TypeError):
            sup_conf = None

    if (parsed_conf is None or parsed_conf == 0.0) and sup_conf is not None and sup_conf > 0.0:
        parsed_conf = sup_conf
    elif parsed_conf is None:
        parsed_conf = 0.85

    result["confidence"] = max(0.0, min(1.0, parsed_conf))

    # 3. Clean hallucinations
    evidence_count = len(evidence_list) if evidence_list else 0
    if "explanation" in result:
        result["explanation"] = _clean_hallucinations(str(result["explanation"]), evidence_count)
    if "source_analysis" in result:
        result["source_analysis"] = _clean_hallucinations(str(result["source_analysis"]), evidence_count)
    if "reasoning" in result:
        result["reasoning"] = _clean_hallucinations(str(result["reasoning"]), evidence_count)

    # 4. Validate evidence_summary is a list
    if not isinstance(result.get("evidence_summary"), list):
        summary_val = result.get("evidence_summary")
        result["evidence_summary"] = [str(summary_val)] if summary_val else ["Summary provided."]

    required_fields = [
        "final_verdict",
        "explanation",
        "evidence_summary",
        "source_analysis",
        "reasoning",
        "confidence"
    ]

    for field in required_fields:
        if field not in result:
            result[field] = "N/A"

    return result