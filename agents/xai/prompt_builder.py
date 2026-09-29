import json
from typing import Any, Dict, List, Optional


def evidence_to_dict(evidence: Any) -> Dict[str, Any]:
    if hasattr(evidence, "model_dump"):
        return evidence.model_dump()
    if hasattr(evidence, "dict"):
        return evidence.dict()
    if isinstance(evidence, dict):
        return dict(evidence)
    return {"text": str(evidence)}


def build_xai_prompt(
    claim_text: str,
    domain: str,
    evidence_list: List[Any],
    reflection_result: Dict[str, Any],
    supervisor_result: Dict[str, Any],
    attempt_count: int = 1,
    maximum_attempts_reached: bool = False
) -> str:
    evidence_data = [
        evidence_to_dict(evidence)
        for evidence in evidence_list
    ]
    evidence_count = len(evidence_data)

    # Derive baseline grounding confidence from supervisor confidence
    sup_conf_raw = supervisor_result.get("confidence_score", 85)
    try:
        sup_conf_float = float(sup_conf_raw)
        baseline_conf = round(sup_conf_float / 100.0 if sup_conf_float > 1.0 else sup_conf_float, 2)
    except (ValueError, TypeError):
        baseline_conf = 0.85

    supervisor_verdict = supervisor_result.get("final_verdict", "Unverified")

    prompt = f"""You are the Explainable AI (XAI) Agent of a multi-agent fake-news verification system.
Your job is ONLY to explain the final decision made by the Supervisor Agent.

CRITICAL ARCHITECTURAL RULES:
1. DO NOT change the Supervisor's final verdict. It must remain strictly "{supervisor_verdict}".
2. DO NOT make a new verdict.
3. Explain why the Supervisor reached the verdict "{supervisor_verdict}" based on the evidence, Reflection metrics, and attempt history.
4. Mention source credibility: distinguish semantic similarity from source authority.
   - If the source is an unknown blog or has very low authority (e.g., authority=0.10), explicitly highlight that the source is weak/unreliable.
   - Do NOT call an unknown blog authoritative or reliable.
5. STRICT ANTI-HALLUCINATION RULES:
   - Total evidence items provided: {evidence_count}.
   - If only 1 evidence item is provided ({evidence_count} <= 1), do NOT say "multiple sources confirm" or "sources agree". Refer to "the single provided source".
   - Do NOT invent sources, URLs, or evidence.
   - Do NOT claim evidence comes from the Reserve Bank of India (RBI) unless RBI is actually listed in the EVIDENCE below!
6. GROUNDED EXPLANATION CONFIDENCE:
   - Return a valid confidence float between 0.0 and 1.0 representing the grounding of the explanation.
   - Ground it in the Supervisor's confidence (Supervisor confidence: {sup_conf_raw}, expected ~{baseline_conf}).
   - Do NOT return 0.0 unless the grounding is completely broken.

CLAIM:
"{claim_text}"

DOMAIN:
{domain}

RETRIEVAL METRICS:
- Attempt Count: {attempt_count}
- Maximum Attempts Reached: {maximum_attempts_reached}

EVIDENCE:
{json.dumps(evidence_data, default=str, indent=2)}

REFLECTION RESULT:
{json.dumps(reflection_result, default=str, indent=2)}

SUPERVISOR RESULT:
{json.dumps(supervisor_result, default=str, indent=2)}

Return ONLY valid JSON with this exact schema:
{{
    "final_verdict": "{supervisor_verdict}",
    "explanation": "<Clear, plain-language explanation of the Supervisor's verdict for a general reader>",
    "evidence_summary": [
        "<Key finding 1 from provided evidence>",
        "<Key finding 2 from provided evidence>"
    ],
    "source_analysis": "<Analysis of source credibility and authority vs similarity>",
    "reasoning": "<Step-by-step reasoning connecting evidence, Reflection evaluation, and the Supervisor verdict>",
    "confidence": {baseline_conf}
}}
"""
    return prompt