import json
import requests
from typing import Any

from ..retrieval.evidence_schema import Evidence
from .verification_schema import VerificationResult


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama service cannot be reached or returns an error."""


DEFAULT_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:3b"
MAX_EVIDENCE_CHARS = 4000


def _build_prompt(claim: str, evidence: Evidence) -> str:
    """Construct the prompt sent to the LLM.

    The prompt follows the specification: include CLAIM and EVIDENCE sections.
    """
    evidence_text = evidence.text.strip()
    if len(evidence_text) > MAX_EVIDENCE_CHARS:
        evidence_text = evidence_text[:MAX_EVIDENCE_CHARS].rstrip() + "..."

    prompt = (
        f"CLAIM:\n{claim}\n\n"
        f"EVIDENCE:\nEvidence ID: {evidence.evidence_id}\n"
        f"Text: {evidence_text}\n"
        "\nEvaluate the relationship between the claim and the evidence. "
        "Return exactly one valid JSON object and no markdown, prose, code fences, or extra keys. "
        "The JSON object MUST contain all three fields: relationship, confidence, reasoning. "
        "Every field is mandatory; never omit a field or return null. "
        "relationship MUST be exactly one of SUPPORTS, CONTRADICTS, INSUFFICIENT. "
        "confidence MUST be a JSON number from 0.0 to 1.0. "
        "reasoning MUST be a non-empty JSON string explaining the classification using only the evidence. "
        "Use this exact shape: "
        '{"relationship":"SUPPORTS|CONTRADICTS|INSUFFICIENT",'
        '"confidence":0.0,"reasoning":"brief evidence-based explanation"}. '
        "Do NOT invent any information."
    )
    return prompt


def verify_evidence(
    claim: str,
    evidence: Evidence,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    timeout: int = 300,
) -> VerificationResult:
    """Send a single evidence item to Ollama and parse the response.

    Args:
        claim: The claim string.
        evidence: An Evidence object.
        endpoint: Ollama generate endpoint.
        model: Model name (default qwen2.5:3b).
        timeout: Request timeout seconds.

    Returns:
        VerificationResult instance.

    Raises:
        OllamaUnavailableError: If Ollama cannot be contacted or returns non‑200.
        ValueError: If the response JSON is malformed or fails validation.
    """
    prompt = _build_prompt(claim, evidence)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # Instruct the model to output JSON; some models respect "format": "json"
        "format": "json",
    }
    try:
        response = requests.post(endpoint, json=payload, timeout=timeout)
    except Exception as exc:
        raise OllamaUnavailableError(f"Failed to connect to Ollama at {endpoint}: {exc}") from exc

    if response.status_code != 200:
        raise OllamaUnavailableError(
            f"Ollama returned status {response.status_code}: {response.text}"
        )

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ollama response is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Ollama response JSON must be an object")

    # Ollama's generate endpoint returns a dict with a "response" field containing the model output.
    # We expect the model to output raw JSON, possibly with surrounding whitespace.
    raw_output: Any = data.get("response")
    if raw_output is None:
        raise ValueError("Ollama response missing 'response' field")
    if not isinstance(raw_output, str):
        raise ValueError("Ollama response field must contain a JSON string")

    # Attempt to parse the raw output as JSON.
    try:
        result_json = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model output is not valid JSON: {exc}\nOutput was: {raw_output}") from exc

    if not isinstance(result_json, dict):
        raise ValueError("Model output JSON must be an object")

    required_fields = ("relationship", "confidence", "reasoning")
    missing_fields = [field for field in required_fields if field not in result_json]
    if missing_fields:
        raise ValueError(
            "Model output missing required field(s): " + ", ".join(missing_fields)
        )

    # Build the VerificationResult, adding additional required fields.
    verification = VerificationResult(
        evidence_id=evidence.evidence_id,
        claim=claim,
        relationship=result_json["relationship"],
        confidence=result_json["confidence"],
        reasoning=result_json["reasoning"],
        verifier=model,
    )
    return verification
