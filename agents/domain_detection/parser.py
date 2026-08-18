"""
parser.py

Parses and cleans the response returned by the LLM.
"""

import json
import re
from loguru import logger


def clean_response(response: str) -> str:
    """
    Extracts JSON from the LLM response.
    """
    response = response.strip()

    # JSON inside ```json ... ```
    match = re.search(
        r"```json\s*(.*?)\s*```",
        response,
        re.DOTALL
    )

    if match:
        return match.group(1).strip()

    # JSON inside ``` ... ```
    match = re.search(
        r"```\s*(.*?)\s*```",
        response,
        re.DOTALL
    )

    if match:
        return match.group(1).strip()

    # JSON object with extra text before/after it
    start = response.find("{")
    end = response.rfind("}")

    if start != -1 and end != -1 and start < end:
        return response[start:end + 1].strip()

    return response


def parse_response(response: str) -> dict:
    """
    Converts the LLM response into a Python dictionary.
    """

    try:
        cleaned_response = clean_response(response)

        parsed = json.loads(cleaned_response)

        logger.success("Response parsed successfully.")

        return parsed

    except json.JSONDecodeError as e:

        logger.error(f"JSON Parsing Error: {e}")

        return {
            "domain": "Unknown",
            "confidence": 0.0,
            "reason": "Invalid JSON returned by LLM."
        }