"""
validator.py

Validates the response returned by the Domain Detection Agent.
"""

from pydantic import BaseModel, Field, ValidationError
from typing import Literal

SUPPORTED_DOMAINS = ("Medical", "Politics", "Finance", "Science")


class DomainResponse(BaseModel):
    """
    Expected output from the Domain Detection Agent.
    """

    domain: Literal[
        "Medical",
        "Politics",
        "Finance",
        "Science",
        "Unknown"
    ]

    confidence: float = Field(ge=0.0, le=1.0)

    reason: str


def validate_response(response: object) -> bool:
    """
    Validate the LLM response using Pydantic.
    """

    if not isinstance(response, dict):
        return False

    try:
        validated = DomainResponse(**response)
        return bool(validated.reason.strip())

    except (ValidationError, TypeError) as e:
        print("\nValidation Error:")
        print(e)
        return False