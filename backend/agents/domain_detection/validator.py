"""
validator.py

Validates the response returned by the Domain Detection Agent.
"""

from pydantic import BaseModel, Field, ValidationError
from typing import Literal


class DomainResponse(BaseModel):
    """
    Expected output from the Domain Detection Agent.
    """

    domain: Literal[
        "Medical",
        "Politics",
        "Finance",
        "Technology",
        "Unknown"
    ]

    confidence: float = Field(ge=0.0, le=1.0)

    reason: str


def validate_response(response: dict) -> bool:
    """
    Validate the LLM response using Pydantic.
    """

    try:
        DomainResponse(**response)
        return True

    except ValidationError as e:
        print("\nValidation Error:")
        print(e)
        return False