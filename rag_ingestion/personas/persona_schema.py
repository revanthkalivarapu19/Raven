# src/personas/persona_schema.py
"""Schema for Persona insights.
Defines the output contract for all persona agents.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, validator


class PersonaInsight(BaseModel):
    """Result produced by a persona analysis.

    Fields:
        persona: Name of the persona (e.g., "journalist").
        assessment: Normalized assessment – one of SUPPORTED, CONTRADICTED,
                    INSUFFICIENT, CONFLICTING.
        confidence: Float in [0.0, 1.0] indicating certainty.
        reasoning: Human‑readable explanation (placeholder for now).
        evidence_ids: List of evidence identifiers used in the analysis.
    """

    persona: str
    assessment: Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT", "CONFLICTING"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    evidence_ids: List[str]

    @validator("evidence_ids")
    def non_empty_ids(cls, v: List[str]):
        if not v:
            raise ValueError("evidence_ids must contain at least one id")
        return v
