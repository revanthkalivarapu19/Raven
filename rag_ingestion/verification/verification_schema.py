from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, validator


class VerificationResult(BaseModel):
    evidence_id: str
    claim: str
    relationship: Literal["SUPPORTS", "CONTRADICTS", "INSUFFICIENT"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    verifier: str

    @validator("confidence")
    def confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        return v


class OverallVerification(BaseModel):
    claim: str
    results: List[VerificationResult]
    overall_assessment: Literal["SUPPORTED", "CONTRADICTED", "CONFLICTING_EVIDENCE", "INSUFFICIENT_EVIDENCE"]
    overall_confidence: float = Field(..., ge=0.0, le=1.0)

    @validator("overall_confidence")
    def overall_confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("overall_confidence must be between 0.0 and 1.0")
        return v


class VerificationFailure(BaseModel):
    """Explicit failure state; it contains no fabricated verdict."""

    status: Literal["VERIFICATION_FAILURE"] = "VERIFICATION_FAILURE"
    claim: str
    evidence_count: int = Field(..., ge=0)
    error_type: str
    message: str

    @classmethod
    def from_exception(cls, claim: str, evidence_count: int, exc: Exception) -> "VerificationFailure":
        return cls(
            claim=claim,
            evidence_count=evidence_count,
            error_type=type(exc).__name__,
            message=str(exc),
        )
