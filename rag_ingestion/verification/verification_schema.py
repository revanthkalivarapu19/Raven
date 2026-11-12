from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, validator

from .claim_scope import ClaimScope


class VerificationResult(BaseModel):
    evidence_id: str
    claim: str
    relationship: Literal["SUPPORTS", "CONTRADICTS", "INSUFFICIENT"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    verifier: str
    same_event: bool = Field(
        default=True,
        description=(
            "For event/news claims, true only when the evidence concerns the same event."
        ),
    )
    claim_aligned: bool = Field(
        default=False,
        description=(
            "True when evidence matches the claim proposition and relevant context. "
            "For event claims this requires valid event alignment; for scientific/general "
            "claims it does not require a literal dated event."
        ),
    )
    directly_establishes_proposition: bool = Field(
        default=False,
        description=(
            "True only when the evidence directly establishes the claim as stated "
            "(SUPPORTS) or explicitly establishes its opposite (CONTRADICTS). "
            "Topical relevance or a related mechanism alone is not sufficient."
        ),
    )
    direct_evidence_excerpt: Optional[str] = Field(
        default=None,
        description="The evidence sentence accepted by the claim–evidence directness gate.",
    )
    verifier_relationship: Optional[Literal["SUPPORTS", "CONTRADICTS", "INSUFFICIENT"]] = None
    constraint_result: Optional[str] = None
    rejection_reason: Optional[str] = None
    verifier_failed: bool = False

    @validator("confidence")
    def confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        return v


class OverallVerification(BaseModel):
    claim: str
    claim_scope: ClaimScope = "GENERAL"
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
