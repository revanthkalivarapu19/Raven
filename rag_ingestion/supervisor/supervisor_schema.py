# src/supervisor/supervisor_schema.py
"""Supervisor context schema.
Aggregates the claim, domain, retrieved evidence, and verification results.
Provides a deterministic mapping from evidence_id to its VerificationResult.
"""

from __future__ import annotations

from typing import List, Dict

from pydantic import BaseModel, validator, Field, model_validator

from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification.verification_schema import VerificationResult


class SupervisorContext(BaseModel):
    """Container for the full reasoning context.

    - ``claim`` and ``domain`` are preserved for downstream persona agents.
    - ``evidence`` is the full list of retrieved Evidence objects.
    - ``verification_results`` is the list of VerificationResult objects (may be a subset).
    - ``mapping`` provides deterministic lookup of a verification result by ``evidence_id``.
    """

    claim: str
    domain: str
    evidence: List[Evidence]
    verification_results: List[VerificationResult]
    mapping: Dict[str, VerificationResult] = Field(default_factory=dict)

    @validator("evidence")
    def unique_evidence_ids(cls, v: List[Evidence]):
        ids = [e.evidence_id for e in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate evidence_id found in evidence list")
        return v

    @validator("verification_results")
    def verification_checks(
        cls,
        v: List[VerificationResult],
        values,
    ):
        # Ensure no duplicate evidence_id in verification results
        ids = [vr.evidence_id for vr in v]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate evidence_id found in verification_results"
            )
        # Ensure each verification result references an existing evidence_id
        evidence_list: List[Evidence] = values.get("evidence", [])
        evidence_ids = {e.evidence_id for e in evidence_list}
        for eid in ids:
            if eid not in evidence_ids:
                raise ValueError(
                    f"VerificationResult references unknown evidence_id: {eid}"
                )
        return v

    @model_validator(mode="after")
    def build_mapping(self) -> "SupervisorContext":
        # Build deterministic mapping from evidence_id to VerificationResult
        mapping = {vr.evidence_id: vr for vr in self.verification_results}
        object.__setattr__(self, "mapping", mapping)
        return self
