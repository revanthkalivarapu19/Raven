# src/supervisor/supervisor_agent.py
"""Supervisor agent implementation.
Provides a deterministic method to build a SupervisorContext from evidence and verification results.
No external calls, purely deterministic validation.
"""

from __future__ import annotations

from typing import List

from rag_ingestion.supervisor.supervisor_schema import SupervisorContext
from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification.verification_schema import VerificationResult


class SupervisorAgent:
    """Static agent for constructing the SupervisorContext.

    The method validates input types, enforces unique IDs, and builds the mapping.
    It does **not** create placeholder VerificationResult objects for missing evidence.
    """

    @staticmethod
    def build_context(
        claim: str,
        domain: str,
        evidence: List[Evidence],
        verification_results: List[VerificationResult],
    ) -> SupervisorContext:
        """Build and return a validated SupervisorContext.

        Parameters
        ----------
        claim: str
            Original claim text.
        domain: str
            Domain of the claim.
        evidence: List[Evidence]
            Retrieved evidence list.
        verification_results: List[VerificationResult]
            Verification results (may be a subset of evidence).
        """
        # Type checks (basic)
        if not isinstance(claim, str):
            raise TypeError("claim must be a string")
        if not isinstance(domain, str):
            raise TypeError("domain must be a string")
        if not isinstance(evidence, list):
            raise TypeError("evidence must be a list of Evidence objects")
        if not isinstance(verification_results, list):
            raise TypeError("verification_results must be a list of VerificationResult objects")

        # Let the SupervisorContext model perform validation and mapping construction.
        context = SupervisorContext(
            claim=claim,
            domain=domain,
            evidence=evidence,
            verification_results=verification_results,
        )
        return context
