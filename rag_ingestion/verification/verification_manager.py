import logging
from typing import List

from ..retrieval.evidence_schema import Evidence
from .ollama_verifier import verify_evidence, OllamaUnavailableError
from .verification_schema import VerificationResult, OverallVerification

logger = logging.getLogger(__name__)


class EvidenceVerificationManager:
    """Manager orchestrating verification of multiple evidence items.

     The aggregation logic follows deterministic rules:
     * If any SUPPORTS with confidence >= 0.7 and any CONTRADICTS with confidence >= 0.7,
       overall assessment is ``CONFLICTING_EVIDENCE``.
     * Else if any CONTRADICTS with confidence >= 0.7 (and no strong SUPPORTS),
       overall assessment is ``CONTRADICTED``.
     * Else if at least one SUPPORTS with confidence >= 0.7,
       overall assessment is ``SUPPORTED``.
     * Otherwise the assessment is ``INSUFFICIENT_EVIDENCE``.
     Overall confidence is the maximum of the average supporting confidence and the average
     contradictory confidence (0.0 when the category is absent).
    """

    def __init__(self, *, model: str = "qwen2.5:3b", timeout: int = 300) -> None:
        self.model = model
        self.timeout = timeout

    def verify(self, claim: str, evidence_list: List[Evidence]) -> OverallVerification:
        results: List[VerificationResult] = []
        for ev in evidence_list:
            try:
                result = verify_evidence(claim, ev, model=self.model, timeout=self.timeout)
                results.append(result)
            except OllamaUnavailableError as exc:
                logger.error("Ollama unavailable: %s", exc)
                raise
            except Exception as exc:
                logger.error("Verification failed for evidence %s: %s", ev.evidence_id, exc)
                raise

        # Aggregation
        supports = [r for r in results if r.relationship == "SUPPORTS"]
        contradicts = [r for r in results if r.relationship == "CONTRADICTS"]
        insufficient = [r for r in results if r.relationship == "INSUFFICIENT"]

        def avg_confidence(items: List[VerificationResult]) -> float:
            return sum(r.confidence for r in items) / len(items) if items else 0.0

        sup_conf = avg_confidence(supports)
        con_conf = avg_confidence(contradicts)

        # Determine overall assessment
        has_strong_supports = any(
            r.relationship == "SUPPORTS" and r.confidence >= 0.7 for r in results
        )
        has_strong_contradicts = any(
            r.relationship == "CONTRADICTS" and r.confidence >= 0.7 for r in results
        )

        if has_strong_supports and has_strong_contradicts:
            overall_assessment = "CONFLICTING_EVIDENCE"
        elif has_strong_contradicts:
            overall_assessment = "CONTRADICTED"
        elif has_strong_supports:
            overall_assessment = "SUPPORTED"
        else:
            overall_assessment = "INSUFFICIENT_EVIDENCE"

        overall_confidence = max(sup_conf, con_conf)

        overall = OverallVerification(
            claim=claim,
            results=results,
            overall_assessment=overall_assessment,
            overall_confidence=overall_confidence,
        )
        return overall
