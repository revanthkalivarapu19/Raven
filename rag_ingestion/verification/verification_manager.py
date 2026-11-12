import logging
import time
import uuid
from datetime import datetime, timezone
from typing import List

from ..retrieval.evidence_schema import Evidence
from .ollama_verifier import verify_evidence_batch
from .verification_schema import VerificationResult, OverallVerification
from .claim_scope import ClaimScope, classify_claim_scope, verification_can_establish_claim

logger = logging.getLogger(__name__)


class EvidenceVerificationManager:
    """Manager orchestrating verification of multiple evidence items.

     The aggregation logic follows deterministic rules:
     * If any scope-eligible SUPPORTS with confidence >= 0.7 and any
       scope-eligible CONTRADICTS with confidence >= 0.7,
       overall assessment is ``CONFLICTING_EVIDENCE``.
     * Else if any CONTRADICTS with confidence >= 0.7 (and no strong SUPPORTS),
       overall assessment is ``CONTRADICTED``.
     * Else if at least one SUPPORTS with confidence >= 0.7,
       overall assessment is ``SUPPORTED``.
     * Otherwise the assessment is ``INSUFFICIENT_EVIDENCE``.
     Overall confidence is the maximum of the average supporting confidence and the average
     contradictory confidence (0.0 when the category is absent).
    """

    def __init__(self, *, model: str = "qwen2.5:3b", timeout: int = 35) -> None:
        self.model = model
        self.timeout = timeout

    def verify(
        self, claim: str, evidence_list: List[Evidence], *,
        claim_scope: ClaimScope | None = None,
        claim_id: str = "C1",
        verification_pass_id: str | None = None,
        caller: str = "verification_manager.verify",
        retry_number: int = 0,
        reason: str = "final_ranked_bundle",
    ) -> OverallVerification:
        claim_scope = claim_scope or classify_claim_scope(claim)
        started = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        verification_pass_id = verification_pass_id or f"{claim_id}:V:{uuid.uuid4().hex[:8]}"
        evidence_ids = [item.evidence_id for item in evidence_list]
        checked_evidence: list[Evidence] = []
        verifier_calls = 0
        logger.info(
            "verification_call event=start claim_id=%s verification_pass_id=%s pass=1 "
            "caller=%s evidence_count=%d evidence_ids=%s start_time=%s retry_number=%d reason=%s",
            claim_id, verification_pass_id, caller, len(evidence_list), evidence_ids,
            started_at, retry_number, reason,
        )
        try:
            results = []
            # Adjudicate the strongest ranked passage alone. A direct decision
            # ends the work; only insufficient/ambiguous evidence advances to
            # the next ranked item, avoiding noisy mixed bundles.
            for evidence in evidence_list:
                checked_evidence.append(evidence)
                verifier_calls += 1
                item_results = verify_evidence_batch(
                    claim, [evidence], model=self.model, timeout=self.timeout,
                    claim_id=claim_id,
                )
                if item_results:
                    results.extend(item_results)
                    decision = item_results[0]
                    if decision.verifier_failed or verification_can_establish_claim(decision, claim_scope):
                        break
        except Exception as exc:
            finished = time.perf_counter()
            logger.warning(
                "verification_call event=end claim_id=%s verification_pass_id=%s caller=%s "
                "evidence_count=%d evidence_ids=%s start_time=%s end_time=%s duration_seconds=%.4f "
                "retry_number=%d reason=%s outcome=failed_closed error_type=%s error=%s",
                claim_id, verification_pass_id, caller, len(evidence_list), evidence_ids,
                started_at, datetime.now(timezone.utc).isoformat(), finished - started,
                retry_number, reason, type(exc).__name__, str(exc),
            )
            results = [VerificationResult(
                evidence_id=item.evidence_id, claim=claim,
                relationship="INSUFFICIENT", confidence=0.0,
                reasoning=f"Verifier call failed: {type(exc).__name__}.",
                verifier=self.model, same_event=False, claim_aligned=False,
                directly_establishes_proposition=False,
                verifier_relationship="INSUFFICIENT", constraint_result="NOT_RUN",
                rejection_reason=f"{type(exc).__name__}: {exc}", verifier_failed=True,
            ) for item in checked_evidence]
        # Keep the verifier boundary total: callers receive one result per
        # submitted item even if a substituted backend violates the contract.
        by_id = {}
        duplicate_ids = set()
        expected_ids = {item.evidence_id for item in checked_evidence}
        for result in results:
            if result.evidence_id not in expected_ids:
                continue
            if result.evidence_id in by_id:
                duplicate_ids.add(result.evidence_id)
            else:
                by_id[result.evidence_id] = result
        normalized_results = []
        for evidence in checked_evidence:
            result = by_id.get(evidence.evidence_id)
            if result is None or evidence.evidence_id in duplicate_ids:
                reason = "verifier omitted or duplicated this evidence ID"
                result = VerificationResult(
                    evidence_id=evidence.evidence_id, claim=claim,
                    relationship="INSUFFICIENT", confidence=0.0,
                    reasoning=reason, verifier=self.model, same_event=False,
                    claim_aligned=False, directly_establishes_proposition=False,
                    verifier_relationship="INSUFFICIENT", constraint_result="NOT_RUN",
                    rejection_reason=reason, verifier_failed=True,
                )
            normalized_results.append(result)
        results = normalized_results
        verification_failed = any(item.verifier_failed for item in results)
        finished = time.perf_counter()
        logger.info(
            "verification_call event=end claim_id=%s verification_pass_id=%s caller=%s "
            "candidate_count=%d evidence_count=%d evidence_ids=%s start_time=%s end_time=%s duration_seconds=%.4f "
            "retry_number=%d reason=%s outcome=%s ollama_calls=%d",
            claim_id, verification_pass_id, caller, len(evidence_list), len(checked_evidence),
            [item.evidence_id for item in checked_evidence],
            started_at, datetime.now(timezone.utc).isoformat(), finished - started,
            retry_number, reason, "failed_closed" if verification_failed else "completed",
            verifier_calls,
        )
        logger.info(
            "performance_diag phase=verification_aggregation elapsed_seconds=%.4f "
            "evidence_count=%d",
            time.perf_counter() - started, len(results),
        )

        # Aggregation
        supports = [
            r for r in results
            if r.relationship == "SUPPORTS"
            and verification_can_establish_claim(r, claim_scope)
        ]
        contradicts = [
            r for r in results
            if r.relationship == "CONTRADICTS"
            and verification_can_establish_claim(r, claim_scope)
        ]
        def avg_confidence(items: List[VerificationResult]) -> float:
            return sum(r.confidence for r in items) / len(items) if items else 0.0

        sup_conf = avg_confidence(supports)
        con_conf = avg_confidence(contradicts)

        # Determine overall assessment
        has_strong_supports = any(
            r.confidence >= 0.7 for r in supports
        )
        has_strong_contradicts = any(
            r.confidence >= 0.7 for r in contradicts
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
            claim_scope=claim_scope,
            results=results,
            overall_assessment=overall_assessment,
            overall_confidence=overall_confidence,
        )
        logger.info(
            "verification_summary claim=%r evidence_count=%d accepted_count=%d "
            "final_relationship=%s overall_confidence=%.2f",
            claim, len(results), len(supports) + len(contradicts),
            overall_assessment, overall_confidence,
        )
        return overall
