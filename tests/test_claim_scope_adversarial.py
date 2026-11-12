"""Characterize automatic claim scope and explicit scope propagation."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agents import graph
from rag_ingestion.pipeline.claim_processing import ClaimProcessingPipeline
from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification import verification_manager
from rag_ingestion.verification.claim_scope import classify_claim_scope
from rag_ingestion.verification.verification_schema import OverallVerification, VerificationResult


EVENT_CLAIMS = [
    ("The Reserve Bank of India reduced the repo rate by 0.50%.", "EVENT_SPECIFIC"),
    ("The government approved the bill on Tuesday.", "EVENT_SPECIFIC"),
    ("Company X acquired Company Y in March.", "EVENT_SPECIFIC"),
    ("The regulator fined the company $10 million yesterday.", "EVENT_SPECIFIC"),
    ("The court rejected the appeal in 2026.", "EVENT_SPECIFIC"),
]

GENERAL_CLAIMS = [
    "Governments reduce interest rates to stimulate economic growth.",
    "Smoking increases the risk of lung cancer.",
    "Companies often acquire competitors to expand market share.",
    "Courts can reject appeals when legal requirements are not met.",
    "Exercise improves cardiovascular health.",
]

AMBIGUOUS_CLAIMS = [
    "Scientists found that the drug reduces blood pressure.",
    "The government has increased taxes in recent years.",
    "RBI cuts rates to support economic growth.",
    "Studies show that the treatment is effective.",
]

RBI_CLAIM = EVENT_CLAIMS[0][0]
GENERAL_CLAIM = GENERAL_CLAIMS[1]


def _evidence():
    return Evidence(
        evidence_id="scope-1", chunk_id="chunk-1", document_id="doc-1",
        text="Controlled evidence.", domain="general", source="test",
        language="en", metadata={},
    )


def _manager(monkeypatch):
    def fake_verify(claim, evidence_list, **_kwargs):
        return [VerificationResult(
            evidence_id=evidence.evidence_id, claim=claim,
            relationship="CONTRADICTS", same_event=False, claim_aligned=True,
            confidence=1.0, reasoning="Controlled response.", verifier="test",
            directly_establishes_proposition=True,
        ) for evidence in evidence_list]

    monkeypatch.setattr(verification_manager, "verify_evidence_batch", fake_verify)
    return verification_manager.EvidenceVerificationManager()


@pytest.mark.parametrize("claim,observed", EVENT_CLAIMS)
def test_event_claim_current_classification_is_deterministic(claim, observed):
    assert {classify_claim_scope(claim) for _ in range(20)} == {observed}


@pytest.mark.parametrize("claim", GENERAL_CLAIMS)
def test_general_claim_current_classification_is_deterministic(claim):
    assert {classify_claim_scope(claim) for _ in range(20)} == {"GENERAL"}


@pytest.mark.parametrize("claim", AMBIGUOUS_CLAIMS)
def test_ambiguous_claim_current_classification_is_deterministic(claim):
    assert {classify_claim_scope(claim) for _ in range(20)} == {"GENERAL"}


@pytest.mark.parametrize(
    "claim",
    [
        "The government approved the bill today.",
        "The court rejected the appeal last year.",
        "The regulator scheduled the hearing for tomorrow.",
        "The regulator scheduled the hearing for this week.",
        "The regulator scheduled the hearing for last week.",
        "The regulator scheduled the hearing for next week.",
        "The regulator scheduled the hearing for this month.",
        "The regulator scheduled the hearing for last month.",
        "The regulator scheduled the hearing for next month.",
        "The regulator scheduled the hearing for this year.",
        "The regulator scheduled the hearing for next year.",
    ],
)
def test_relative_time_anchor_marks_completed_occurrence(claim):
    assert classify_claim_scope(claim) == "EVENT_SPECIFIC"


@pytest.mark.parametrize(
    "claim",
    ["Governments reduce interest rates.", "Companies often acquire competitors."],
)
def test_short_general_claims_stay_general(claim):
    assert classify_claim_scope(claim) == "GENERAL"


def test_explicit_general_override_remains_general(monkeypatch):
    result = _manager(monkeypatch).verify(
        RBI_CLAIM, [_evidence()], claim_scope="GENERAL",
    )
    assert classify_claim_scope(RBI_CLAIM) == "EVENT_SPECIFIC"
    assert result.claim_scope == "GENERAL"
    assert result.overall_assessment == "CONTRADICTED"


def test_explicit_event_override_remains_event_specific(monkeypatch):
    result = _manager(monkeypatch).verify(
        GENERAL_CLAIM, [_evidence()], claim_scope="EVENT_SPECIFIC",
    )
    assert classify_claim_scope(GENERAL_CLAIM) == "GENERAL"
    assert result.claim_scope == "EVENT_SPECIFIC"
    assert result.overall_assessment == "INSUFFICIENT_EVIDENCE"


def test_explicit_override_precedes_automatic_classification(monkeypatch):
    manager = _manager(monkeypatch)
    auto = manager.verify(RBI_CLAIM, [_evidence()])
    overridden = manager.verify(RBI_CLAIM, [_evidence()], claim_scope="GENERAL")
    assert auto.claim_scope == "EVENT_SPECIFIC"
    assert overridden.claim_scope == "GENERAL"
    assert auto.overall_assessment != overridden.overall_assessment


def test_graph_extraction_keeps_explicit_override(monkeypatch):
    monkeypatch.setattr(
        graph, "get_claim_agent",
        lambda: type("Agent", (), {"extract_claim": lambda self, text: RBI_CLAIM})(),
    )
    extracted = graph.claim_extraction_node({
        "processed_text": RBI_CLAIM, "claim_scope": "GENERAL",
    })
    assert extracted["claim"] == RBI_CLAIM
    assert extracted["claim_scope"] == "GENERAL"
    assert len(extracted["atomic_claims"]) == 1
    assert extracted["atomic_claims"][0].text == RBI_CLAIM
    assert extracted["decomposition_diagnostics"]["component_count"] == 1


@pytest.mark.parametrize("scope", [None, ""])
def test_missing_or_empty_scope_falls_back_deterministically(monkeypatch, scope):
    manager = _manager(monkeypatch)
    results = [manager.verify(RBI_CLAIM, [_evidence()], claim_scope=scope) for _ in range(5)]
    assert {(result.claim_scope, result.overall_assessment) for result in results} == {
        ("EVENT_SPECIFIC", "INSUFFICIENT_EVIDENCE")
    }


def test_scope_values_are_restricted_to_allowed_enum():
    claims = [claim for claim, _ in EVENT_CLAIMS] + GENERAL_CLAIMS + AMBIGUOUS_CLAIMS
    assert {classify_claim_scope(claim) for claim in claims} <= {
        "GENERAL", "EVENT_SPECIFIC"
    }
    with pytest.raises(ValidationError):
        OverallVerification(
            claim=RBI_CLAIM, claim_scope="UNKNOWN", results=[],
            overall_assessment="INSUFFICIENT_EVIDENCE", overall_confidence=0.0,
        )


def test_graph_scope_survives_into_verification_aggregation(monkeypatch):
    manager = _manager(monkeypatch)
    calls = []
    pipeline = ClaimProcessingPipeline.__new__(ClaimProcessingPipeline)
    pipeline.verification_manager = manager

    class Pipeline:
        def process(self, *, claim, claim_scope, **_kwargs):
            calls.append(claim_scope)
            verification, failure = pipeline._verify_once(
                claim, [_evidence()], claim_scope=claim_scope,
            )
            return SimpleNamespace(
                local_evidence=[], external_evidence=[], fused_evidence=[_evidence()],
                verification_result=verification, verification_failure=failure,
                local_retrieval_error=None, external_source_errors=[],
            )

    monkeypatch.setattr(graph, "get_claim_processing_pipeline", lambda: Pipeline())
    monkeypatch.setattr(graph, "_build_retrieval_queries", lambda **_kwargs: [])
    result = graph.evidence_verification_node({
        "claim": RBI_CLAIM, "claim_scope": "GENERAL",
        "domain": {"domain": "general"},
    })
    assert calls == ["GENERAL"]
    assert result["verification_result"].claim_scope == "GENERAL"
    assert result["verification_result"].overall_assessment == "CONTRADICTED"


def test_pipeline_process_forwards_explicit_scope_before_verification(monkeypatch):
    seen = []
    pipeline = ClaimProcessingPipeline.__new__(ClaimProcessingPipeline)
    monkeypatch.setattr(
        pipeline, "_process_multiple_queries",
        lambda **kwargs: seen.append(kwargs["claim_scope"]),
    )
    pipeline.process(
        RBI_CLAIM, "general", retrieval_queries=[], claim_scope="GENERAL",
    )
    assert seen == ["GENERAL"]
