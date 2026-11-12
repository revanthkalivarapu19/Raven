import csv
import json
from types import SimpleNamespace

import pytest

from benchmarks.verification import (
    CATEGORIES,
    _write_results,
    _run_graph_until_supervisor,
    _run_verifier_only,
    calculate_metrics,
    evaluate_dataset,
    load_dataset,
    normalize_label,
    raven_category,
)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Real", "SUPPORTED"),
        ("true", "SUPPORTED"),
        ("Fake", "CONTRADICTED"),
        ("false", "CONTRADICTED"),
        ("unknown", "UNVERIFIED"),
        ("insufficient_evidence", "UNVERIFIED"),
        ("conflicting_evidence", "CONFLICTING"),
    ],
)
def test_ground_truth_label_normalization(label, expected):
    assert normalize_label(label) == expected


def test_unknown_ground_truth_label_is_rejected():
    with pytest.raises(ValueError, match="Unsupported evaluation label"):
        normalize_label("maybe")


def test_metrics_separate_abstentions_from_confident_errors():
    metrics = calculate_metrics(
        ["SUPPORTED", "CONTRADICTED", "UNVERIFIED", "CONFLICTING"],
        ["UNVERIFIED", "SUPPORTED", "UNVERIFIED", "CONTRADICTED"],
        [1.0, 2.0, 3.0, 4.0],
    )

    assert metrics["accuracy"] == pytest.approx(0.25)
    assert metrics["unverified_rate"] == pytest.approx(0.5)
    assert metrics["abstention_count"] == 2
    assert metrics["correct_abstention_count"] == 1
    assert metrics["safe_abstention_count"] == 1
    assert metrics["abstention_accuracy"] == pytest.approx(0.5)
    assert metrics["correct_abstention_rate"] == pytest.approx(1.0)
    assert metrics["incorrect_confident_prediction_count"] == 2
    assert metrics["classification_error_count"] == 2
    assert metrics["macro_precision"] == pytest.approx(0.125)
    assert metrics["macro_recall"] == pytest.approx(0.25)
    assert metrics["macro_f1"] == pytest.approx(1 / 6)
    assert metrics["confusion_matrix"]["SUPPORTED"]["UNVERIFIED"] == 1
    assert metrics["per_class"]["SUPPORTED"]["support"] == 1
    assert metrics["runtime_seconds"]["average"] == pytest.approx(2.5)
    assert metrics["runtime_seconds"]["p50"] == pytest.approx(2.5)
    assert metrics["runtime_seconds"]["p95"] == pytest.approx(3.85)
    assert set(metrics["confusion_matrix"]) == set(CATEGORIES)


def test_raven_conflict_is_reported_even_when_final_verdict_abstains():
    state = {
        "verification_result": {"overall_assessment": "CONFLICTING_EVIDENCE"},
        "final_output": {"final_verdict": "Unverified"},
    }
    assert raven_category(state) == "CONFLICTING"


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"final_output": {"final_verdict": "Real"}}, "SUPPORTED"),
        ({"supervisor_result": {"final_verdict": "Fake"}}, "CONTRADICTED"),
        ({"final_output": {"final_verdict": "Unverified"}}, "UNVERIFIED"),
        ({}, "UNVERIFIED"),
    ],
)
def test_raven_final_verdict_mapping(state, expected):
    assert raven_category(state) == expected


@pytest.mark.parametrize("suffix", [".csv", ".jsonl"])
def test_dataset_loader_reads_required_and_optional_fields(tmp_path, suffix):
    path = tmp_path / f"claims{suffix}"
    example = {
        "claim": " RBI reduced the repo rate. ",
        "ground_truth_label": "true",
        "domain": "Finance",
        "evidence": "optional reference evidence",
        "source": "optional reference source",
    }
    if suffix == ".csv":
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=example)
            writer.writeheader()
            writer.writerow(example)
    else:
        path.write_text(json.dumps(example) + "\n", encoding="utf-8")

    rows = load_dataset(path)
    assert rows[0]["claim"] == "RBI reduced the repo rate."
    assert rows[0]["ground_truth_label"] == "SUPPORTED"
    assert rows[0]["domain"] == "Finance"
    assert rows[0]["evidence"] == "optional reference evidence"


def test_evaluation_calls_real_runner_and_records_fallback_separately():
    calls = []

    def pipeline_runner(*, text):
        calls.append(text)
        return {
            "verification_result": {
                "overall_assessment": "INSUFFICIENT_EVIDENCE",
                "overall_confidence": 0.2,
            },
            "domain": {"domain": "Finance"},
            "verification_failure": None,
            "fused_evidence": [],
            "final_output": {
                "final_verdict": "Unverified",
                "confidence": 0.0,
            },
            "supervisor_result": {
                "reasoning_summary": "Insufficient evidence to decide.",
            },
            "errors": [],
        }

    rows, metrics = evaluate_dataset(
        [{
            "claim": "A tested claim",
            "ground_truth_label": "SUPPORTED",
            "domain": "Finance",
        }],
        pipeline_runner,
    )

    assert calls == ["A tested claim"]
    assert rows[0]["raven_verdict"] == "UNVERIFIED"
    assert rows[0]["outcome"] == "UNVERIFIED"
    assert rows[0]["verification_status"] == "INSUFFICIENT_EVIDENCE"
    assert rows[0]["error_or_fallback_reason"] == "Insufficient evidence to decide."
    assert rows[0]["dataset_annotations"] == {"domain": "Finance"}
    assert metrics["safe_abstention_count"] == 1
    assert metrics["incorrect_confident_prediction_count"] == 0
    assert metrics["supported_domain_examples_evaluated"] == 1


def test_pipeline_exception_is_recorded_separately_from_abstention():
    def failing_runner(*, text):
        raise RuntimeError("controlled pipeline failure")

    rows, metrics = evaluate_dataset(
        [{"claim": "A claim", "ground_truth_label": "CONTRADICTED"}],
        failing_runner,
    )

    assert rows[0]["raven_verdict"] is None
    assert rows[0]["verification_status"] == "PIPELINE_ERROR"
    assert rows[0]["error_or_fallback_reason"] == (
        "RuntimeError: controlled pipeline failure"
    )
    assert rows[0]["outcome"] == "PIPELINE_ERROR"
    assert metrics["total_example_count"] == 1
    assert metrics["completed_count"] == 0
    assert metrics["pipeline_error_count"] == 1
    assert metrics["example_count"] == 0
    assert metrics["safe_abstention_count"] == 0


def test_results_are_written_as_machine_readable_json_and_csv(tmp_path):
    results = [{
        "example_id": 1,
        "claim": "A claim",
        "ground_truth": "SUPPORTED",
        "raven_verdict": "UNVERIFIED",
        "confidence": 0.0,
        "verification_status": "INSUFFICIENT_EVIDENCE",
        "evidence_count": 0,
        "runtime_seconds": 0.25,
        "error_or_fallback_reason": "Insufficient evidence.",
        "outcome": "UNVERIFIED",
        "verifier_diagnostics": [{
            "evidence_index": 1,
            "evidence_id": "benchmark-1-1",
            "evidence_text": "Reference evidence.",
            "verification_relationship": "INSUFFICIENT",
            "confidence": 0.9,
            "same_event": False,
            "claim_aligned": False,
            "verification_status": "COMPLETED",
            "error_category": None,
            "elapsed_seconds": 0.25,
        }],
        "final_aggregation": {
            "final_verification_status": "INSUFFICIENT_EVIDENCE",
            "final_aggregate_relationship": "INSUFFICIENT",
            "aggregate_confidence": 0.0,
            "result_type": "safe_abstention",
        },
        "per_example_diagnosis": "insufficient_evidence",
        "runtime_diagnostics": {
            "total_elapsed_seconds": 0.25,
            "evidence_items_processed": 1,
            "verifier_calls": 1,
            "timed_out": False,
            "error_categories": [],
        },
        "dataset_annotations": {"domain": "Finance"},
    }]
    metrics = calculate_metrics(["SUPPORTED"], ["UNVERIFIED"], [0.25])

    json_path, csv_path = _write_results(results, metrics, tmp_path / "raven.v1")

    assert json_path.name == "raven.v1.json"
    assert csv_path.name == "raven.v1.csv"
    json_result = json.loads(json_path.read_text(encoding="utf-8"))["examples"][0]
    assert json_result == results[0]
    with csv_path.open(encoding="utf-8", newline="") as stream:
        csv_result = next(csv.DictReader(stream))
    assert csv_result["raven_verdict"] == "UNVERIFIED"
    assert csv_result["outcome"] == "UNVERIFIED"
    assert json.loads(csv_result["dataset_annotations"]) == {"domain": "Finance"}
    assert json.loads(csv_result["verifier_diagnostics"]) == results[0]["verifier_diagnostics"]
    assert json.loads(csv_result["final_aggregation"]) == results[0]["final_aggregation"]
    assert json.loads(csv_result["runtime_diagnostics"]) == results[0]["runtime_diagnostics"]
    assert csv_result["per_example_diagnosis"] == "insufficient_evidence"


def test_unknown_domain_is_separately_counted_and_excluded():
    calls = []

    def pipeline_runner(*, text):
        calls.append(text)
        return {
            "domain": {"domain": "Unknown"},
            "verification_result": {"overall_assessment": "INSUFFICIENT_EVIDENCE"},
            "final_output": {"final_verdict": "Unverified"},
            "fused_evidence": [],
            "errors": [],
        }

    rows, metrics = evaluate_dataset(
        [{
            "claim": "A claim outside Raven's domains",
            "ground_truth_label": "UNVERIFIED",
            "domain": "Finance",
        }],
        pipeline_runner,
        supported_domains_only=True,
    )

    assert calls == ["A claim outside Raven's domains"]
    assert rows[0]["outcome"] == "UNSUPPORTED_DOMAIN"
    assert rows[0]["raven_verdict"] is None
    assert metrics["supported_domain_examples_evaluated"] == 0
    assert metrics["unsupported_domain_count"] == 1
    assert metrics["example_count"] == 0


def test_unsupported_dataset_domain_is_excluded_without_running_graph():
    def pipeline_runner(**_kwargs):
        pytest.fail("unsupported dataset domains must not run the graph")

    rows, metrics = evaluate_dataset(
        [{
            "claim": "A sports claim",
            "ground_truth_label": "SUPPORTED",
            "domain": "Sports",
        }],
        pipeline_runner,
        supported_domains_only=True,
    )

    assert rows[0]["outcome"] == "UNSUPPORTED_DOMAIN"
    assert metrics["unsupported_domain_count"] == 1
    assert metrics["supported_domain_examples_evaluated"] == 0
    assert metrics["pipeline_error_count"] == 0


def test_supported_domain_filter_requires_explicit_domains():
    with pytest.raises(ValueError, match="requires an explicit domain"):
        evaluate_dataset(
            [{"claim": "A claim", "ground_truth_label": "SUPPORTED"}],
            lambda **_kwargs: {},
            supported_domains_only=True,
        )


def test_verdict_stream_stops_before_xai(monkeypatch):
    from agents import graph as raven_graph

    events = []

    class FakeApp:
        def stream(self, _state, *, stream_mode):
            assert stream_mode == "values"

            def snapshots():
                events.append("supervisor")
                yield {
                    "domain": {"domain": "Finance"},
                    "supervisor_result": {"final_verdict": "Real"},
                }
                events.append("xai")
                yield {
                    "domain": {"domain": "Finance"},
                    "supervisor_result": {"final_verdict": "Real"},
                    "final_output": {"final_verdict": "Real"},
                }

            return snapshots()

    monkeypatch.setattr(raven_graph, "get_raven_graph", lambda: FakeApp())
    monkeypatch.setattr(
        raven_graph,
        "_config",
        SimpleNamespace(request_deadline_seconds=10),
    )

    state = _run_graph_until_supervisor("A benchmark claim")

    assert state["supervisor_result"]["final_verdict"] == "Real"
    assert "final_output" not in state
    assert events == ["supervisor"]


@pytest.mark.parametrize(
    ("relationship", "same_event", "expected", "expected_relationship"),
    [
        ("SUPPORTS", True, "SUPPORTED", "SUPPORTS"),
        ("CONTRADICTS", True, "CONTRADICTED", "CONTRADICTS"),
        ("CONTRADICTS", False, "UNVERIFIED", "INSUFFICIENT"),
        ("INSUFFICIENT", False, "UNVERIFIED", "INSUFFICIENT"),
    ],
)
def test_verifier_only_uses_production_verification_semantics(
    monkeypatch, relationship, same_event, expected, expected_relationship
):
    from rag_ingestion.verification import ollama_verifier

    calls = []

    def generate(_call):
        calls.append(True)
        value = {
            "SUPPORTS": "SUPPORTED",
            "CONTRADICTS": "CONTRADICTED",
            "INSUFFICIENT": "INSUFFICIENT",
        }[expected_relationship]
        return {"response": json.dumps({
            "results": [{"evidence_id": "benchmark-controlled-1-1", "relationship": value}],
        })}

    monkeypatch.setattr(ollama_verifier._LLM_RESILIENCE, "generate", generate)
    reference_text = (
        "Controlled reference: The agency did not reduce its rate on Friday; it increased it."
        if expected_relationship == "CONTRADICTS"
        else "Controlled reference: The agency reduced its rate on Friday."
    )
    state = _run_verifier_only(
        {
            "claim": "The agency reduced its rate on Friday.",
            "evidence": reference_text,
            "source": "Controlled reference source",
        },
        "controlled-1",
    )

    assert calls == [True]
    assert state["verification_result"].results[0].relationship == expected_relationship
    assert state["verification_result"].results[0].same_event is (same_event and expected_relationship != "INSUFFICIENT")
    assert state["final_output"]["final_verdict"] == expected
    assert state["fused_evidence"][0].text.startswith("Controlled reference:")
    assert state["verifier_diagnostics"][0]["verification_relationship"] == expected_relationship
    assert state["verifier_diagnostics"][0]["same_event"] is (same_event and expected_relationship != "INSUFFICIENT")
    assert state["verifier_diagnostics"][0]["verification_status"] == "COMPLETED"
    assert state["verifier_diagnostics"][0]["elapsed_seconds"] >= 0


def test_verifier_only_records_per_call_and_aggregate_diagnostics(monkeypatch):
    from rag_ingestion.verification import ollama_verifier

    def generate(_call):
        return {"response": json.dumps({
            "results": [{"evidence_id": "benchmark-1-1", "relationship": "SUPPORTED"}],
        })}

    monkeypatch.setattr(ollama_verifier._LLM_RESILIENCE, "generate", generate)
    rows, _ = evaluate_dataset(
        [{
            "claim": "The drug reduces blood pressure.",
            "ground_truth_label": "SUPPORTED",
            "evidence": "The drug reduces blood pressure.",
            "source": "Fixture source",
        }],
        mode="verifier-only",
    )

    result = rows[0]
    assert result["verifier_diagnostics"] == [{
        "evidence_index": 1,
        "evidence_id": "benchmark-1-1",
        "evidence_text": "The drug reduces blood pressure.",
        "verification_relationship": "SUPPORTS",
        "confidence": 0.9,
        "same_event": True,
        "claim_aligned": True,
        "verification_status": "COMPLETED",
        "error_category": None,
        "exception_type": None,
        "error": None,
        "elapsed_seconds": result["verifier_diagnostics"][0]["elapsed_seconds"],
    }]
    assert result["final_aggregation"] == {
        "final_verification_status": "SUPPORTED",
        "final_aggregate_relationship": "SUPPORTS",
        "aggregate_confidence": 0.9,
        "result_type": "confident_classification",
    }
    assert result["per_example_diagnosis"] == "correct_confident"
    assert result["runtime_diagnostics"]["evidence_items_processed"] == 1
    assert result["runtime_diagnostics"]["verifier_calls"] == 1
    assert result["runtime_diagnostics"]["timed_out"] is False


def test_verifier_timeout_fails_closed_as_unverified(monkeypatch):
    from rag_ingestion.llm_resilience import LLMInvocationError
    from rag_ingestion.verification import ollama_verifier

    def timeout(_call):
        raise LLMInvocationError("bounded request timed out", "timeout", retryable=True)

    monkeypatch.setattr(ollama_verifier._LLM_RESILIENCE, "generate", timeout)
    rows, metrics = evaluate_dataset(
        [{
            "claim": "A claim",
            "ground_truth_label": "SUPPORTED",
            "evidence": "Reference evidence.",
        }],
        mode="verifier-only",
    )

    result = rows[0]
    assert result["outcome"] == "UNVERIFIED"
    assert result["verifier_diagnostics"][0]["error_category"] == "timeout"
    assert result["verifier_diagnostics"][0]["verification_status"] == "FAILED_CLOSED"
    assert result["runtime_diagnostics"]["timed_out"] is True
    assert result["runtime_diagnostics"]["error_categories"] == ["timeout"]
    assert result["per_example_diagnosis"] == "verification_error"
    assert metrics["pipeline_error_count"] == 0


def test_verifier_only_rejects_reference_ids_without_evidence_text():
    with pytest.raises(ValueError, match="not evidence text"):
        _run_verifier_only(
            {
                "claim": "A claim",
                "evidence": json.dumps([[[1, 2, "Wikipedia_Page", 0]]]),
            },
            "fever-id",
        )
