"""Evaluate Raven verdicts or the production evidence verifier on labeled data.

Run with:
    python -m benchmarks.verification DATASET.csv
    python -m benchmarks.verification DATASET.csv --supported-domains-only
    python -m benchmarks.verification DATASET.jsonl --mode verifier-only

Dataset rows require ``claim`` and ``ground_truth_label``. Optional fields such
as ``domain``, ``evidence``, and ``source`` are retained as dataset annotations.
Verdict mode streams Raven's graph only through the supervisor verdict, omitting
downstream XAI. Verifier-only mode bypasses retrieval and calls the production
evidence verification manager with supplied evidence text. In verifier-only
mode, ``evidence`` must be plain text, a JSON object with a ``text`` field, or
a JSON list of those objects/text values; evidence IDs or citation tuples alone
are not evidence text.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


CATEGORIES = ("SUPPORTED", "CONTRADICTED", "UNVERIFIED", "CONFLICTING")
OUTCOMES = (*CATEGORIES, "UNSUPPORTED_DOMAIN", "PIPELINE_ERROR")
# Keep benchmark filtering aligned with DomainDetectionAgent's public output
# vocabulary. Unknown is intentionally excluded from supported-domain scoring.
SUPPORTED_DOMAINS = frozenset({"FINANCE", "POLITICS", "MEDICAL", "SCIENCE"})
LABEL_ALIASES = {
    "supported": "SUPPORTED",
    "support": "SUPPORTED",
    "true": "SUPPORTED",
    "real": "SUPPORTED",
    "contradicted": "CONTRADICTED",
    "contradict": "CONTRADICTED",
    "false": "CONTRADICTED",
    "fake": "CONTRADICTED",
    "unverified": "UNVERIFIED",
    "unverified/unknown": "UNVERIFIED",
    "unknown": "UNVERIFIED",
    "insufficient": "UNVERIFIED",
    "insufficient_evidence": "UNVERIFIED",
    "conflicting": "CONFLICTING",
    "conflict": "CONFLICTING",
    "conflicting_evidence": "CONFLICTING",
}


def normalize_label(label: Any) -> str:
    """Map common dataset and Raven labels to the four evaluation categories."""
    key = str(label).strip().lower().replace("-", "_")
    try:
        return LABEL_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported evaluation label {label!r}; expected one of "
            f"{', '.join(CATEGORIES)} or a supported alias"
        ) from exc


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load labeled examples from CSV or JSON Lines."""
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        with dataset_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    elif suffix in {".jsonl", ".ndjson"}:
        rows = []
        with dataset_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at {dataset_path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"JSONL row {line_number} must be a JSON object"
                    )
                rows.append(row)
    else:
        raise ValueError("Dataset extension must be .csv, .jsonl, or .ndjson")

    validated = []
    for row_number, row in enumerate(rows, start=1):
        claim = row.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError(f"Dataset row {row_number} requires a non-empty claim")
        label = row.get("ground_truth_label")
        if label is None or not str(label).strip():
            raise ValueError(
                f"Dataset row {row_number} requires ground_truth_label"
            )
        validated.append({
            **row,
            "claim": claim.strip(),
            "ground_truth_label": normalize_label(label),
        })
    if not validated:
        raise ValueError(f"Dataset contains no examples: {dataset_path}")
    return validated


def _read_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def raven_category(state: dict[str, Any]) -> str:
    """Extract the final verdict, preserving explicit verification conflicts."""
    verification = state.get("verification_result")
    assessment = _read_value(verification, "overall_assessment")
    normalized_assessment = str(
        getattr(assessment, "value", assessment)
    ).strip().upper()
    if normalized_assessment == "CONFLICTING_EVIDENCE":
        return "CONFLICTING"

    final_output = state.get("final_output") or {}
    verdict = _read_value(final_output, "final_verdict")
    if verdict is None:
        verdict = _read_value(state.get("supervisor_result") or {}, "final_verdict")
    if verdict is not None:
        try:
            return normalize_label(verdict)
        except ValueError:
            return "UNVERIFIED"
    if normalized_assessment in {"SUPPORTED", "CONTRADICTED"}:
        return normalized_assessment
    return "UNVERIFIED"


def _domain_name(state: dict[str, Any]) -> str:
    domain = _read_value(state.get("domain"), "domain")
    return str(domain).strip() if domain is not None else ""


def _is_supported_domain(domain: Any) -> bool:
    return isinstance(domain, str) and domain.strip().upper() in SUPPORTED_DOMAINS


def _run_graph_until_supervisor(text: str) -> dict[str, Any]:
    """Stream the production graph through its verdict and stop before XAI."""
    from agents import graph as raven_graph
    from rag_ingestion.llm_resilience import request_deadline

    app = raven_graph.get_raven_graph()
    started = time.monotonic()
    initial_state = {
        "text": text,
        "image_path": None,
        "errors": [],
        "attempt_count": 1,
        "maximum_attempts_reached": False,
        "request_started_at": started,
        "request_deadline_at": started + raven_graph._config.request_deadline_seconds,
        "request_diagnostics": {
            "request_deadline_seconds": raven_graph._config.request_deadline_seconds,
        },
    }
    state = dict(initial_state)
    stream = app.stream(initial_state, stream_mode="values")
    try:
        with request_deadline(initial_state["request_deadline_at"]):
            for snapshot in stream:
                state = dict(snapshot)
                if state.get("supervisor_result"):
                    break
            else:
                raise RuntimeError("Raven graph ended before producing a supervisor verdict")
    finally:
        stream.close()
    finished = time.monotonic()
    diagnostics = dict(state.get("request_diagnostics") or {})
    diagnostics["total_duration_seconds"] = finished - started
    diagnostics["deadline_respected"] = finished <= initial_state["request_deadline_at"]
    diagnostics["deadline_reached"] = not diagnostics["deadline_respected"]
    state["request_diagnostics"] = diagnostics
    return state


def _make_verifier_evidence(row: dict[str, Any], example_id: Any) -> list[Any]:
    from rag_ingestion.retrieval.evidence_schema import Evidence

    raw_evidence = row.get("evidence")
    if raw_evidence is None or not str(raw_evidence).strip():
        raise ValueError("verifier-only mode requires non-empty evidence text")
    if isinstance(raw_evidence, str):
        try:
            decoded = json.loads(raw_evidence)
        except json.JSONDecodeError:
            decoded = raw_evidence
        if isinstance(decoded, (dict, list)):
            raw_evidence = decoded
    entries = raw_evidence if isinstance(raw_evidence, list) else [raw_evidence]
    source_value = row.get("source") or "Benchmark reference"
    evidence_items = []
    for index, entry in enumerate(entries, start=1):
        if isinstance(entry, dict):
            evidence_text = entry.get("text")
            source = entry.get("source") or source_value
            evidence_id = entry.get("evidence_id") or f"benchmark-{example_id}-{index}"
            url = entry.get("url")
        else:
            evidence_text = entry if isinstance(entry, str) else None
            source = source_value
            evidence_id = f"benchmark-{example_id}-{index}"
            url = None
        if not isinstance(evidence_text, str) or not evidence_text.strip():
            raise ValueError(
                "verifier-only evidence must be text or an object with non-empty text; "
                "dataset evidence references are not evidence text"
            )
        evidence_items.append(Evidence(
            evidence_id=str(evidence_id),
            chunk_id=str(evidence_id),
            document_id=f"benchmark-{example_id}",
            text=evidence_text,
            domain=str(row.get("domain") or "Benchmark"),
            source=str(source),
            language="en",
            retrieval_method="external",
            metadata={"benchmark_fixture": True},
            url=url,
        ))
    if not evidence_items:
        raise ValueError("verifier-only mode requires at least one evidence item")
    return evidence_items


def _run_verifier_only(
    row: dict[str, Any],
    example_id: Any,
    call_diagnostics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Use production verification/aggregation directly, without retrieval."""
    from rag_ingestion.verification import verification_manager
    from rag_ingestion.verification.verification_manager import EvidenceVerificationManager

    evidence_items = _make_verifier_evidence(row, example_id)
    call_diagnostics = call_diagnostics if call_diagnostics is not None else []
    original_verify_batch = verification_manager.verify_evidence_batch

    def timed_verify_batch(claim: str, evidence: Any, **kwargs: Any) -> Any:
        diagnostics = [{
            "evidence_index": index,
            "evidence_id": item.evidence_id,
            "evidence_text": item.text,
            "verification_relationship": None,
            "confidence": None,
            "same_event": None,
            "claim_aligned": None,
            "verification_status": "IN_PROGRESS",
            "error_category": None,
            "exception_type": None,
            "error": None,
        } for index, item in enumerate(evidence, start=1)]
        call_diagnostics.extend(diagnostics)
        started = time.perf_counter()
        try:
            results = original_verify_batch(claim, evidence, **kwargs)
        except Exception as exc:
            for diagnostic in diagnostics:
                diagnostic.update({
                    "verification_status": "ERROR",
                    "error_category": _exception_category(exc),
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                })
            raise
        elapsed = time.perf_counter() - started
        for diagnostic, result in zip(diagnostics, results):
            failed_closed = result.verifier_failed
            failure_text = result.rejection_reason or ""
            timed_out = "timeout" in failure_text.casefold() or "timed out" in failure_text.casefold()
            diagnostic.update({
                "verification_relationship": result.relationship,
                "confidence": result.confidence,
                "same_event": result.same_event,
                "claim_aligned": result.claim_aligned,
                "verification_status": "FAILED_CLOSED" if failed_closed else "COMPLETED",
                "error_category": "timeout" if failed_closed and timed_out else "invalid_response" if failed_closed else None,
                "exception_type": "VerifierCallFailure" if failed_closed else None,
                "error": failure_text if failed_closed else None,
                "elapsed_seconds": elapsed,
            })
        return results

    verification_manager.verify_evidence_batch = timed_verify_batch
    try:
        verification = EvidenceVerificationManager().verify(row["claim"], evidence_items)
    finally:
        verification_manager.verify_evidence_batch = original_verify_batch
    if verification.overall_assessment == "CONFLICTING_EVIDENCE":
        verdict = "CONFLICTING"
    elif verification.overall_assessment in {"SUPPORTED", "CONTRADICTED"}:
        verdict = verification.overall_assessment
    else:
        verdict = "UNVERIFIED"
    return {
        "claim": row["claim"],
        "domain": {"domain": row.get("domain") or "VERIFIER_ONLY"},
        "verification_result": verification,
        "verification_failure": None,
        "supervisor_result": {"final_verdict": verdict},
        "final_output": {"final_verdict": verdict, "confidence": verification.overall_confidence},
        "fused_evidence": evidence_items,
        "errors": [],
        "verifier_diagnostics": call_diagnostics,
        "verifier_calls": 1,
    }


def _exception_category(exc: Exception) -> str:
    current: BaseException | None = exc
    while current is not None:
        category = getattr(current, "kind", None)
        if category:
            return str(category)
        current = current.__cause__ or current.__context__
    if isinstance(exc, (TimeoutError,)):
        return "timeout"
    if isinstance(exc, ValueError):
        return "invalid_response"
    return type(exc).__name__.lower()


def _diagnose_example(
    ground_truth: str,
    prediction: str | None,
    outcome: str,
    verifier_diagnostics: Sequence[dict[str, Any]],
) -> str:
    if any(item.get("error_category") for item in verifier_diagnostics):
        return "verification_error"
    if outcome == "PIPELINE_ERROR":
        return "pipeline_error"
    if outcome == "UNSUPPORTED_DOMAIN":
        return "evidence_mismatch"
    if prediction is None:
        return "pipeline_error"
    if prediction != "UNVERIFIED":
        return "correct_confident" if prediction == ground_truth else "incorrect_confident"
    if ground_truth == "UNVERIFIED":
        return "correct_unverified"
    if any(
        item.get("verification_relationship") in {"SUPPORTS", "CONTRADICTS"}
        and item.get("claim_aligned") is False
        for item in verifier_diagnostics
    ):
        return "evidence_mismatch"
    if verifier_diagnostics:
        return "insufficient_evidence"
    return "incorrect_unverified"


def _final_aggregation_diagnostics(
    state: dict[str, Any],
    outcome: str,
    prediction: str | None,
) -> dict[str, Any]:
    verification = state.get("verification_result")
    status = _read_value(verification, "overall_assessment")
    if status is None:
        status = outcome
    status = str(getattr(status, "value", status))
    relationship = {
        "SUPPORTED": "SUPPORTS",
        "CONTRADICTED": "CONTRADICTS",
        "CONFLICTING_EVIDENCE": "CONFLICTING",
        "INSUFFICIENT_EVIDENCE": "INSUFFICIENT",
    }.get(status)
    if relationship is None and outcome == "PIPELINE_ERROR":
        relationship = "ERROR"
    confidence = _read_value(verification, "overall_confidence")
    if confidence is None:
        confidence = _read_value(
            state.get("final_output") or {},
            "confidence",
        )
    if outcome == "PIPELINE_ERROR":
        disposition = "pipeline_error"
    elif prediction == "UNVERIFIED":
        disposition = "safe_abstention"
    else:
        disposition = "confident_classification"
    return {
        "final_verification_status": status,
        "final_aggregate_relationship": relationship,
        "aggregate_confidence": confidence,
        "result_type": disposition,
    }


def _runtime_metrics(runtimes: Sequence[float]) -> dict[str, float]:
    if not runtimes:
        return {"average": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(runtimes)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "average": statistics.fmean(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
    }


def calculate_metrics(
    ground_truth: Sequence[str],
    predictions: Sequence[str],
    runtimes: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Calculate fixed-label classification and separate abstention metrics."""
    if len(ground_truth) != len(predictions):
        raise ValueError("Ground-truth and prediction counts must match")
    if runtimes is not None and len(runtimes) != len(ground_truth):
        raise ValueError("Runtime and prediction counts must match")

    truths = [normalize_label(label) for label in ground_truth]
    predicted = [normalize_label(label) for label in predictions]
    count = len(truths)
    confusion = {
        label: {prediction: 0 for prediction in CATEGORIES}
        for label in CATEGORIES
    }
    class_metrics = {}
    for truth, prediction in zip(truths, predicted):
        confusion[truth][prediction] += 1

    for label in CATEGORIES:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in CATEGORIES if other != label)
        false_negative = sum(confusion[label][other] for other in CATEGORIES if other != label)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[label].values()),
        }

    abstentions = [
        (truth, prediction)
        for truth, prediction in zip(truths, predicted)
        if prediction == "UNVERIFIED"
    ]
    correct_abstentions = sum(truth == "UNVERIFIED" for truth, _ in abstentions)
    safe_abstentions = sum(
        truth != "UNVERIFIED" for truth, _ in abstentions
    )
    incorrect_confident = sum(
        prediction != "UNVERIFIED" and prediction != truth
        for truth, prediction in zip(truths, predicted)
    )
    accuracy = (
        sum(truth == prediction for truth, prediction in zip(truths, predicted)) / count
        if count else 0.0
    )

    return {
        "example_count": count,
        "accuracy": accuracy,
        "macro_precision": statistics.fmean(
            values["precision"] for values in class_metrics.values()
        ),
        "macro_recall": statistics.fmean(
            values["recall"] for values in class_metrics.values()
        ),
        "macro_f1": statistics.fmean(
            values["f1"] for values in class_metrics.values()
        ),
        "per_class": class_metrics,
        "confusion_matrix": confusion,
        "unverified_rate": (
            sum(prediction == "UNVERIFIED" for prediction in predicted) / count
            if count else 0.0
        ),
        "abstention_count": len(abstentions),
        "abstention_accuracy": (
            correct_abstentions / len(abstentions) if abstentions else 0.0
        ),
        "correct_abstention_count": correct_abstentions,
        "correct_abstention_rate": (
            correct_abstentions / sum(truth == "UNVERIFIED" for truth in truths)
            if any(truth == "UNVERIFIED" for truth in truths) else 0.0
        ),
        "safe_abstention_count": safe_abstentions,
        "incorrect_confident_prediction_count": incorrect_confident,
        "classification_error_count": incorrect_confident,
        "runtime_seconds": _runtime_metrics(runtimes or []),
    }


def _reason_for_state(state: dict[str, Any]) -> str | None:
    failure = state.get("verification_failure")
    if failure is not None:
        message = _read_value(failure, "message")
        if message:
            return str(message)
        return str(failure)
    errors = state.get("errors") or []
    if errors:
        return "; ".join(str(error) for error in errors)
    if raven_category(state) == "UNVERIFIED":
        supervisor = state.get("supervisor_result") or {}
        return (
            _read_value(supervisor, "reasoning_summary")
            or "Raven returned an unverified verdict."
        )
    return None


def evaluate_dataset(
    rows: Sequence[dict[str, Any]],
    run_pipeline: Any = None,
    *,
    mode: str = "verdict",
    supported_domains_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate rows in verdict or verifier-only mode and separate exclusions."""
    if mode not in {"verdict", "verifier-only"}:
        raise ValueError("mode must be 'verdict' or 'verifier-only'")
    if supported_domains_only:
        missing_domains = [
            index for index, row in enumerate(rows, start=1)
            if not str(row.get("domain") or "").strip()
        ]
        if missing_domains:
            raise ValueError(
                "--supported-domains-only requires an explicit domain for every "
                f"dataset row; missing domain at row(s): {missing_domains}"
            )
    if mode == "verdict" and run_pipeline is None:
        run_pipeline = _run_graph_until_supervisor

    results = []
    for index, row in enumerate(rows, start=1):
        claim = row["claim"]
        example_id = row.get("id", index)
        annotated_domain = row.get("domain")
        if (
            mode == "verdict"
            and annotated_domain
            and not _is_supported_domain(annotated_domain)
        ):
            results.append(_excluded_result(
                row,
                example_id,
                "UNSUPPORTED_DOMAIN",
                f"Dataset domain {annotated_domain!r} is outside Raven's supported domains.",
                index,
            ))
            continue
        if (
            mode == "verdict"
            and supported_domains_only
            and not _is_supported_domain(annotated_domain)
        ):
            results.append(_excluded_result(
                row,
                example_id,
                "UNSUPPORTED_DOMAIN",
                f"Dataset domain {annotated_domain!r} is outside Raven's supported domains.",
                index,
            ))
            continue

        started = time.perf_counter()
        state: dict[str, Any] = {}
        call_diagnostics: list[dict[str, Any]] = []
        error = None
        try:
            pipeline_state = (
                _run_verifier_only(row, example_id, call_diagnostics)
                if mode == "verifier-only"
                else run_pipeline(text=claim)
            )
            if not isinstance(pipeline_state, dict):
                raise TypeError(
                    f"Raven pipeline returned {type(pipeline_state).__name__}, expected dict"
                )
            state = pipeline_state
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        runtime = time.perf_counter() - started

        verification = state.get("verification_result")
        failure = state.get("verification_failure")
        status = _read_value(verification, "overall_assessment")
        if status is None:
            status = _read_value(failure, "status")
        if status is None:
            status = "PIPELINE_ERROR" if error else "MISSING_VERIFICATION"
        else:
            status = str(getattr(status, "value", status))
        if failure is not None:
            status = "PIPELINE_ERROR"
            if error is None:
                error = str(_read_value(failure, "message") or failure)

        final_output = state.get("final_output") or {}
        supervisor = state.get("supervisor_result") or {}
        confidence = _read_value(final_output, "confidence")
        if confidence is None:
            confidence = _read_value(supervisor, "confidence_score")
        if confidence is None:
            confidence = _read_value(verification, "overall_confidence")

        actual_domain = (
            "VERIFIER_ONLY"
            if mode == "verifier-only"
            else _domain_name(state)
        )
        if error or status in {"PIPELINE_ERROR", "MISSING_VERIFICATION"}:
            outcome = "PIPELINE_ERROR"
            prediction = None
            if error is None:
                error = f"Raven produced status {status} without a verification result."
        elif mode == "verdict" and not _is_supported_domain(actual_domain):
            outcome = "UNSUPPORTED_DOMAIN"
            prediction = None
            error = (
                f"Raven detected unsupported domain {actual_domain!r}; "
                "excluded from classification metrics."
            )
        else:
            prediction = raven_category(state)
            outcome = prediction
        if not call_diagnostics:
            call_diagnostics = list(state.get("verifier_diagnostics") or [])
        diagnosis = _diagnose_example(
            row["ground_truth_label"],
            prediction,
            outcome,
            call_diagnostics,
        )
        error_categories = list(dict.fromkeys(
            item["error_category"]
            for item in call_diagnostics
            if item.get("error_category")
        ))
        timed_out = any(category == "timeout" for category in error_categories)
        result = {
            "example_id": example_id,
            "claim": claim,
            "ground_truth": row["ground_truth_label"],
            "raven_verdict": prediction,
            "outcome": outcome,
            "domain": actual_domain,
            "confidence": confidence,
            "verification_status": status,
            "evidence_count": len(state.get("fused_evidence") or []),
            "runtime_seconds": runtime,
            "error_or_fallback_reason": error or _reason_for_state(state),
            "verifier_diagnostics": call_diagnostics,
            "final_aggregation": _final_aggregation_diagnostics(
                state,
                outcome,
                prediction,
            ),
            "per_example_diagnosis": diagnosis,
            "runtime_diagnostics": {
                "total_elapsed_seconds": runtime,
                "evidence_items_processed": len(call_diagnostics),
                "verifier_calls": state.get("verifier_calls", len(call_diagnostics)),
                "timed_out": timed_out,
                "error_categories": error_categories,
            },
            "dataset_annotations": {
                key: value
                for key, value in row.items()
                if key not in {"id", "claim", "ground_truth_label"}
            },
        }
        results.append(result)

    completed = [
        row for row in results
        if row["outcome"] in CATEGORIES
    ]
    metrics = calculate_metrics(
        [row["ground_truth"] for row in completed],
        [row["raven_verdict"] for row in completed],
        [row["runtime_seconds"] for row in completed],
    )
    metrics["total_example_count"] = len(results)
    metrics["supported_domain_examples_evaluated"] = (
        len(completed) if mode == "verdict" else 0
    )
    metrics["verifier_only_examples_evaluated"] = (
        len(completed) if mode == "verifier-only" else 0
    )
    metrics["unsupported_domain_count"] = sum(
        row["outcome"] == "UNSUPPORTED_DOMAIN" for row in results
    )
    metrics["pipeline_error_count"] = sum(
        row["outcome"] == "PIPELINE_ERROR" for row in results
    )
    metrics["completed_count"] = len(completed)
    metrics["actual_classification_accuracy"] = metrics["accuracy"]
    metrics["safe_abstention_rate"] = (
        metrics["safe_abstention_count"]
        / sum(row["ground_truth"] != "UNVERIFIED" for row in completed)
        if any(row["ground_truth"] != "UNVERIFIED" for row in completed)
        else 0.0
    )
    metrics["mode"] = mode
    return results, metrics


def _excluded_result(
    row: dict[str, Any],
    example_id: Any,
    outcome: str,
    reason: str,
    index: int,
) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "claim": row["claim"],
        "ground_truth": row["ground_truth_label"],
        "raven_verdict": None,
        "outcome": outcome,
        "domain": row.get("domain"),
        "confidence": None,
        "verification_status": outcome,
        "evidence_count": 0,
        "runtime_seconds": 0.0,
        "error_or_fallback_reason": reason,
        "verifier_diagnostics": [],
        "final_aggregation": {
            "final_verification_status": outcome,
            "final_aggregate_relationship": None,
            "aggregate_confidence": None,
            "result_type": (
                "pipeline_error" if outcome == "PIPELINE_ERROR" else "safe_abstention"
            ),
        },
        "per_example_diagnosis": (
            "pipeline_error" if outcome == "PIPELINE_ERROR" else "evidence_mismatch"
        ),
        "runtime_diagnostics": {
            "total_elapsed_seconds": 0.0,
            "evidence_items_processed": 0,
            "verifier_calls": 0,
            "timed_out": False,
            "error_categories": [],
        },
        "dataset_annotations": {
            key: value
            for key, value in row.items()
            if key not in {"id", "claim", "ground_truth_label"}
        },
        "dataset_row": index,
    }


def _write_results(
    results: Sequence[dict[str, Any]],
    metrics: dict[str, Any],
    output_prefix: str | Path,
) -> tuple[Path, Path]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_name(f"{prefix.name}.json")
    csv_path = prefix.with_name(f"{prefix.name}.csv")
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {"metrics": metrics, "examples": results},
            stream,
            ensure_ascii=False,
            indent=2,
        )
        stream.write("\n")

    fieldnames = [
        "example_id",
        "claim",
        "ground_truth",
        "raven_verdict",
        "outcome",
        "domain",
        "confidence",
        "verification_status",
        "evidence_count",
        "runtime_seconds",
        "error_or_fallback_reason",
        "verifier_diagnostics",
        "final_aggregation",
        "per_example_diagnosis",
        "runtime_diagnostics",
        "outcome",
        "dataset_annotations",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = dict(result)
            row["dataset_annotations"] = json.dumps(
                row["dataset_annotations"], ensure_ascii=False
            )
            for field in (
                "verifier_diagnostics",
                "final_aggregation",
                "runtime_diagnostics",
            ):
                row[field] = json.dumps(row.get(field, []), ensure_ascii=False)
            writer.writerow(row)
    return json_path, csv_path


def _print_summary(metrics: dict[str, Any]) -> None:
    rows = [
        ("Mode", metrics["mode"]),
        ("Examples", metrics["example_count"]),
        ("Supported-domain evaluated", metrics["supported_domain_examples_evaluated"]),
        ("Unsupported-domain excluded", metrics["unsupported_domain_count"]),
        ("Pipeline errors excluded", metrics["pipeline_error_count"]),
        ("Actual classification accuracy", f"{metrics['actual_classification_accuracy']:.3f}"),
        ("Macro precision", f"{metrics['macro_precision']:.3f}"),
        ("Macro recall", f"{metrics['macro_recall']:.3f}"),
        ("Macro F1", f"{metrics['macro_f1']:.3f}"),
        ("UNVERIFIED rate", f"{metrics['unverified_rate']:.3f}"),
        ("Safe abstention rate", f"{metrics['safe_abstention_rate']:.3f}"),
        ("Safe abstentions", metrics["safe_abstention_count"]),
        ("Incorrect confident", metrics["incorrect_confident_prediction_count"]),
        ("Average runtime (s)", f"{metrics['runtime_seconds']['average']:.3f}"),
        ("P50 runtime (s)", f"{metrics['runtime_seconds']['p50']:.3f}"),
        ("P95 runtime (s)", f"{metrics['runtime_seconds']['p95']:.3f}"),
    ]
    label_width = max(len(label) for label, _ in rows)
    value_width = max(len(str(value)) for _, value in rows)
    border = f"+-{'-' * label_width}-+-{'-' * value_width}-+"
    print(border)
    print(f"| {'Metric'.ljust(label_width)} | {'Value'.ljust(value_width)} |")
    print(border)
    for label, value in rows:
        print(f"| {label.ljust(label_width)} | {str(value).ljust(value_width)} |")
    print(border)
    print("\nPer-class precision / recall / F1:")
    print("| Category | Precision | Recall | F1 | Support |")
    print("|---|---:|---:|---:|---:|")
    for label, values in metrics["per_class"].items():
        print(
            f"| {label} | {values['precision']:.3f} | {values['recall']:.3f} "
            f"| {values['f1']:.3f} | {values['support']} |"
        )
    print(
        "\nAbstentions on supported/contradicted/conflicting ground truth are "
        "reported as safe abstentions, not incorrect confident predictions."
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Input .csv, .jsonl, or .ndjson dataset")
    parser.add_argument(
        "--mode",
        choices=("verdict", "verifier-only"),
        default="verdict",
        help=(
            "verdict stops Raven's graph after supervisor output; verifier-only "
            "uses supplied evidence and the production verifier without retrieval"
        ),
    )
    parser.add_argument(
        "--supported-domains-only",
        action="store_true",
        help="Require explicit domain values; exclude rows outside Finance, Politics, Medical, Technology",
    )
    parser.add_argument(
        "--output-prefix",
        help="Output path prefix (default: <dataset_stem>_benchmark)",
    )
    args = parser.parse_args(argv)

    rows = load_dataset(args.dataset)
    results, metrics = evaluate_dataset(
        rows,
        mode=args.mode,
        supported_domains_only=args.supported_domains_only,
    )
    prefix = args.output_prefix or (
        Path(args.dataset).with_name(f"{Path(args.dataset).stem}_benchmark")
    )
    json_path, csv_path = _write_results(results, metrics, prefix)
    _print_summary(metrics)
    print(f"\nJSON results: {json_path}")
    print(f"CSV results:  {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
