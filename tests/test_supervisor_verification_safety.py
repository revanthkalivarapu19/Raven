import ast
import logging
from pathlib import Path


def _load_supervisor_node():
    graph_path = Path(__file__).resolve().parents[1] / "agents" / "graph.py"
    tree = ast.parse(graph_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "supervisor_node"
    )
    namespace = {
        "RavenState": dict,
        "Dict": dict,
        "Any": object,
        "logger": logging.getLogger(__name__),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(graph_path), "exec"), namespace)
    return namespace


def _state(verification_result=None, verification_failure=None):
    return {
        "claim": "A test claim.",
        "domain": {"domain": "general"},
        "fused_evidence": [],
        "reflection_result": {},
        "persona_insights": [],
        "verification_result": verification_result,
        "verification_failure": verification_failure,
    }


def _unexpected_call(value):
    raise AssertionError("unexpected supervisor call")


def test_missing_verification_is_unverified_without_supervisor_call():
    namespace = _load_supervisor_node()
    calls = []
    namespace["get_supervisor_agent"] = lambda: lambda value: calls.append(value)

    result = namespace["supervisor_node"](_state())

    assert result["supervisor_result"]["final_verdict"] == "Unverified"
    assert result["supervisor_result"]["confidence_score"] == 0
    assert calls == []


def test_verification_failure_is_preserved_and_unverified():
    namespace = _load_supervisor_node()
    failure = {"status": "VERIFICATION_FAILURE"}
    namespace["get_supervisor_agent"] = lambda: _unexpected_call

    result = namespace["supervisor_node"](_state(verification_failure=failure))

    assert result["supervisor_result"]["final_verdict"] == "Unverified"
    assert result["verification_failure"] is failure


def test_insufficient_evidence_is_unverified():
    namespace = _load_supervisor_node()
    namespace["get_supervisor_agent"] = lambda: _unexpected_call

    result = namespace["supervisor_node"](
        _state({"overall_assessment": "INSUFFICIENT_EVIDENCE"})
    )

    assert result["supervisor_result"]["final_verdict"] == "Unverified"
    assert result["supervisor_result"]["confidence_score"] == 0


def test_conflicting_evidence_is_unverified():
    namespace = _load_supervisor_node()
    namespace["get_supervisor_agent"] = lambda: _unexpected_call

    result = namespace["supervisor_node"](
        _state({"overall_assessment": "CONFLICTING_EVIDENCE"})
    )

    assert result["supervisor_result"]["final_verdict"] == "Unverified"
    assert result["supervisor_result"]["trust_score"] == 0


def test_supported_verification_maps_to_real_without_waiting_for_supervisor():
    namespace = _load_supervisor_node()
    calls = []
    namespace["get_supervisor_agent"] = lambda: lambda value: calls.append(value)

    result = namespace["supervisor_node"](
        _state({
            "overall_assessment": "SUPPORTED", "overall_confidence": 0.91,
            "results": [{"evidence_id": "support-1", "relationship": "SUPPORTS",
                         "claim_aligned": True, "directly_establishes_proposition": True,
                         "same_event": True, "confidence": 0.91}],
        })
    )

    assert result["supervisor_result"]["final_verdict"] == "Real"
    assert result["supervisor_result"]["confidence_score"] == 91
    assert result["supervisor_result"]["key_evidence_used"] == ["support-1"]
    assert calls == []


def test_contradicting_verification_maps_to_fake_without_waiting_for_supervisor():
    namespace = _load_supervisor_node()
    namespace["get_supervisor_agent"] = lambda: _unexpected_call

    result = namespace["supervisor_node"](
        _state({
            "overall_assessment": "CONTRADICTED", "overall_confidence": 1.0,
            "results": [{"evidence_id": "contra-1", "relationship": "CONTRADICTS",
                         "claim_aligned": True, "directly_establishes_proposition": True,
                         "same_event": True, "confidence": 1.0}],
        })
    )

    assert result["supervisor_result"]["final_verdict"] == "Fake"
    assert result["supervisor_result"]["confidence_score"] == 100


def test_unsupported_or_ineligible_verification_stays_unverified():
    namespace = _load_supervisor_node()
    namespace["get_supervisor_agent"] = lambda: _unexpected_call

    for assessment, relationship in (("SUPPORTED", "SUPPORTS"), ("CONTRADICTED", "CONTRADICTS")):
        result = namespace["supervisor_node"](
            _state({"overall_assessment": assessment, "overall_confidence": 1.0,
                    "results": [{"evidence_id": "unclear", "relationship": relationship,
                                 "claim_aligned": False,
                                 "directly_establishes_proposition": False,
                                 "same_event": True, "confidence": 1.0}]})
        )
        assert result["supervisor_result"]["final_verdict"] == "Unverified"
