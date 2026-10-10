import ast
import difflib
import json
import logging
import re
from pathlib import Path
from typing import Optional


def _load_query_builder(responses=None):
    graph_path = Path(__file__).resolve().parents[1] / "agents" / "graph.py"
    tree = ast.parse(graph_path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_build_retrieval_queries")

    class MockLLMService:
        calls = []
        remaining = []

        def generate(self, prompt):
            self.calls.append(prompt)
            if not self.remaining:
                raise AssertionError("mock planner response exhausted")
            response = self.remaining.pop(0)
            return response if isinstance(response, str) else json.dumps(response)

    MockLLMService.remaining = list(responses or [{
        "queries": ['"Rahul Gandhi"', '"Priyanka Gandhi"',
                     '"Akashvani Bhavan"', '"INDIA bloc"']
    }])
    namespace = {
        "re": re, "json": json, "difflib": difflib, "Optional": Optional,
        "LLMService": MockLLMService,
        "logger": logging.getLogger(__name__),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(graph_path), "exec"), namespace)
    return namespace["_build_retrieval_queries"], MockLLMService


def _assert_grounded(queries, source):
    source_terms = set(re.findall(r"[^\W_]+(?:[.%][^\W_]+)*", source.casefold()))
    for query in queries:
        assert "?" not in query
        assert all(term.casefold() in source_terms for term in re.findall(
            r"[^\W_]+(?:[.%][^\W_]+)*", query
        ))


def test_political_input_is_dynamic_and_grounded():
    build_queries, _ = _load_query_builder([{"queries": [
        '"Rahul Gandhi" "Akashvani Bhavan"',
        '"Priyanka Gandhi" "Akashvani Bhavan"', '"INDIA bloc"']
    }])
    claim = "Rahul Gandhi and others were booked for forcible entry into Akashvani Bhavan."
    context = ("Rahul Gandhi, Priyanka Gandhi, several others booked over forcible "
               "entry into Akashvani Bhavan during INDIA bloc protest")
    queries = build_queries(claim, context, "political")
    assert 3 <= len(queries) <= 4
    assert len({query.casefold() for query in queries}) == len(queries)
    assert '"Rahul Gandhi"' in " ".join(queries)
    assert '"Akashvani Bhavan"' in " ".join(queries)
    assert '"INDIA bloc"' in " ".join(queries)
    _assert_grounded(queries, f"{claim} {context}")


def test_medical_input_uses_its_own_model_queries():
    build_queries, _ = _load_query_builder([{"queries": [
        "Aspirin cardiovascular risk", "clinical study aspirin", "aspirin adults"]
    }])
    claim = "Aspirin may reduce cardiovascular risk in adults."
    context = "A recent clinical study examined aspirin and cardiovascular risk."
    queries = build_queries(claim, context, "medical")
    assert 3 <= len(queries) <= 4
    _assert_grounded(queries, f"{claim} {context}")
    assert any("Aspirin" in query for query in queries)


def test_finance_input_preserves_numbers_and_entities():
    build_queries, _ = _load_query_builder([{"queries": [
        '"Federal Reserve" interest rates', '"Federal Reserve" 0.25%',
        "Federal Reserve decision"]
    }])
    claim = "The Federal Reserve lowered interest rates by 0.25%."
    context = "Markets reacted to the Federal Reserve interest rate decision."
    queries = build_queries(claim, context, "finance")
    assert 3 <= len(queries) <= 4
    assert any('"Federal Reserve"' in query for query in queries)
    assert any("0.25%" in query for query in queries)
    _assert_grounded(queries, f"{claim} {context}")


def test_generic_glued_token_is_repaired_and_returned():
    build_queries, _ = _load_query_builder([{"queries": [
        "forcibleentry", "report", "entry forcible"]}])
    source = "The report described forcible entry into a building."
    queries = build_queries(source, source, "general")
    assert "forcible entry" in queries
    assert "forcibleentry" not in queries


def test_ungrounded_output_is_rejected_and_retry_receives_feedback():
    build_queries, service = _load_query_builder([
        {"queries": ["invented synonym", "claim", "context"]},
        {"queries": ["Aspirin cardiovascular risk", "clinical study aspirin", "aspirin adults"]},
    ])
    claim = "Aspirin may reduce cardiovascular risk in adults."
    context = "A recent clinical study examined aspirin and cardiovascular risk."
    queries = build_queries(claim, context, "medical")
    assert len(queries) == 3
    assert len(service.calls) == 2
    assert "failed validation" in service.calls[1]
    assert "invented synonym" in service.calls[1]


def test_two_invalid_attempts_fall_back_to_original_claim():
    build_queries, service = _load_query_builder([
        {"queries": ["invented one", "invented two", "invented three"]},
        {"queries": ["still invented", "another invention", "not grounded"]},
    ])
    claim = "A claim with source terms."
    assert build_queries(claim, claim, "general") == [claim]
    assert len(service.calls) == 2


def test_duplicate_model_queries_are_deduplicated():
    build_queries, _ = _load_query_builder([{"queries": [
        "Federal Reserve", "federal reserve", "interest rates", "0.25%"]}])
    claim = "The Federal Reserve lowered interest rates by 0.25%."
    queries = build_queries(claim, claim, "finance")
    assert len(queries) == 3
    assert len({query.casefold() for query in queries}) == 3


def test_malformed_model_output_falls_back_after_retry():
    build_queries, service = _load_query_builder(["not json", "still not json"])
    claim = "A short factual claim."
    assert build_queries(claim, claim, "general") == [claim]
    assert len(service.calls) == 2


def test_generic_claim_allows_reordering_and_inflection():
    build_queries, _ = _load_query_builder([{"queries": [
        "rates reduced Federal Reserve", "markets reacted rate", "0.25%"]}])
    claim = "The Federal Reserve reduced rates by 0.25%."
    context = "Markets reacted after the Federal Reserve rate decision."

    queries = build_queries(claim, context, "general")

    assert 3 <= len(queries) <= 4
    assert "rates reduced Federal Reserve" in queries
    assert "markets reacted rate" in queries


def test_multi_word_phrase_spacing_is_preserved():
    build_queries, _ = _load_query_builder([{"queries": [
        '"World Health Organization" update',
        '"World Health Organization" report',
        "health organization"]}])
    claim = "The World Health Organization published a report."
    context = "The World Health Organization issued an update."

    queries = build_queries(claim, context, "science")

    assert any("World Health Organization" in query for query in queries)
    assert all("WorldHealthOrganization" not in query for query in queries)
