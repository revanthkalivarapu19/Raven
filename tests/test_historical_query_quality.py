import ast
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional


def _builder(response):
    path = Path(__file__).resolve().parents[1] / "agents" / "graph.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_build_retrieval_queries")

    class MockLLM:
        def generate(self, prompt):
            return json.dumps(response)

    namespace = {"re": re, "json": json, "difflib": __import__("difflib"),
                 "Optional": Optional, "LLMService": MockLLM, "time": time,
                 "logger": logging.getLogger(__name__)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_build_retrieval_queries"]


def test_explicit_year_is_preserved():
    claim = "The agency changed the policy in 2020."
    build = _builder({"queries": ["agency policy 2020", "policy change 2020", "agency 2020"]})
    assert all("2020" in query for query in build(claim, claim, "general"))


def test_month_and_year_are_preserved():
    claim = "The committee approved the measure in June 2025."
    build = _builder({"queries": ["committee measure June 2025", "approved June 2025", "measure 2025"]})
    queries = build(claim, claim, "general")
    assert "June" in " ".join(queries)
    assert "2025" in " ".join(queries)


def test_relative_date_is_not_converted():
    claim = "The meeting happened last Friday."
    build = _builder({"queries": ["meeting last Friday", "meeting Friday", "last Friday meeting"]})
    queries = build(claim, claim, "general")
    assert "last Friday" in " ".join(queries)
    assert not re.search(r"\b20\d{2}\b", " ".join(queries))


def test_numeric_value_is_preserved():
    claim = "The rate changed by 0.50%."
    build = _builder({"queries": ["rate 0.50%", "changed 0.50%", "rate change 0.50%"]})
    assert "0.50%" in " ".join(build(claim, claim, "finance"))


def test_historical_rbi_fixture_is_runtime_agnostic():
    claim = "The Reserve Bank of India reduced the repo rate by 0.50% in June 2025."
    build = _builder({"queries": [
        '"Reserve Bank of India" repo rate June 2025',
        '"repo rate" 0.50% June 2025',
        '"Reserve Bank of India" reduced 2025',
    ]})
    queries = build(claim, claim, "finance")
    assert "June 2025" in " ".join(queries)
    assert "0.50%" in " ".join(queries)
