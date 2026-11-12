import json
from types import SimpleNamespace

import pytest

from agents.claim_decomposition import decompose_claim, is_likely_compound


def test_geocentric_claim_splits_into_orbital_and_center_claims():
    claim = "The Earth is the center of the solar system and the Sun orbits Earth."
    result = decompose_claim(claim, fallback_claim=claim, generate=lambda _prompt: json.dumps({
        "claims": ["Earth is the center of the solar system.", "The Sun orbits Earth."]
    }))
    assert [item.text for item in result.claims] == [
        "Earth is the center of the solar system.",
        "The Sun orbits Earth.",
    ]


def test_compound_claim_uses_grounded_model_decomposition():
    claim = (
        "Thick smog and hazardous air quality (AQI crossing 400 in several areas "
        "and over 200 firecracker timing violation cases filed) have blanketed "
        "New Delhi following Diwali celebrations"
    )

    result = decompose_claim(claim, fallback_claim=claim, generate=lambda _prompt: json.dumps({"claims": [
        "Thick smog and hazardous air quality have blanketed New Delhi following Diwali celebrations",
        "AQI crossing 400 in several areas",
        "Over 200 firecracker timing violation cases filed",
    ]}))

    assert [item.id for item in result.claims] == ["C1", "C2", "C3"]
    assert "smog" in result.claims[0].text.lower()
    assert "AQI crossing 400" in result.claims[1].text
    assert "Over 200" in result.claims[2].text


def test_current_news_compounds_are_not_tied_to_known_names():
    telangana = (
        "Telangana Chief Minister announced that the radial road from Shamshabad "
        "to Kodangal will be completed by December 2027, HMWS&SB fined "
        "Akshayakalpa Farms ₹20,000 for wasting water, and Hyderabad firefighters "
        "battled multiple Diwali fire incidents across the state."
    )
    iran = (
        "Marco Rubio noted that US-Iran talks have not been productive, calling "
        "Iranian demands unacceptable, while Donald Trump stated that renewed "
        "US strikes on Iran are highly unlikely."
    )

    telangana_result = decompose_claim(telangana, fallback_claim=telangana, generate=lambda _prompt: json.dumps({"claims": [
        "Telangana Chief Minister announced that the radial road from Shamshabad to Kodangal will be completed by December 2027",
        "HMWS&SB fined Akshayakalpa Farms ₹20,000 for wasting water",
        "Hyderabad firefighters battled multiple Diwali fire incidents across the state",
    ]}))
    iran_result = decompose_claim(iran, fallback_claim=iran, generate=lambda _prompt: json.dumps({"claims": [
        "Marco Rubio noted that US-Iran talks have not been productive",
        "Donald Trump stated that renewed US strikes on Iran are highly unlikely",
    ]}))

    assert len(telangana_result.claims) == 3
    assert len(iran_result.claims) == 2
    assert "HMWS&SB fined" in telangana_result.claims[1].text
    assert "Hyderabad firefighters battled" in telangana_result.claims[2].text
    assert "Donald Trump stated" in iran_result.claims[1].text
from agents.query_planner import plan_atomic_queries, plan_queries
from agents.graph import _aggregate_claim_records
from rag_ingestion.verification.verification_schema import OverallVerification, VerificationResult


def test_concise_claim_does_not_call_decomposition_model():
    result = decompose_claim(
        "Smoking increases the risk of lung cancer.",
        fallback_claim="Smoking increases the risk of lung cancer.",
        generate=lambda _prompt: pytest.fail("concise claims should not use LLM decomposition"),
    )
    assert [item.text for item in result.claims] == ["Smoking increases the risk of lung cancer."]
    assert not result.used_fallback


def test_compound_claim_decomposition_is_structured_and_grounded():
    source = (
        "Telangana Chief Minister announced that the radial road from Shamshabad to Kodangal "
        "will be completed by December 2027, HMWS&SB fined Akshayakalpa Farms ₹20,000 "
        "for wasting water, and Hyderabad firefighters battled multiple Diwali fire incidents "
        "across the state."
    )
    values = [
        "Telangana Chief Minister announced that the radial road from Shamshabad to Kodangal will be completed by December 2027",
        "HMWS&SB fined Akshayakalpa Farms ₹20,000 for wasting water",
        "Hyderabad firefighters battled multiple Diwali fire incidents across the state",
    ]
    result = decompose_claim(source, fallback_claim=source, generate=lambda _prompt: json.dumps({"claims": values}))
    assert len(result.claims) == 3
    assert [claim.id for claim in result.claims] == ["C1", "C2", "C3"]
    assert "2027" in result.claims[0].temporal_context["expressions"][0]
    assert "Akshayakalpa Farms" in result.claims[1].entities
    assert not result.used_fallback


@pytest.mark.parametrize("output", ["not json", "{}", '{"claims": ["invented claim"]}'])
def test_malformed_or_ungrounded_decomposition_falls_back_to_single_claim(output):
    source = "Asha announced the new road date and Ravi confirmed the revised annual budget yesterday."
    fallback = "Asha announced the road date."
    result = decompose_claim(source, fallback_claim=fallback, generate=lambda _prompt: output)
    assert result.used_fallback
    assert result.claims[0].text == fallback


def test_graph_preserves_standalone_input_without_rewriting_it(monkeypatch):
    from agents import graph

    source = (
        "On 8 November 2026, the European Commission announced a €2 billion "
        "wildfire recovery fund for Portugal."
    )

    class FailedDecompositionAgent:
        class LLM:
            @staticmethod
            def generate(_prompt):
                return '{"claims": ["On 8 November 2026"]}'

        llm = LLM()

        @staticmethod
        def extract_claim(_text):
            return "On 8 November 2026"

    monkeypatch.setattr(graph, "get_claim_agent", lambda: FailedDecompositionAgent())
    result = graph.claim_extraction_node({"text": source})

    assert result["claim"] == source
    assert result["atomic_claims"][0].text == source
    assert result["decomposition_diagnostics"]["used_fallback"] is False


def test_query_planner_uses_original_concise_claim_without_expansion():
    claim = "Marco Rubio said Iran talks were not productive."
    result = plan_queries(claim, expanded_planner=lambda **_kwargs: pytest.fail("unexpected expansion"))
    assert result == [claim]


def test_query_planner_bounds_expanded_queries_and_falls_back():
    claim = "This is a deliberately long compound claim with many grounded factual details and a second independent factual proposition, including dates, locations, people, organizations, amounts, actions, events, relevant context, and a final qualifying relationship that matters."
    assert plan_queries(claim, expanded_planner=lambda **_kwargs: ["a", "b", "c", "d"]) == ["a", "b", "c"]
    assert plan_queries(claim, expanded_planner=lambda **_kwargs: []) == [claim]


@pytest.mark.parametrize(("claim", "required_terms"), [
    ("AQI crossing 400 in several areas", {"AQI", "400"}),
    ("over 200 firecracker timing violation cases filed", {"200", "firecracker", "violation"}),
    ("radial road from Shamshabad to Kodangal by December 2027", {"Shamshabad", "Kodangal", "December", "2027"}),
    ("HMWS&SB fined Akshayakalpa Farms ₹20,000 for wasting water", {"HMWS", "Akshayakalpa", "20,000"}),
    ("Marco Rubio called Iranian demands unacceptable", {"Marco", "Rubio", "Iranian", "unacceptable"}),
])
def test_atomic_query_planner_emits_compact_grounded_queries(claim, required_terms):
    queries = plan_atomic_queries(claim)

    assert 2 <= len(queries) <= 3
    assert all(len(query.split()) <= 12 for query in queries)
    assert all("?" not in query and query[-1] != "." for query in queries)
    combined = " ".join(queries).casefold()
    assert all(term.casefold() in combined for term in required_terms)
    assert all(
        token.casefold() in claim.casefold()
        for query in queries[:-1]
        for token in query.replace('"', "").split()
    )
    from agents.query_planner import _opposing_relation_query
    opposing = _opposing_relation_query(claim)
    if opposing:
        assert queries[-1] == opposing
        assert "not" in opposing.casefold()


def test_atomic_query_planner_recovers_relevant_context_and_current_date():
    claim = "AQI crossing 400 in several areas"
    context = (
        "Thick smog and hazardous air quality (AQI crossing 400 in several areas) "
        "have blanketed New Delhi following Diwali celebrations"
    )

    queries = plan_atomic_queries(claim, context=context, include_current_date=True)
    joined = " ".join(queries).casefold()

    assert "delhi" in joined
    assert "diwali" in joined
    assert "400" in joined
    assert "november 2026" in joined
    grounded_source = f"{claim} {context} November 2026".casefold()
    assert all(token.casefold() in grounded_source for query in queries for token in query.replace('"', "").split())


def test_compound_numeric_facts_get_separate_compact_queries():
    claim = (
        "Thick smog and hazardous air quality (AQI crossing 400 in several areas "
        "and over 200 firecracker timing violation cases filed) have blanketed "
        "New Delhi following Diwali celebrations"
    )
    queries = plan_atomic_queries(claim, context=claim, include_current_date=True)

    assert len(queries) == 3
    assert "400" in queries[0] and "200" not in queries[0]
    assert "200" in queries[1] and "400" not in queries[1]
    assert "400" not in queries[1] and "200" not in queries[0]
    assert all("November 2026" in query for query in queries[:2])
    assert "not" in queries[2].casefold()


@pytest.mark.parametrize(("claim", "opposing"), [
    ("A researcher has a signed permit", "A researcher does not have a signed permit"),
    ("The application is available", "The application is not available"),
    ("The service reduced its monthly fee", "The service did not reduce its monthly fee"),
    ("The satellite orbits the planet", "The satellite does not orbit the planet"),
    ("Water freezes at 0 degrees Celsius", "Water does not freeze at 0 degrees Celsius"),
    ("USB-C plugs can be inserted in either orientation", "USB-C plugs cannot be inserted in either orientation"),
])
def test_opposing_relation_query_is_derived_from_generic_claim_structure(claim, opposing):
    from agents.query_planner import _opposing_relation_query

    assert _opposing_relation_query(claim) == opposing
    queries = plan_atomic_queries(claim)
    assert queries[-1] == opposing


def test_each_atomic_component_uses_one_bounded_primary_query(monkeypatch):
    from agents import graph
    from agents.claim_decomposition import AtomicClaim

    calls = []

    class PipelineProbe:
        def process(self, **kwargs):
            calls.append(kwargs)
            verification = OverallVerification(
                claim=kwargs["claim"], results=[], overall_assessment="SUPPORTED",
                overall_confidence=0.8,
            )
            return SimpleNamespace(
                local_evidence=[], external_evidence=[], fused_evidence=[],
                verification_result=verification, verification_failure=None,
                provider_statuses={}, external_source_errors=[],
                retrieval_diagnostics={"providers": [], "final_evidence_count": 1},
            )

    monkeypatch.setattr(graph, "get_claim_processing_pipeline", lambda: PipelineProbe())
    original = (
        "AQI crossed 400 in Delhi following Diwali celebrations, and over 200 "
        "firecracker timing violations were filed in Delhi following Diwali."
    )
    components = [
        AtomicClaim(id="C1", text="AQI crossed 400 in several areas"),
        AtomicClaim(id="C2", text="Over 200 firecracker timing violations were filed"),
    ]

    result = graph._verify_atomic_claims(
        {"claim": original, "processed_text": original, "errors": []},
        components,
        "general",
    )

    assert len(calls) == 2
    assert [call["claim"] for call in calls] == [item.text for item in components]
    assert all(1 <= len(call["retrieval_queries"]) <= 3 for call in calls)
    assert "400" in " ".join(calls[0]["retrieval_queries"])
    assert "200" in " ".join(calls[1]["retrieval_queries"])
    assert result["retrieval_diagnostics"]["claims"][0]["component_id"] == "C1"
    assert result["retrieval_diagnostics"]["claims"][1]["component_id"] == "C2"


def _verification(claim, relationship, assessment):
    item = VerificationResult(
        evidence_id=f"ev-{claim}", claim=claim, relationship=relationship,
        confidence=0.9, reasoning="direct", verifier="unit", same_event=True,
        claim_aligned=True, directly_establishes_proposition=relationship != "INSUFFICIENT",
    )
    return OverallVerification(
        claim=claim, results=[item], overall_assessment=assessment,
        overall_confidence=0.9 if assessment in {"SUPPORTED", "CONTRADICTED"} else 0,
    )


def test_claim_aggregate_uses_conjunction_semantics_and_preserves_unresolved_failures():
    records = [
        {"verification_result": _verification("C1", "SUPPORTS", "SUPPORTED")},
        {"verification_result": _verification("C2", "SUPPORTS", "SUPPORTED")},
    ]
    assessment, confidence, results, failure = _aggregate_claim_records("compound", records)
    assert (assessment, confidence, len(results), failure) == ("SUPPORTED", 0.9, 2, None)

    mixed = [records[0], {"verification_result": _verification("C2", "CONTRADICTS", "CONTRADICTED")}]
    assessment, confidence, _, failure = _aggregate_claim_records("compound", mixed)
    assert assessment == "CONTRADICTED"
    assert confidence == 0.9
    assert failure is None

    missing = [records[0], {"verification_failure": SimpleNamespace(error_type="Timeout")}]
    assessment, confidence, _, failure = _aggregate_claim_records("compound", missing)
    assert assessment == "INSUFFICIENT_EVIDENCE"
    assert confidence == 0
    assert failure.error_type == "Timeout"

    contradicted_with_unresolved = [
        {"verification_result": _verification("C1", "CONTRADICTS", "CONTRADICTED")},
        {"verification_failure": SimpleNamespace(error_type="Timeout")},
    ]
    assessment, confidence, _, failure = _aggregate_claim_records(
        "compound", contradicted_with_unresolved
    )
    assert assessment == "CONTRADICTED"
    assert confidence == 0.9
    assert failure is None
