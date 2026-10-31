import httpx
import ollama
import pytest

from agents.claim_extraction.claim_extraction_agent import (
    ClaimExtractionAgent,
    ClaimExtractionFidelityError,
    validate_claim_fidelity,
)


def _agent_with_response(response):
    agent = ClaimExtractionAgent()
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return response

    agent.llm.generate = generate
    return agent, prompts


def test_preserves_entity_percentage_relative_date_and_qualifier():
    text = "The Reserve Bank of India reduced the repo rate by 0.50% on Friday to boost economic growth."
    response = "The Reserve Bank of India reduced the repo rate by 0.50% on Friday to boost economic growth."
    agent, prompts = _agent_with_response(response)
    claim = agent.extract_claim(text)
    assert "Reserve Bank of India" in claim
    assert "repo rate" in claim
    assert "0.50%" in claim
    assert "on Friday" in claim
    assert "boost economic growth" in claim
    assert len(prompts) == 1


def test_preserves_numeric_date_and_year():
    source = "The agency approved 42 devices on 12 March 2024."
    agent, _ = _agent_with_response(source)
    claim = agent.extract_claim(source)
    assert "42" in claim
    assert "12 March 2024" in claim


def test_preserves_negation():
    source = "The study did not find evidence of harm in 2023."
    agent, _ = _agent_with_response(source)
    claim = agent.extract_claim(source)
    assert "did not" in claim
    assert "2023" in claim


def test_multiple_claims_remain_one_extracted_claim():
    agent, _ = _agent_with_response("The first policy was approved in 2022.")
    claims = agent.extract_claims("The first policy was approved in 2022. A second statement followed.")
    assert claims == ["The first policy was approved in 2022."]


@pytest.mark.parametrize(
    ("source", "candidate", "expected_violation"),
    [
        ("The agency approved 12 permits in 2025. Separately, the lab reported 7 samples in 2024.",
         "The agency approved 12 permits in 2025.", None),
        ("The agency approved 12 permits.", "The agency approved 13 permits.", "numeric"),
        ("The agency approved the plan in 2025.", "The agency approved the plan in 2026.", "temporal"),
        ("The fund increased its holdings.", "The fund decreased its holdings.", "polarity"),
        ("The study did not find a benefit.", "The study found a benefit.", "negation"),
        ("The bank reduced the rate by 0.50%.", "The bank reduced the rate to 0.50%.", "numeric"),
        ("The bank cut the rate by 0.50%.", "The bank reduced the rate by 0.50%.", None),
        ("The agency announced a new policy. The policy aims to increase output and reduce waste.",
         "The agency announced a new policy to increase output and reduce waste.", None),
        ("NASA launched a spacecraft from Florida on Tuesday. The craft will study a distant planet.",
         "NASA launched a spacecraft from Florida on Tuesday for a mission studying a distant planet.", None),
        ("The RBI raised the repo rate to 6.5 percent. The central bank increased interest rates to 6.5 percent.",
         "The RBI increased the repo rate to 6.5 percent.", None),
    ],
    ids=("omit-background", "numeric-mutation", "temporal-mutation", "polarity-mutation",
         "negation-mutation", "by-to-mutation", "synonym", "combined-grounded-clauses",
         "temporal-entity-paraphrase", "repeated-claim"),
)
def test_claim_fidelity_focused_regressions(source, candidate, expected_violation):
    violations = validate_claim_fidelity(source, candidate)
    if expected_violation is None:
        assert violations == []
    else:
        assert any(expected_violation in violation for violation in violations)


def test_one_main_claim_from_multiple_sentences():
    source = "The agency approved a policy in 2022. A separate study was published in 2023."
    candidate = "The agency approved a policy in 2022."
    agent, _ = _agent_with_response(candidate)
    assert agent.extract_claims(source) == [candidate]


def test_cleaning_does_not_remove_material_terms():
    agent = ClaimExtractionAgent()
    cleaned = agent.clean_claim('Claim: "The company did not report 15% growth in 2021."')
    assert cleaned == "The company did not report 15% growth in 2021."


@pytest.mark.parametrize(
    ("source", "candidate", "expected_violation"),
    [
        (
            "The agency approved 12.5% of applications.",
            "The agency approved 15% of applications.",
            "numeric",
        ),
        (
            "The product weighs 1.25 kilograms.",
            "The product weighs 1.52 kilograms.",
            "numeric",
        ),
        (
            "The agency approved the plan on 12 March 2025.",
            "The agency approved the plan on 13 March 2025.",
            "numeric",
        ),
        (
            "The agency announced the change last year.",
            "The agency announced the change this year.",
            "temporal",
        ),
        (
            "The study did not find evidence of harm.",
            "The study found evidence of harm.",
            "negation",
        ),
        (
            "The rate increased by 0.5%.",
            "The rate decreased by 0.5%.",
            "polarity",
        ),
        (
            "The central bank reduced the rate by 0.50%.",
            "The central bank reduced the rate to 0.50%.",
            "numeric",
        ),
        (
            "The committee approved the measure.",
            "The committee approved the measure this year.",
            "temporal",
        ),
        (
            "The Central Bank of Exampleland approved the measure.",
            "The Central Bank of Otherland approved the measure.",
            "named entities",
        ),
        (
            "The central bank cut interest rates by 0.50%.",
            "The central bank reduced interest rates by 0.50%.",
            None,
        ),
    ],
    ids=(
        "percentage",
        "decimal",
        "date-and-year",
        "relative-time",
        "negation",
        "increase-decrease-polarity",
        "by-versus-to",
        "no-new-temporal-information",
        "named-entity",
        "legitimate-synonym-paraphrase",
    ),
)
def test_deterministic_fidelity_validator(source, candidate, expected_violation):
    violations = validate_claim_fidelity(source, candidate)
    if expected_violation is None:
        assert violations == []
    else:
        assert any(expected_violation in violation for violation in violations)


def test_fidelity_failure_allows_only_one_corrective_generation():
    agent, prompts = _agent_with_response(
        "The agency approved 5.50% this year."
    )
    responses = iter(
        [
            "The agency approved 5.50% this year.",
            "The agency approved 5.50% this year.",
        ]
    )
    agent.llm.generate = lambda prompt: (prompts.append(prompt), next(responses))[1]

    assert agent.extract_claim("The agency approved 0.50%.") == "The agency approved 0.50%."

    assert len(prompts) == 2
    assert "The agency approved 0.50%." in prompts[1]
    assert "The agency approved 5.50% this year." in prompts[1]


def test_fidelity_failure_returns_corrected_claim_after_one_retry():
    agent, prompts = _agent_with_response("")
    responses = iter(
        [
            "The central bank raised the rate to 0.50% this year.",
            "The central bank reduced the rate by 0.50%.",
        ]
    )
    agent.llm.generate = lambda prompt: (prompts.append(prompt), next(responses))[1]

    claim = agent.extract_claim("The central bank cut the rate by 0.50%.")

    assert claim == "The central bank reduced the rate by 0.50%."
    assert len(prompts) == 2


@pytest.mark.parametrize("source", [
    "The bank reduced the rate by 0.50%.",
    "The agency approved the plan on 12 March 2025.",
    "The agency announced the change last year.",
    "The Reserve Bank of India reduced the repo rate by 0.50%.",
])
def test_valid_standalone_claim_is_fallback_after_invalid_rewrite(source):
    agent, _ = _agent_with_response("")
    responses = iter(["The fund approved 2.75% yesterday.", "The fund approved 2.75% yesterday."])
    agent.llm.generate = lambda _prompt: next(responses)

    assert agent.extract_claim(source) == source


def test_whitespace_only_temporal_variation_is_non_material():
    assert validate_claim_fidelity(
        "The agency approved the plan last week.",
        "The agency approved the plan last   week.",
    ) == []


@pytest.mark.integration
def test_qwen_preserves_fidelity_for_rbi_reduction_input():
    model_name = "qwen2.5:3b"
    try:
        available_models = {
            model.model for model in ollama.Client(timeout=5).list().models
        }
    except (httpx.ConnectError, httpx.TimeoutException, ConnectionError) as exc:
        pytest.skip(f"Ollama integration is unavailable: {exc}")
    if model_name not in available_models:
        pytest.skip(f"Ollama integration requires the local {model_name} model")

    source = (
        "The Reserve Bank of India reduced the repo rate by 0.50% "
        "to support economic growth."
    )
    agent = ClaimExtractionAgent()
    claim = agent.extract_claim(source)

    assert validate_claim_fidelity(source, claim) == []
