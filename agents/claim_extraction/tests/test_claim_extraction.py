import pytest

from agents.claim_extraction.claim_extraction_agent import ClaimExtractionAgent
from agents.claim_extraction.services.llm_service import LLMService


def test_rbi_repeated_claim():

    agent = ClaimExtractionAgent()

    text = """
    The RBI raised the repo rate to 6.5 percent. The central bank
    increased interest rates to 6.5 percent. India's repo rate now
    stands at 6.5 percent. The decision was taken as part of the
    central bank's monetary policy measures to control inflation
    and maintain economic stability.
    """

    claim = agent.extract_claim(text)

    print("\nInput:")
    print(text)

    print("\nExtracted Claim:")
    print(claim)

    assert claim != ""
    assert len(claim) < len(text) / 2


def test_long_paragraph():

    agent = ClaimExtractionAgent()

    text = """
    The Reserve Bank of India announced its latest monetary policy
    decision after reviewing several economic indicators. The
    central bank decided to maintain the repo rate at 6.5 percent.
    Officials considered inflation trends, economic growth,
    liquidity conditions, global economic developments and other
    financial indicators before making the decision. The RBI also
    stated that it would continue monitoring economic conditions
    before making future changes to interest rates.
    """

    claim = agent.extract_claim(text)

    print("\nExtracted Claim:")
    print(claim)

    assert claim != ""
    assert len(claim) < len(text) / 2


def test_nasa():

    agent = ClaimExtractionAgent()

    text = """
    NASA successfully launched a new spacecraft from Florida on
    Tuesday. The spacecraft is part of a scientific mission that
    will study a distant planet. Scientists expect the mission to
    collect important scientific data over several years. NASA
    engineers monitored the launch and confirmed that the spacecraft
    entered its planned trajectory.
    """

    claim = agent.extract_claim(text)

    print("\nExtracted Claim:")
    print(claim)

    assert claim != ""
    assert "NASA" in claim


def test_government():

    agent = ClaimExtractionAgent()

    text = """
    The government announced a new renewable energy policy.
    The policy is intended to increase renewable energy production.
    Officials said the policy will support clean energy development
    and reduce dependence on conventional energy sources. The
    government expects the policy to encourage additional investment
    in renewable energy projects.
    """

    claim = agent.extract_claim(text)

    print("\nExtracted Claim:")
    print(claim)

    assert claim != ""
    assert len(claim) < len(text) / 2


def test_empty_input():

    agent = ClaimExtractionAgent()

    claim = agent.extract_claim("")

    assert claim == ""


def _agent_with_response(response):
    agent = ClaimExtractionAgent()
    agent.llm.generate = lambda _prompt: response
    return agent


@pytest.mark.parametrize(
    ("source", "extracted", "required_details"),
    [
        (
            "FSSAI issued notices to Dr. Reddy's, Nestle Health Science, Amazon, "
            "Flipkart, and Netmeds regarding misleading product claims.",
            "FSSAI issued notices to Dr. Reddy's, Nestle Health Science, Amazon, "
            "Flipkart, and Netmeds regarding misleading product claims.",
            (
                "Dr. Reddy's",
                "Nestle Health Science",
                "Amazon",
                "Flipkart",
                "Netmeds",
            ),
        ),
        (
            "The agency approved 42 products with a 12.5% increase.",
            "The agency approved 42 products with a 12.5% increase.",
            ("42", "12.5%"),
        ),
        (
            "The regulator recently issued notices on 12 March 2025.",
            "The regulator recently issued notices on 12 March 2025.",
            ("recently", "12 March 2025"),
        ),
        (
            "The agency yesterday announced its decision last week.",
            "The agency yesterday announced its decision last week.",
            ("yesterday", "last week"),
        ),
        (
            "The regulator issued notices this year on Friday.",
            "The regulator issued notices this year on Friday.",
            ("this year", "on Friday"),
        ),
        (
            "The agency revised its guidance in 2025.",
            "The agency revised its guidance in 2025.",
            ("in 2025",),
        ),
        (
            "Nestle Health Science and Dr. Reddy's received notices.",
            "Nestle Health Science and Dr. Reddy's received notices.",
            ("Nestle Health Science", "Dr. Reddy's"),
        ),
    ],
    ids=(
        "multi-entity-list",
        "numbers-and-percentages",
        "time-and-date",
        "relative-time-qualifiers",
        "year-and-day-qualifiers",
        "year-qualifier",
        "organizations",
    ),
)
def test_extracted_claim_preserves_material_details(source, extracted, required_details):
    agent = _agent_with_response(extracted)

    claim = agent.extract_claim(source)

    for detail in required_details:
        assert detail in claim


def test_system_prompt_requires_exact_named_entity_preservation(monkeypatch):
    service = LLMService()
    captured = {}

    def fake_chat(**kwargs):
        captured["messages"] = kwargs["messages"]
        return {"message": {"content": "A concise factual claim."}}

    monkeypatch.setattr(service._resilience, "chat", lambda call: call())
    monkeypatch.setattr("agents.claim_extraction.services.llm_service.ollama.chat", fake_chat)

    service.generate("A source paragraph.")

    system_prompt = " ".join(captured["messages"][0]["content"].split())
    assert "Never change a number, percentage, date, year, quantity, named entity" in system_prompt
    assert "never introduce factual information absent from the input" in system_prompt
    assert "Preserve factual polarity and directional/comparative meaning" in system_prompt
    assert "changed \"by\" a value or changed \"to\" a value" in system_prompt
    assert "all underlying factual values and relations remain unchanged" in system_prompt
    assert "every name in a multi-entity list" in system_prompt
    assert "never replace those names with a generic group label" in system_prompt
    assert "such as \"major companies\"" in system_prompt
    assert "include every listed member in the extracted claim" in system_prompt
    assert "even if the result exceeds the usual word target" in system_prompt
    assert "Preserve every explicit temporal qualifier" in system_prompt
    assert "\"recently\"" in system_prompt
    assert "\"yesterday\"" in system_prompt
    assert "\"last week\"" in system_prompt
    assert "\"this year\"" in system_prompt
    assert "\"in 2025\"" in system_prompt
    assert "\"on Friday\"" in system_prompt
    assert "Do not omit, weaken, or generalize these expressions" in system_prompt
    assert "use more words when necessary" in system_prompt


def test_request_prompt_preserves_explicit_multi_entity_lists():
    agent = ClaimExtractionAgent()
    captured = {}

    def fake_generate(prompt):
        captured["prompt"] = prompt
        return "FSSAI issued notices to major companies, including Alpha and Beta."

    agent.llm.generate = fake_generate
    agent.extract_claim(
        "FSSAI issued notices to major companies, including Alpha and Beta."
    )

    assert "preserve every explicitly named person, organization" in captured["prompt"]
    assert "every item in a list introduced by wording" in captured["prompt"]
    assert "Never replace named list items with a" in captured["prompt"]
    assert "even when that makes the claim longer" in captured["prompt"]
    assert "Preserve explicit temporal qualifiers" in captured["prompt"]
    assert "Do not omit, weaken, or generalize explicit time information" in captured["prompt"]


def test_clean_claim_normalizes_curly_apostrophes():
    agent = ClaimExtractionAgent()

    claim = agent.clean_claim(
        "Dr. Reddy’s and Nestle Health Science received notices."
    )

    assert claim == "Dr. Reddy's and Nestle Health Science received notices."