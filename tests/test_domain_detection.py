import json

import pytest

from agents.domain_detection.domain_agent import DomainDetectionAgent


@pytest.mark.parametrize(
    ("claim", "domain", "reason"),
    [
        (
            "An agency warned consumers about misleading claims made for a dietary supplement.",
            "Medical",
            "The dietary supplement claim concerns health and consumer safety.",
        ),
        (
            "The central bank raised its policy interest rate to control inflation.",
            "Finance",
            "The interest rate and inflation are monetary policy topics.",
        ),
        (
            "The opposition party announced a nationwide protest against the new bill.",
            "Politics",
            "The opposition party and protest are political actions.",
        ),
        (
            "Researchers observed a previously unknown particle in the experiment.",
            "Science",
            "The experiment reports a scientific finding.",
        ),
    ],
)
def test_supported_domains_are_returned_with_existing_output_schema(
    monkeypatch, claim, domain, reason
):
    response = {"domain": domain, "confidence": 0.9, "reason": reason}
    agent = DomainDetectionAgent()
    monkeypatch.setattr(agent, "call_llm", lambda prompt: json.dumps(response))

    result = agent.detect_domain(claim)

    assert result["domain"] == domain
    assert 0.0 < result["confidence"] <= 0.9
    assert set(result) == {"domain", "confidence", "reason"}
    assert result["reason"]


def test_prompt_lists_only_supported_domains_and_constrains_unknown():
    prompt = DomainDetectionAgent().build_prompt("A representative claim.")

    assert "1. Medical\n2. Politics\n3. Finance\n4. Science" in prompt
    assert "Never return Unknown for a claim that clearly fits" in prompt
    assert "reason must be non-empty" in prompt
    assert "company names do not imply finance" in prompt.lower()
    assert "product names do not imply finance" in prompt.lower()
    assert "MAIN SUBJECT and FACTUAL ACTION" in prompt


def test_unknown_with_high_confidence_recovers_clear_medical_claim(monkeypatch):
    agent = DomainDetectionAgent()
    response = {
        "domain": "Unknown",
        "confidence": 1.0,
        "reason": "",
    }
    monkeypatch.setattr(agent, "call_llm", lambda prompt: json.dumps(response))

    result = agent.detect_domain(
        "An agency warned consumers about misleading claims made for a dietary supplement."
    )

    assert result["domain"] == "Medical"
    assert result["confidence"] < 1.0
    assert result["reason"]


def test_goyal_eci_political_response_is_kept_without_recovery_cue(monkeypatch):
    claim = (
        "Goyal defended ECI against Congress, citing former officials' "
        "appointments and roles."
    )
    response = {
        "domain": "Politics",
        "confidence": 0.9,
        "reason": (
            "The claim involves political figures and political actions, "
            "including Goyal defending the ECI against Congress."
        ),
    }
    agent = DomainDetectionAgent()
    monkeypatch.setattr(agent, "call_llm", lambda prompt: json.dumps(response))

    result = agent.detect_domain(claim)

    assert result == response


def test_herzog_eisenkot_government_formation_claim_routes_to_politics(monkeypatch):
    agent = DomainDetectionAgent()
    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda _prompt: json.dumps({
            "domain": "Unknown", "confidence": 0.9, "reason": "Unclear."
        }),
    )

    result = agent.detect_domain(
        "Israeli President Isaac Herzog has tasked opposition politician Gadi "
        "Eisenkot with forming a government in a major setback for Benjamin Netanyahu."
    )

    assert result["domain"] == "Politics"
    assert result["confidence"] == 0.65


def test_health_product_regulatory_action_is_not_classified_as_finance(monkeypatch):
    agent = DomainDetectionAgent()
    response = {
        "domain": "Finance",
        "confidence": 0.9,
        "reason": "This involves financial aspects of the named companies.",
    }
    monkeypatch.setattr(agent, "call_llm", lambda prompt: json.dumps(response))

    result = agent.detect_domain(
        "A health regulator issued notices to several companies about misleading health and product claims."
    )

    assert result["domain"] == "Medical"
    assert result["domain"] != "Finance"
    assert "health" in result["reason"].lower()
    assert "safety" in result["reason"].lower()


@pytest.mark.parametrize(
    ("claim", "wrong_domain", "wrong_reason", "expected_domain"),
    [
        (
            "The opposition party announced a nationwide protest against the new bill.",
            "Finance",
            "The claim concerns the company's revenue and financial performance.",
            "Politics",
        ),
        (
            "An agency warned consumers about misleading claims made for a dietary supplement.",
            "Finance",
            "The claim concerns company revenue and profit.",
            "Medical",
        ),
    ],
)
def test_inconsistent_supported_response_is_recovered(
    monkeypatch, claim, wrong_domain, wrong_reason, expected_domain
):
    agent = DomainDetectionAgent()
    response = {
        "domain": wrong_domain,
        "confidence": 1.0,
        "reason": wrong_reason,
    }
    monkeypatch.setattr(agent, "call_llm", lambda prompt: json.dumps(response))

    result = agent.detect_domain(claim)

    assert result["domain"] == expected_domain
    assert result["confidence"] <= 0.75


@pytest.mark.parametrize(
    "response",
    [
        "not valid JSON",
        json.dumps({"domain": "Technology", "confidence": 0.9, "reason": "Unsupported."}),
        json.dumps({"domain": "Unknown", "confidence": 1.0, "reason": ""}),
    ],
)
def test_invalid_or_unknown_response_does_not_override_clear_claim(
    monkeypatch, response
):
    agent = DomainDetectionAgent()
    monkeypatch.setattr(agent, "call_llm", lambda prompt: response)

    result = agent.detect_domain(
        "The central bank raised its policy interest rate to control inflation."
    )

    assert result["domain"] == "Finance"
    assert result["confidence"] < 1.0
    assert result["reason"]


@pytest.mark.parametrize(
    "response",
    [
        "not valid JSON",
        json.dumps({"domain": "Technology", "confidence": 1.0, "reason": "Unsupported."}),
        json.dumps({"domain": "Unknown", "confidence": 1.0, "reason": ""}),
    ],
)
def test_invalid_or_unknown_response_fails_safely_for_ambiguous_claim(
    monkeypatch, response
):
    agent = DomainDetectionAgent()
    monkeypatch.setattr(agent, "call_llm", lambda prompt: response)

    result = agent.detect_domain("A town opened a new community center.")

    assert result == {
        "domain": "Unknown",
        "confidence": 0.0,
        "reason": "The claim is ambiguous or lacks a clear supported-domain subject.",
    }


def test_overlapping_strong_domain_signals_fail_safely(monkeypatch):
    agent = DomainDetectionAgent()
    response = json.dumps(
        {"domain": "Unknown", "confidence": 1.0, "reason": "Unclear."}
    )
    monkeypatch.setattr(agent, "call_llm", lambda prompt: response)

    result = agent.detect_domain(
        "A political party announced a health research policy."
    )

    assert result["domain"] == "Unknown"
    assert result["confidence"] == 0.0


def test_high_confidence_finance_with_nonsensical_reason_does_not_bypass_validation(
    monkeypatch,
):
    agent = DomainDetectionAgent()
    response = {
        "domain": "Finance",
        "confidence": 1.0,
        "reason": "The company announced a major change to its operations.",
    }
    monkeypatch.setattr(agent, "call_llm", lambda prompt: json.dumps(response))

    result = agent.detect_domain(
        "An agency warned consumers about misleading claims made for a dietary supplement."
    )

    assert result["domain"] == "Medical"
    assert result["confidence"] < 1.0
    assert "dietary supplement" not in result["reason"].lower()
    assert "health" in result["reason"].lower()


def test_ambiguous_company_claim_does_not_become_finance(monkeypatch):
    agent = DomainDetectionAgent()
    response = {
        "domain": "Finance",
        "confidence": 1.0,
        "reason": "The company announced a major change to its operations.",
    }
    monkeypatch.setattr(agent, "call_llm", lambda prompt: json.dumps(response))

    result = agent.detect_domain("The company announced a major change to its operations.")

    assert result["domain"] == "Unknown"
    assert result["confidence"] == 0.0


def test_unknown_response_recovers_unique_finance_signal(monkeypatch):
    agent = DomainDetectionAgent()
    response = {
        "domain": "Unknown",
        "confidence": 1.0,
        "reason": "Unclear.",
    }
    monkeypatch.setattr(agent, "call_llm", lambda prompt: json.dumps(response))

    result = agent.detect_domain(
        "The central bank raised its policy interest rate to control inflation."
    )

    assert result["domain"] == "Finance"
    assert result["confidence"] <= 0.75


def test_biomedical_claim_recovers_medical_domain(monkeypatch):
    agent = DomainDetectionAgent()
    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda _prompt: json.dumps({
            "domain": "Science",
            "confidence": 0.9,
            "reason": "The claim concerns a scientific finding about antiviral effects.",
        }),
    )

    result = agent.detect_domain(
        "IRG1 has antiviral effects against neurotropic viruses."
    )

    assert result["domain"] == "Medical"


def test_nonmedical_scientific_claim_remains_science(monkeypatch):
    agent = DomainDetectionAgent()
    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda _prompt: json.dumps({
            "domain": "Science",
            "confidence": 0.9,
            "reason": "The claim reports an experimental finding about a particle.",
        }),
    )

    result = agent.detect_domain(
        "Researchers observed a previously unknown particle in the experiment."
    )

    assert result["domain"] == "Science"


@pytest.mark.parametrize(
    "claim",
    [
        "Earth orbits the Sun.",
        "Earth revolves around the Sun.",
        "Planets orbit the Sun.",
    ],
)
def test_unknown_model_domain_does_not_use_claim_specific_recovery(monkeypatch, claim):
    agent = DomainDetectionAgent()
    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda _prompt: json.dumps({
            "domain": "Unknown",
            "confidence": 0.9,
            "reason": "Unclear.",
        }),
    )

    result = agent.detect_domain(claim)

    assert result["domain"] == "Unknown"
    assert result["confidence"] == 0.0


def test_unrelated_claim_does_not_recover_science_domain(monkeypatch):
    agent = DomainDetectionAgent()
    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda _prompt: json.dumps({
            "domain": "Unknown",
            "confidence": 0.9,
            "reason": "Unclear.",
        }),
    )

    result = agent.detect_domain("A town opened a new community center.")

    assert result["domain"] == "Unknown"
    assert result["confidence"] == 0.0


def test_medicine_alias_is_normalized_to_medical(monkeypatch):
    agent = DomainDetectionAgent()
    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda _prompt: json.dumps({
            "domain": "Medicine",
            "confidence": 0.8,
            "reason": "The claim concerns medical treatment and health.",
        }),
    )

    result = agent.detect_domain("The treatment reduces infection risk.")

    assert result["domain"] == "Medical"


def test_benchmark_supported_domain_filter_matches_production_contract():
    from benchmarks.verification import _is_supported_domain

    assert all(
        _is_supported_domain(domain)
        for domain in ("Medical", "Politics", "Finance", "Science")
    )
    assert not _is_supported_domain("Technology")
    assert not _is_supported_domain("Unknown")
