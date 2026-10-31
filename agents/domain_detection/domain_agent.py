"""
domain_agent.py

Domain Detection Agent for the RAVEN Framework.
"""

import ollama
import re
from rag_ingestion.llm_resilience import LLMResilience

_LLM_RESILIENCE = LLMResilience()
from loguru import logger

from .prompt import DOMAIN_PROMPT
from .parser import parse_response
from .validator import SUPPORTED_DOMAINS, validate_response


_STRONG_DOMAIN_SIGNALS = {
    "Medical": (
        r"\bhealth(?:care|[- ]related)?\b",
        r"\bhealth[- ](?:and[- ]food[- ]|food[- ])?safety\b",
        r"\bhealth(?:care)?[- ]regulat(?:or|ion)\b",
        r"\bhealth(?:care)?[- ]product claims?\b",
        r"\bmedical|medicine|disease|vaccine|clinical|treatment|drug\b",
        r"\b(?:virus|viruses|viral|antiviral|pathogen|infection|infectious|"
        r"immune|immunity|neuron|neuronal|gene|genes|protein|proteins|enzyme|"
        r"enzymes|mutation|mutations|transplant|macrophage|basophil|antibody|"
        r"antibodies|fibrosis|inflammation|cancer|tumou?r|pharmacological)\b",
        r"\bdietary supplements?\b",
    ),
    "Politics": (
        r"\bopposition party\b",
        r"\bpolitical parties?\b",
        r"\belection|parliament|protest|legislation|political\b",
        r"\bpresident|prime minister\b",
    ),
    "Finance": (
        r"\b(?:repo|policy|interest) rates?\b",
        r"\bfinancial markets?|stock markets?|investments?|currency|currencies\b",
        r"\binflation|taxation|financial performance|financial results|profits?\b",
    ),
    "Science": (
        r"\bscientific (?:findings?|theor(?:y|ies)|research results?)\b",
        r"\bexperiment(?:s|al)?\b",
        r"\b(?:unknown|new|novel) particle\b",
        r"\bscientific|physics|chemistry|biology|astronomy\b",
    ),
}

_DOMAIN_REASON_SIGNALS = {
    "Medical": (
        r"\bhealth|medical|medicine|disease|vaccine|clinical|treatment|drug|safety\b",
    ),
    "Politics": (
        r"\bpolitic|election|party|government|official|appointment|protest|"
        r"legislation|parliament|president|minister\b",
    ),
    "Finance": (
        r"\bfinanc|money|market|bank|tax|investment|currency|rate|inflation|"
        r"revenue|profit|economic\b",
    ),
    "Science": (
        r"\bscientific|science|research|experiment|finding|theory|physics|"
        r"chemistry|biology|astronomy\b",
    ),
}

_REASON_STOP_WORDS = {
    "about", "after", "against", "and", "because", "by", "claim", "clearly",
    "domain", "from", "in", "involves", "is", "it", "of", "on", "relates",
    "the", "this", "to", "with",
}


def _strong_domain_signal(claim: str) -> str | None:
    """Return a domain only when a high-specificity signal is exclusive."""
    matching_domains = []
    for domain in SUPPORTED_DOMAINS:
        if any(re.search(pattern, claim, flags=re.IGNORECASE) for pattern in _STRONG_DOMAIN_SIGNALS[domain]):
            matching_domains.append(domain)
    return matching_domains[0] if len(matching_domains) == 1 else None


def _reason_supports_claim(claim: str, reason: str, domain: str) -> bool:
    """Require both domain-specific explanation and a lexical link to the claim."""
    if not any(
        re.search(pattern, reason, flags=re.IGNORECASE)
        for pattern in _DOMAIN_REASON_SIGNALS[domain]
    ):
        return False

    claim_terms = {
        term.casefold()
        for term in re.findall(r"[^\W_]+", claim)
        if len(term) > 2 and term.casefold() not in _REASON_STOP_WORDS
    }
    reason_terms = {
        term.casefold()
        for term in re.findall(r"[^\W_]+", reason)
        if len(term) > 2 and term.casefold() not in _REASON_STOP_WORDS
    }
    return bool(claim_terms & reason_terms)


def _domain_reason(domain: str) -> str:
    return {
        "Medical": "The claim concerns health, healthcare, or health-related safety.",
        "Politics": "The claim concerns political activity or government legislation.",
        "Finance": "The claim concerns a financial measure, market, or economic indicator.",
        "Science": "The claim concerns a scientific finding or experiment.",
    }[domain]


class DomainDetectionAgent:
    """
    Domain Detection Agent

    Responsibilities:
    - Receive a claim
    - Generate prompt
    - Send prompt to the configured Ollama model
    - Parse response
    - Validate response
    - Return structured JSON
    """

    def __init__(self, model_name: str = "qwen2.5:3b"):
        self.model_name = model_name

    def build_prompt(self, claim: str) -> str:
        """
        Replace the placeholder in the prompt with the actual claim.
        """
        return DOMAIN_PROMPT.format(claim=claim)

    def call_llm(self, prompt: str) -> str:
        """
        Sends the prompt to Ollama and returns the raw response.
        """

        logger.info("Sending prompt to configured Ollama model...")

        response = _LLM_RESILIENCE.chat(lambda: ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        ))

        llm_output = response["message"]["content"]

        print("\n========== RAW LLM RESPONSE ==========")
        print(repr(llm_output))
        print("======================================\n")

        return llm_output

    def detect_domain(self, claim: str) -> dict:
        """
        Main function for domain detection.
        """

        try:

            # Step 1
            prompt = self.build_prompt(claim)

            # Step 2
            llm_response = self.call_llm(prompt)

            # Step 3
            parsed_response = parse_response(llm_response)

            signal_domain = _strong_domain_signal(claim)
            valid_response = validate_response(parsed_response)
            if valid_response and parsed_response["domain"] in SUPPORTED_DOMAINS:
                model_domain = parsed_response["domain"]
                if signal_domain and signal_domain != model_domain:
                    return {
                        "domain": signal_domain,
                        "confidence": min(parsed_response["confidence"], 0.75),
                        "reason": _domain_reason(signal_domain),
                    }
                if _reason_supports_claim(
                    claim, parsed_response["reason"], model_domain
                ):
                    return parsed_response

            if signal_domain:
                confidence = 0.65
                if (
                    valid_response
                    and parsed_response.get("domain") == signal_domain
                ):
                    confidence = min(parsed_response["confidence"], 0.75)
                return {
                    "domain": signal_domain,
                    "confidence": confidence,
                    "reason": _domain_reason(signal_domain),
                }

            logger.warning(
                "Domain response was unsupported, incoherent, or not claim-related; "
                "no unique recovery signal was found."
            )
            return {
                "domain": "Unknown",
                "confidence": 0.0,
                "reason": "The claim is ambiguous or lacks a clear supported-domain subject.",
            }

        except Exception as e:

            logger.exception(e)

            return {
                "domain": "Unknown",
                "confidence": 0.0,
                "reason": str(e)
            }
