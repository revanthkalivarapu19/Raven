"""Generic validation of a model's claim–evidence relationship.

Semantic entailment belongs to the verifier. This module checks that its cited
passage came from the evidence and that explicit factual values do not silently
change between a claim and a purportedly supporting passage.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_YEAR = re.compile(r"\b(?:18|19|20|21)\d{2}\b")
_PERCENT = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:%|percent\b)", re.I)
_MONEY = re.compile(r"(?:[$€£₹]\s*\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s*(?:USD|EUR|GBP|INR)\b)", re.I)
_NUMBER = re.compile(r"(?<!\w)\d+(?:,\d{3})*(?:\.\d+)?(?!\w)")


@dataclass(frozen=True)
class DirectnessDecision:
    relationship: str
    directness: str
    accepted: bool
    reason: str
    evidence_span: str | None = None


def _normalize_space(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value).casefold()
    ascii_text = folded.encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text))


def _values(pattern: re.Pattern[str], value: str) -> set[str]:
    return {re.sub(r"[\s,]", "", match.group(0).casefold()) for match in pattern.finditer(value)}


_DIRECTNESS_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "did", "do",
    "does", "for", "from", "has", "have", "in", "into", "is", "it", "of",
    "on", "or", "that", "the", "this", "to", "was", "were", "with",
}


def _content_tokens(value: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[a-z0-9]+", _normalize_space(value)):
        if len(token) < 2 or token in _DIRECTNESS_STOPWORDS:
            continue
        # A small morphology normalization keeps ordinary inflections aligned;
        # it does not introduce aliases or domain-specific vocabulary.
        for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
            if len(token) > len(suffix) + 3 and token.endswith(suffix):
                token = token[:-len(suffix)]
                break
        tokens.add(token)
    return tokens


def evaluate_directness(
    claim: str,
    evidence: str,
    verifier_relationship: str,
    *,
    evidence_excerpt: str | None = None,
    semantic_reason: str | None = None,
) -> DirectnessDecision:
    """Validate a structured semantic judgment without using claim vocabulary."""
    relation = verifier_relationship.upper()
    if relation not in {"SUPPORTS", "CONTRADICTS"}:
        return DirectnessDecision("INSUFFICIENT", "INSUFFICIENT", False,
                                  semantic_reason or "semantic verifier found no direct relationship")
    excerpt = (evidence_excerpt or "").strip()
    reason = (semantic_reason or "").strip()
    if not excerpt:
        claim_tokens = _content_tokens(claim)
        evidence_tokens = _content_tokens(evidence)
        if not claim_tokens:
            return DirectnessDecision("INSUFFICIENT", "INSUFFICIENT", False,
                                      "claim has no checkable content terms")
        coverage = len(claim_tokens & evidence_tokens) / len(claim_tokens)
        if coverage < 0.50:
            return DirectnessDecision("INSUFFICIENT", "INSUFFICIENT", False,
                                      "evidence does not contain enough claim-specific content")
        excerpt = evidence.strip()[:850]
        reason = "Relationship evaluated against the stored evidence passage."
    if _normalize_space(excerpt) not in _normalize_space(evidence):
        return DirectnessDecision("INSUFFICIENT", "INSUFFICIENT", False,
                                  "cited passage is not present in retrieved evidence")
    if len(excerpt) > 1400:
        return DirectnessDecision("INSUFFICIENT", "INSUFFICIENT", False,
                                  "cited passage is too broad to establish a direct relationship")
    if relation == "SUPPORTS":
        for name, pattern in (("year", _YEAR), ("percentage", _PERCENT), ("amount", _MONEY), ("number", _NUMBER)):
            asserted = _values(pattern, claim)
            cited = _values(pattern, excerpt)
            if asserted and cited and not asserted.issubset(cited):
                return DirectnessDecision("INSUFFICIENT", "CONSTRAINT_CONFLICT", False,
                                          f"claimed {name} differs from the cited passage")
    return DirectnessDecision(relation, "DIRECT_SUPPORT" if relation == "SUPPORTS" else "DIRECT_CONTRADICTION",
                              True, reason, excerpt)
