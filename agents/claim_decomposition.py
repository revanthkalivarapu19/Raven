"""Bounded, validated decomposition of compound user claims."""

from __future__ import annotations

import json
import re
from typing import Callable, Optional

from pydantic import BaseModel, Field, ValidationError


class AtomicClaim(BaseModel):
    id: str
    text: str = Field(min_length=4, max_length=1200)
    entities: list[str] = Field(default_factory=list)
    domain: Optional[str] = None
    temporal_context: dict[str, list[str]] = Field(default_factory=dict)
    search_queries: list[str] = Field(default_factory=list)


class DecompositionResult(BaseModel):
    claims: list[AtomicClaim] = Field(min_length=1, max_length=5)
    used_fallback: bool = False
    failure_category: Optional[str] = None


_TIME = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s+\d{4}\b|"
    r"\b(?:today|yesterday|tomorrow|recently|currently|last\s+\w+|"
    r"this\s+year|by\s+[A-Z][a-z]+\s+\d{4})\b", re.IGNORECASE,
)
_ENTITY = re.compile(r"\b[A-Z][\w&.'’-]*(?:\s+[A-Z][\w&.'’-]*){0,4}|\b[A-Z]{2,}\b")
_NUMBER = re.compile(r"(?:₹|\$|€|£)?\s*\d[\d,.]*(?:\s*(?:%|percent|million|billion|thousand))?", re.I)
_STOP = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "is", "it", "of", "on", "or", "the", "to", "was", "were", "with", "while"}


def is_likely_compound(text: str) -> bool:
    """Avoid an LLM call for a concise single proposition."""
    words = re.findall(r"\b\w+\b", text)
    conjunctions = len(re.findall(r"\b(?:and|while|whereas|but)\b", text, re.I))
    return bool(
        ";" in text
        or ("," in text and conjunctions >= 1 and len(words) >= 8)
        or re.search(r"\bwhile\b|\bwhereas\b", text, re.I)
        or (conjunctions >= 1 and len(words) >= 10)
        or ("(" in text and ")" in text and conjunctions >= 1)
        or (conjunctions >= 1 and len(words) > 32)
    )


def _entities(text: str) -> list[str]:
    found = []
    for match in _ENTITY.finditer(text):
        value = match.group(0).strip(" .,'’-")
        if value.casefold() not in {"the", "a", "an", "while", "but"} and value not in found:
            found.append(value)
    return found


def _grounded(source: str, candidate: str) -> bool:
    source_fold = source.casefold()
    if any(value.casefold() not in source_fold for value in _entities(candidate)):
        return False
    source_numbers = {re.sub(r"[\s,.]+", "", n).casefold() for n in _NUMBER.findall(source)}
    candidate_numbers = {re.sub(r"[\s,.]+", "", n).casefold() for n in _NUMBER.findall(candidate)}
    if not candidate_numbers.issubset(source_numbers):
        return False
    source_tokens = set(re.findall(r"[\w₹$€£]+", source_fold))
    candidate_tokens = {t for t in re.findall(r"[\w₹$€£]+", candidate.casefold()) if t not in _STOP}
    if not candidate_tokens:
        return False
    overlap = sum(
        token in source_tokens or any(
            len(token) >= 5 and (token.startswith(src) or src.startswith(token))
            for src in source_tokens
        )
        for token in candidate_tokens
    ) / len(candidate_tokens)
    return overlap >= 0.72


def _make_claims(texts: list[str]) -> list[AtomicClaim]:
    claims = []
    for index, text in enumerate(texts, start=1):
        cleaned = " ".join(text.split()).strip(" -\t\r\n")
        temporal = _TIME.findall(cleaned)
        claims.append(AtomicClaim(
            id=f"C{index}", text=cleaned, entities=_entities(cleaned),
            temporal_context={"expressions": temporal},
            search_queries=[cleaned],
        ))
    return claims


def _split_clear_conjunctions(source: str) -> list[str] | None:
    """Split clear clause punctuation; leave ambiguous conjunctions to the model."""
    text = " ".join(source.split()).strip()
    contrasted = [part.strip(" ,;") for part in re.split(r",\s*(?:while|whereas)\s+", text, flags=re.I)]
    if len(contrasted) == 2 and all(len(re.findall(r"\b\w+\b", part)) >= 4 for part in contrasted):
        return contrasted
    parts = [part.strip(" ,;") for part in re.split(
        r";\s*|,\s+(?:(?i:and\s+)?(?=[A-Z][\w&.'’-]*\s+\w+))|,\s*(?i:while|whereas)\s+",
        text,
    )]
    if 2 <= len(parts) <= 5 and all(len(re.findall(r"\b\w+\b", part)) >= 4 for part in parts):
        return parts
    return None


def decompose_claim(
    source: str,
    *,
    fallback_claim: str,
    generate: Optional[Callable[[str], str]] = None,
) -> DecompositionResult:
    """Decompose only likely compounds; malformed/unsafe output falls back."""
    fallback = " ".join((fallback_claim or source).split()).strip()
    deterministic_parts = _split_clear_conjunctions(source)
    if deterministic_parts:
        return DecompositionResult(claims=_make_claims(deterministic_parts))
    if not is_likely_compound(source) or generate is None:
        return DecompositionResult(claims=_make_claims([fallback]))
    prompt = f"""Identify the distinct factual propositions explicitly stated in this input.
Split only when each part can be checked independently. Keep each proposition
grounded in the input: preserve names, amounts, negation, dates, and relationships.
Do not add context or infer a fact. Return JSON only, with 2 to 5 strings in
the `claims` array. If the text is not actually compound, return one string.

INPUT:
{source}

JSON schema: {{"claims": ["atomic proposition"]}}"""
    try:
        raw = generate(prompt)
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("claims"), list):
            raise ValueError("invalid decomposition shape")
        values = parsed["claims"]
        if not 2 <= len(values) <= 5 or any(not isinstance(value, str) for value in values):
            raise ValueError("invalid decomposition count or item type")
        if any(not _grounded(source, value) for value in values):
            raise ValueError("decomposition contains ungrounded content")
        claims = _make_claims(values)
        if len({claim.text.casefold() for claim in claims}) != len(claims):
            raise ValueError("decomposition contains duplicate claims")
        return DecompositionResult(claims=claims)
    except (Exception, ValidationError) as exc:
        category = "malformed_output" if isinstance(exc, (ValueError, json.JSONDecodeError, ValidationError)) else "model_failure"
        return DecompositionResult(
            claims=_make_claims([fallback]), used_fallback=True,
            failure_category=category,
        )
