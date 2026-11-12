import logging
import re
import time
from collections import Counter
from decimal import Decimal, InvalidOperation

from .services.llm_service import LLMService
from rag_ingestion.llm_resilience import LLMInvocationError, remaining_request_budget, request_deadline


_NUMBER_RE = re.compile(
    r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"\s*(?:%|percent(?:age)?|per\s+cent|basis\s+points?|bps|"
    r"thousand|million|billion|trillion|dozen|hundred)?",
    re.IGNORECASE,
)
_MONTHS = (
    "January|February|March|April|May|June|July|August|September|"
    "October|November|December"
)
_WEEKDAYS = (
    "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
)
_DATE_RE = re.compile(
    rf"\b(?:"
    rf"\d{{4}}[-/]\d{{1,2}}[-/]\d{{1,2}}|"
    rf"\d{{1,2}}[-/]\d{{1,2}}[-/]\d{{2,4}}|"
    rf"(?:{_MONTHS})\s+\d{{1,2}}(?:,?\s+\d{{4}})?|"
    rf"\d{{1,2}}\s+(?:{_MONTHS})(?:\s+\d{{4}})?|"
    # A bare 4-digit token is a year only when it is outside a larger numeric
    # value (for example, a percentage or quantity).
    rf"(?<!\d)\d{{4}}(?!\d)|"
    rf"(?:this|last|next|previous|past|coming)\s+"
    rf"(?:day|week|month|year|quarter|decade|morning|afternoon|evening|night)|"
    rf"(?:today|yesterday|tomorrow|recently|currently|formerly|"
    rf"presently|now|soon|earlier|later|annually|yearly)|"
    rf"(?:on\s+)?(?:{_WEEKDAYS})"
    rf")\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(?:not|never|no|without|neither|nor|cannot|can't|"
    r"don't|doesn't|didn't|isn't|aren't|wasn't|weren't|"
    r"haven't|hasn't|hadn't|won't|wouldn't|shouldn't|couldn't)\b",
    re.IGNORECASE,
)
_DIRECTION_PATTERNS = {
    "increase": r"\b(?:increase\w*|rais\w*|hik\w*|ris\w*|rose|"
                r"higher|grew|grown|grow)\b",
    "decrease": r"\b(?:decrease\w*|reduc\w*|cut|cuts|cutting|"
                r"lower\w*|fall|falls|fell|fallen|drop\w*|"
                r"declin\w*)\b",
    "before": r"\bbefore\b",
    "after": r"\bafter\b",
    "more": r"\bmore\b",
    "less": r"\bless\b",
}
_NUMERIC_RELATION_RE = re.compile(
    r"\b(?P<relation>more\s+than|less\s+than|up\s+to|by|to|from|at|"
    r"above|below|over|under)\s*$",
    re.IGNORECASE,
)
_ENTITY_RE = re.compile(
    r"\b[A-Z][a-z]+(?:[’'][A-Za-z]+)?"
    r"(?:\s+(?:(?:of|the|and|for)|[A-Z][a-z]+(?:[’'][A-Za-z]+)?)){0,5}\b"
)
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,}\b")
_ENTITY_LEADING_STOPWORDS = {"a", "an", "the"}
CLAIM_EXTRACTION_BUDGET_SECONDS = 25.0
logger = logging.getLogger(__name__)
_ENTITY_STOPWORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}


def _normalized_number(match):
    value = match.group(0).strip()
    unit_match = re.search(
        r"(?:%|percent(?:age)?|per\s+cent|basis\s+points?|bps|"
        r"thousand|million|billion|trillion|dozen|hundred)$",
        value,
        re.IGNORECASE,
    )
    unit = re.sub(r"\s+", " ", unit_match.group(0).lower()) if unit_match else ""
    if unit in {"%", "percent", "percentage", "per cent"}:
        unit = "percent"
    elif unit in {"basis point", "basis points", "bps"}:
        unit = "basis points"
    number_text = value[:unit_match.start()].strip() if unit_match else value
    try:
        number = Decimal(number_text.replace(",", "")).normalize()
    except InvalidOperation:
        number = number_text.replace(",", "")
    return str(number), unit


def _numeric_facts(text):
    facts = set()
    for match in _NUMBER_RE.finditer(text):
        value = _normalized_number(match)
        prefix = text[max(0, match.start() - 30):match.start()]
        relation = _NUMERIC_RELATION_RE.search(prefix)
        facts.add((value, relation.group("relation").lower() if relation else ""))
    return facts


def _temporal_facts(text):
    facts = set()
    for match in _DATE_RE.finditer(text):
        value = re.sub(r"\s+", " ", match.group(0).strip().lower())
        value = re.sub(r"^(?:on|in|at)\s+", "", value)
        value = value.replace(",", "")
        facts.add(value)
    return facts


def _named_entities(text):
    entities = set()
    for match in _ENTITY_RE.finditer(text):
        words = match.group(0).split()
        while words and words[0].lower() in _ENTITY_LEADING_STOPWORDS:
            words.pop(0)
        while words and words[-1].lower() in {"of", "the", "and", "for"}:
            words.pop()
        if not words:
            continue
        normalized = " ".join(words).lower()
        if normalized in _ENTITY_STOPWORDS:
            continue
        preceding_text = text[:match.start()]
        at_sentence_start = (
            not preceding_text.strip(" \t\r\n\"'([{")
            or preceding_text.rstrip().endswith((".", "!", "?"))
        )
        if len(words) > 1 or not at_sentence_start:
            entities.add(normalized)
    entities.update(match.group(0).lower() for match in _ACRONYM_RE.finditer(text))
    return entities


def _claim_anchor(source, candidate):
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", source) if sentence.strip()]
    if len(sentences) < 2:
        return source

    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "in", "is", "it", "of", "on", "or", "the", "to", "was", "were",
        "with",
    }

    def content_words(text):
        return {
            word.lower()
            for word in re.findall(r"\b[\w.%-]+\b", text)
            if word.lower() not in stopwords
        }

    candidate_words = content_words(candidate)
    if not candidate_words:
        return source
    return max(
        sentences,
        key=lambda sentence: len(candidate_words & content_words(sentence))
        / len(candidate_words),
    )


def validate_claim_fidelity(source, candidate):
    """Return material source/candidate mismatches that make the claim unsafe."""
    violations = []
    if not candidate.strip():
        return ["generated claim is empty"]

    anchor = _claim_anchor(source, candidate)
    source_numbers = _numeric_facts(anchor)
    candidate_numbers = _numeric_facts(candidate)
    if not source_numbers.issubset(candidate_numbers) or not candidate_numbers.issubset(_numeric_facts(source)):
        violations.append("numeric values, units, or numeric relations changed")

    source_temporal = _temporal_facts(anchor)
    candidate_temporal = _temporal_facts(candidate)
    if not source_temporal.issubset(candidate_temporal) or not candidate_temporal.issubset(_temporal_facts(source)):
        violations.append("dates or explicit temporal expressions changed")
    source_negation = bool(_NEGATION_RE.search(anchor))
    candidate_negation = bool(_NEGATION_RE.search(candidate))
    if source_negation != candidate_negation:
        violations.append("negation changed")

    source_directions = {
        direction
        for direction, pattern in _DIRECTION_PATTERNS.items()
        if re.search(pattern, anchor, re.IGNORECASE)
    }
    candidate_directions = {
        direction
        for direction, pattern in _DIRECTION_PATTERNS.items()
        if re.search(pattern, candidate, re.IGNORECASE)
    }
    all_source_directions = {
        direction for direction, pattern in _DIRECTION_PATTERNS.items()
        if re.search(pattern, source, re.IGNORECASE)
    }
    if (source_directions and not source_directions.issubset(candidate_directions)) or not candidate_directions.issubset(all_source_directions):
        violations.append("directional or comparative polarity changed")

    source_entities = _named_entities(source)
    anchor_entities = _named_entities(anchor)
    candidate_entities = _named_entities(candidate)
    if not anchor_entities.issubset(candidate_entities) or not candidate_entities.issubset(source_entities):
        violations.append("named entities changed")
    return violations


class ClaimExtractionFidelityError(ValueError):
    """Raised when generated claims cannot pass factual-fidelity validation."""


def _concise_source_fallback(source):
    """Choose a short source sentence that captures the most repeated content."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", source.strip())
        if sentence.strip()
    ]
    if not sentences:
        return ""
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "in", "is", "it", "of", "on", "or", "the", "to", "was", "were",
        "with",
    }
    words_by_sentence = [
        {word.lower() for word in re.findall(r"\b[\w.%-]+\b", sentence)
         if word.lower() not in stopwords}
        for sentence in sentences
    ]
    frequencies = Counter(word for words in words_by_sentence for word in words)
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (
            len(_named_entities(item[1])),
            sum(frequencies[word] - 1 for word in words_by_sentence[item[0]]),
            -item[0],
        ),
        reverse=True,
    )
    for _, sentence in ranked:
        if len(sentence.split()) <= 30 and not validate_claim_fidelity(source, sentence):
            return sentence
    return ""


class ClaimExtractionAgent:

    def __init__(self):
        self.llm = LLMService()

    def extract_claim(self, text):
        total_started = time.monotonic()
        if not text or not text.strip():
            return ""

        source = text.strip()
        concise_structured = (
            len(source.split()) <= 30
            and len(re.findall(r"[.!?]", source)) <= 1
            and not re.search(
                r"\b(?:and|but|while|whereas|including|respectively)\b|[;:]",
                source, re.IGNORECASE,
            )
            and not re.match(r"^(?:who|what|when|where|why|how|is it true)\b", source, re.I)
        )
        if concise_structured:
            logger.info(
                "claim_extraction complete duration_ms=%.1f model_calls=0 fallback=source_passthrough",
                (time.monotonic() - total_started) * 1000,
            )
            return source

        preprocessing_started = time.monotonic()
        text = source

        prompt = f"""
Extract the main factual claim from the following text.

Keep it concise, but preserve every explicitly named person, organization,
product, and location, including every item in a list introduced by wording
such as "including" or by punctuation. Never replace named list items with a
generic group description. Preserve relevant numbers, dates, time expressions,
actions, and qualifiers even when that makes the claim longer.
Preserve explicit temporal qualifiers that materially affect the claim,
including relative expressions such as "recently", "yesterday", "last week",
and "this year", and specific time references such as "in 2025" or "on Friday".
Do not omit, weaken, or generalize explicit time information to make the claim
shorter; include it in the concise factual claim.
Never change any number, percentage, date, year, quantity, named entity, or
explicit temporal qualifier, and never introduce factual information absent
from the input. Preserve negation and factual polarity. Preserve directional
and comparative meaning, including increase/decrease, rise/fall, before/after,
more/less, and whether a quantity changed "by" a value or changed "to" a value.
A concise paraphrase is allowed only when all underlying factual values and
relations remain unchanged.

TEXT:
{text}

Return only one concise factual claim.
"""

        preprocessing_ms = (time.monotonic() - preprocessing_started) * 1000
        remaining = remaining_request_budget()
        extraction_deadline = total_started + CLAIM_EXTRACTION_BUDGET_SECONDS
        if remaining is not None:
            extraction_deadline = min(extraction_deadline, time.monotonic() + remaining)
        model_calls = 0
        repairs = 0
        repair_duration_ms = 0.0
        fallback_used = False

        def deterministic_fallback(reason):
            nonlocal fallback_used
            fallback_used = True
            fallback_started = time.monotonic()
            candidate = self.clean_claim(_concise_source_fallback(text))
            if candidate and len(candidate.split()) <= 30 and not validate_claim_fidelity(text, candidate):
                logger.info(
                    "claim_extraction deterministic_fallback reason=%s duration_ms=%.1f",
                    reason, (time.monotonic() - fallback_started) * 1000,
                )
                return candidate
            logger.warning(
                "claim_extraction deterministic_fallback_unavailable reason=%s duration_ms=%.1f",
                reason, (time.monotonic() - fallback_started) * 1000,
            )
            raise ClaimExtractionFidelityError(
                "Claim extraction failed and no concise source-grounded fallback was available."
            )

        try:
            with request_deadline(extraction_deadline):
                logger.info("claim_extraction preprocessing duration_ms=%.1f", preprocessing_ms)
                generation_started = time.monotonic()
                try:
                    model_calls += 1
                    claim = self.clean_claim(self.llm.generate(prompt))
                except Exception as exc:
                    reason = "timeout" if isinstance(exc, LLMInvocationError) and exc.kind == "timeout" else "model_failure"
                    return deterministic_fallback(reason)
                logger.info(
                    "claim_extraction generation duration_ms=%.1f model_calls=%d",
                    (time.monotonic() - generation_started) * 1000, model_calls,
                )

                if len(claim.split()) > 30:
                    claim = deterministic_fallback("output_exceeded_concise_limit")
                violations = validate_claim_fidelity(text, claim)
                if violations:
                    correction_prompt = f"""Correct the generated claim using ONLY the original input.
The generated claim failed deterministic factual-fidelity validation.
Detected fidelity violations: {", ".join(violations)}
Do not invent or alter any number, percentage, date, year, quantity, named
entity, temporal qualifier, negation, polarity, comparison, or numeric
direction (including "by" versus "to"). A paraphrase is allowed only if all
facts and relations are unchanged. Return one concise claim, or reproduce the
original input's main factual claim verbatim if needed for accuracy.

ORIGINAL INPUT:
{text}

INVALID GENERATED CLAIM:
{claim}
"""
                    repair_started = time.monotonic()
                    try:
                        repairs += 1
                        model_calls += 1
                        claim = self.clean_claim(self.llm.generate(correction_prompt))
                    except Exception:
                        repair_duration_ms = (time.monotonic() - repair_started) * 1000
                        return deterministic_fallback("fidelity_repair_failed")
                    repair_duration_ms = (time.monotonic() - repair_started) * 1000
                    violations = validate_claim_fidelity(text, claim)
                    if violations:
                        return deterministic_fallback("fidelity_repair_invalid")

                return claim
        finally:
            logger.info(
                "claim_extraction complete duration_ms=%.1f preprocessing_ms=%.1f "
                "repair_ms=%.1f repairs=%d model_calls=%d retries=0 fallback=%s",
                (time.monotonic() - total_started) * 1000,
                preprocessing_ms,
                repair_duration_ms,
                repairs,
                model_calls,
                fallback_used,
            )

    def clean_claim(self, claim):

        if not claim:
            return ""

        claim = claim.strip()

        prefixes = [
            "Claim:",
            "CLAIM:",
            "Main Claim:",
            "Main claim:",
            "Extracted Claim:",
            "Extracted claim:",
            "Output:",
            "OUTPUT:"
        ]

        for prefix in prefixes:
            if claim.startswith(prefix):
                claim = claim[len(prefix):].strip()

        # Remove quotation marks
        claim = claim.strip('"').strip("'")

        # Remove accidental numbering
        if claim.startswith("1."):
            claim = claim[2:].strip()

        # Remove extra spaces/newlines
        claim = " ".join(claim.split())

        return claim.translate(str.maketrans({"‘": "'", "’": "'"}))

    def extract_claims(self, text):

        if not text or not text.strip():
            return []

        claim = self.extract_claim(text)

        if not claim:
            return []

        return [claim]
