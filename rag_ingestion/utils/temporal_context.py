"""Extract conservative, user-facing temporal context without making claims."""

import re
from typing import Any, Dict, Iterable


_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_MONTH_YEAR = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
_FULL_DATE = re.compile(
    r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}\b"
)
_RELATIVE = re.compile(
    r"\b(?:yesterday|today|tomorrow|on\s+)?(?:last|this|next)?\s*"
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|week|month|year)\b",
    re.IGNORECASE,
)


def _claim_timeframe(claim: str) -> str | None:
    for pattern in (_FULL_DATE, _MONTH_YEAR, _RELATIVE, _YEAR):
        match = pattern.search(claim or "")
        if match:
            return match.group(0)
    return None


def build_temporal_context(claim: str, evidence: Iterable[Any]) -> Dict[str, Any]:
    """Return only temporal facts explicitly present in claim/evidence metadata."""
    claim_timeframe = _claim_timeframe(claim)
    evidence_timeframes = []
    for item in evidence or []:
        value = getattr(item, "publication_date", None)
        if value is None and isinstance(item, dict):
            value = item.get("publication_date")
        if value is not None:
            rendered = value.isoformat() if hasattr(value, "isoformat") else str(value)
            year_match = _YEAR.search(rendered)
            rendered = year_match.group(0) if year_match else rendered
            if rendered not in evidence_timeframes:
                evidence_timeframes.append(rendered)

    result: Dict[str, Any] = {
        "claim_timeframe": claim_timeframe or "Not specified",
        "has_explicit_timeframe": bool(claim_timeframe and (
            _FULL_DATE.search(claim_timeframe) or _MONTH_YEAR.search(claim_timeframe) or _YEAR.search(claim_timeframe)
        )),
        "evidence_timeframes": evidence_timeframes,
    }
    if claim_timeframe and evidence_timeframes:
        claim_year = _YEAR.search(claim_timeframe)
        evidence_years = {item for item in evidence_timeframes if _YEAR.fullmatch(item)}
        if claim_year and evidence_years and claim_year.group(0) not in evidence_years:
            result["temporal_note"] = (
                "Evidence includes different publication years; those sources may describe "
                "different events and were not treated as contradictions solely for that reason."
            )
    return result
