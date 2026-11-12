"""Deterministic guards against metadata and boilerplate being treated as evidence."""

import re


_BOILERPLATE = re.compile(
    r"\b(?:skip to (?:main )?content|subscribe|log ?in|log ?out|sign in|"
    r"accept cookies|cookie preferences|privacy policy|terms of use|"
    r"all rights reserved|site navigation|main menu|search this site)\b",
    re.IGNORECASE,
)
_BIBLIOGRAPHIC_ONLY = re.compile(
    r"\b(?:bibliographic (?:record|metadata) only|metadata only|"
    r"no (?:abstract|article body|full text) available|"
    r"no abstract or bibliographic metadata available)\b",
    re.IGNORECASE,
)


def has_substantive_text(text: str) -> bool:
    """Reject obvious boilerplate-only and bibliographic placeholder text.

    This is an additional quality gate; it does not replace claim relevance,
    source checks, or existing minimum-length rules.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    compact = " ".join(text.split())
    if _BIBLIOGRAPHIC_ONLY.search(compact):
        return False
    boilerplate = _BOILERPLATE.findall(compact)
    substantive = _BOILERPLATE.sub(" ", compact)
    words = re.findall(r"\b[\w'-]+\b", substantive)
    if len(words) < 8:
        return False
    boilerplate_chars = sum(len(item) for item in boilerplate)
    return boilerplate_chars / max(len(compact), 1) < 0.35
