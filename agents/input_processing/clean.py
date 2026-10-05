"""
clean.py
========
RAVEN Project - Conservative Text Cleaning Module

Responsibility:
    Cleans OCR noise and formatting artifacts conservatively:
    - Strips unsafe control characters
    - Removes isolated OCR symbol noise
    - Detects and deduplicates consecutive repeated OCR lines/phrases
    - Preserves protected values (numbers, currencies, percentages, hashtags, handles, URLs)
    - Emits diagnostic quality flags without hallucinating corrections
"""

import logging
import re
import sys
from typing import Tuple, List

logger = logging.getLogger(__name__)

RE_CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+')


def _is_garbage_token(token: str) -> bool:
    """
    Determines if a token is clearly standalone OCR garbage.
    Must be conservative to avoid stripping legitimate abbreviations or symbols.
    """
    if not token:
        return False

    # 1. Any alphanumeric character (Unicode letters/numbers) -> keep
    if any(c.isalnum() for c in token):
        return False

    # 2. Keep social media prefixes (@handle, #hashtag) if attached to alphanumeric text
    if (token.startswith("@") or token.startswith("#")) and any(c.isalnum() for c in token):
        return False

    # 3. Known valid repeated punctuation
    valid_repeated = {"...", "..", "!!", "!!!", "??", "???", "--", "---", "==", "-", "+", "%", "$", "₹", "€"}
    if token in valid_repeated:
        return False

    # 4. Known OCR garbage characters
    garbage_chars = set("@#^~|\\_`")
    if any(c in garbage_chars for c in token):
        return True

    # 5. Length > 1 symbol noise (e.g., "**++", "<>")
    if len(token) > 1 and not all(c in ".,!?;:'\"-–—()[]{}" for c in token):
        return True

    return False


def deduplicate_consecutive_lines(text: str) -> Tuple[str, bool]:
    """
    Detects and eliminates identical or near-identical consecutive lines
    frequently produced by overlapping OCR bounding boxes.
    """
    lines = text.split("\n")
    if len(lines) <= 1:
        # Also check for repeated full sentences within a single line
        return text, False

    cleaned_lines = []
    seen_prev = None
    had_duplicates = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue

        # Compare normalized form of stripped line
        norm = re.sub(r'\s+', ' ', stripped).lower()
        if norm == seen_prev:
            had_duplicates = True
            continue

        seen_prev = norm
        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines), had_duplicates


def clean_content_with_flags(text: str) -> Tuple[str, List[str]]:
    """
    Cleans text and returns both the cleaned text and diagnostic flags.
    """
    if text is None:
        return "", []

    flags = []
    text = str(text)

    # 1. Check and remove unsafe control characters
    if RE_CONTROL_CHARS.search(text):
        flags.append("control_characters_removed")
        text = RE_CONTROL_CHARS.sub('', text)

    # 2. Conservative token-level noise removal
    try:
        parts = re.split(r'(\s+)', text)
        result_parts = []
        had_garbage_tokens = False

        for i, part in enumerate(parts):
            if i % 2 == 0:  # Token
                if part and _is_garbage_token(part):
                    had_garbage_tokens = True
                    continue
                result_parts.append(part)
            else:  # Whitespace
                result_parts.append(part)

        if had_garbage_tokens:
            flags.append("ocr_symbol_noise_cleaned")

        text = "".join(result_parts)
    except Exception as e:
        logger.exception(f"Error during token cleaning: {e}")

    # 3. Strip whitespace from lines and deduplicate consecutive repeated lines
    cleaned_lines = [re.sub(r'[ \t]+', ' ', l).strip() for l in text.split("\n")]
    deduped_lines = []
    seen_prev = None
    had_repeated = False
    for line in cleaned_lines:
        if not line:
            deduped_lines.append("")
            continue
        norm = line.lower()
        if norm == seen_prev:
            had_repeated = True
            continue
        seen_prev = norm
        deduped_lines.append(line)

    if had_repeated:
        flags.append("repeated_lines_removed")

    cleaned_text = "\n".join(deduped_lines).strip()
    return cleaned_text, flags


def clean_content(text: str) -> str:
    """
    Backward-compatible clean_content returning only the cleaned string.
    """
    cleaned, _ = clean_content_with_flags(text)
    return cleaned


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    test_cases = [
        "The government announced a new policy.",
        "@@## The government announced a new policy ^^",
        "AI and RBI announced new guidelines.",
        "COVID-19 vaccine reduces risk by 50%.",
        "RBI reduced the repo rate by 0.50%.",
        "The product costs ₹5000 or $100.",
        "GPT-4 and 5G technology were discussed.",
        "Meeting scheduled for 2026-08-09 at 10:30 PM.",
        "తెలంగాణ government announced a new policy.",
        "https://example.com/news",
        "#BreakingNews",
        "@PMOIndia announced the update.",
        "Line one of announcement\nLine one of announcement\nLine two of announcement",
        "@@## ^^ ~~ || !!!@@##",
        "",
        None
    ]

    print("=" * 65)
    print("  RAVEN - Conservative Clean Module Test Suite")
    print("=" * 65)

    for i, case in enumerate(test_cases, 1):
        cleaned, flags = clean_content_with_flags(case)
        print(f"Test {i}:")
        print(f"  Input : {repr(case)}")
        print(f"  Output: {repr(cleaned)}")
        if flags:
            print(f"  Flags : {flags}")
        print("-" * 65)
