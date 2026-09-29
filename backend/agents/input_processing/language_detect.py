"""
language_detect.py
==================
RAVEN Project - Language Detection Module

Responsibility:
    Receives extracted text (from the OCR module or user input) and identifies
    whether the language is English ("en"), Telugu ("te"), or Mixed ("mixed").

This module does NOT perform translation, cleaning for NLP,
normalization, or any form of reasoning. Its sole job is
language identification.

Dependencies:
    pip install langdetect

Usage (from other modules):
    from agents.input_processing.language_detect import detect_language
    lang = detect_language("some extracted text")
    # Returns "en", "te", or "mixed"
"""

import logging
import re
from langdetect import detect, LangDetectException, DetectorFactory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deterministic seed - MUST be set before any detection call.
# Without this, langdetect can return different results on every run
# because it uses a random sampling approach internally.
# Setting seed = 0 locks the randomness so results are reproducible.
# ---------------------------------------------------------------------------
DetectorFactory.seed = 0

# ---------------------------------------------------------------------------
# Constants - the only three languages this module is designed to return.
# ---------------------------------------------------------------------------
LANG_ENGLISH = "en"   # ISO 639-1 code for English
LANG_TELUGU = "te"    # ISO 639-1 code for Telugu
LANG_MIXED = "mixed"  # Custom code for Code-Mixed English+Telugu
DEFAULT_LANG = LANG_ENGLISH  # Fallback when detection fails or is uncertain


def _preprocess_text(text: str) -> str:
    """
    Clean raw text before passing it to the language detector.

    Steps:
        1. Guard against None input.
        2. Strip leading/trailing whitespace.
        3. Replace newlines with spaces (OCR output often has line breaks
           that can confuse detection).
        4. Collapse multiple spaces into one.
    """
    # Step 1 - handle None safely
    if text is None:
        return ""

    # Step 2 - remove leading/trailing whitespace
    text = text.strip()

    # Step 3 - flatten newlines into spaces
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")

    # Step 4 - collapse multiple whitespace characters into one
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _contains_telugu(text: str) -> bool:
    """
    Checks if the text contains any Telugu characters.
    Uses the Unicode block for Telugu: U+0C00 to U+0C7F.
    """
    return bool(re.search(r'[\u0C00-\u0C7F]', text))


def _contains_english(text: str) -> bool:
    """
    Checks if the text contains any English (Latin) words/letters.
    """
    return bool(re.search(r'[a-zA-Z]+', text))


def detect_language(text: str) -> str:
    """
    Detect whether the input text is English, Telugu, or Mixed.

    This function prioritizes script detection (Unicode ranges) because
    langdetect often misclassifies code-mixed sentences.
    
    Logic priority:
        1. Both Telugu and English present -> "mixed"
        2. Only Telugu present -> "te"
        3. Only English present -> "en"
        4. Neither (e.g. symbols/numbers only) -> langdetect fallback
        5. Any exception -> "en"

    Args:
        text: Extracted text (may be messy, None, or empty).

    Returns:
        "en", "te", or "mixed"
    """
    # Preprocess text to handle None, whitespace, newlines
    cleaned_text = _preprocess_text(text)

    if not cleaned_text:
        return DEFAULT_LANG

    has_telugu = _contains_telugu(cleaned_text)
    has_english = _contains_english(cleaned_text)

    # 1. If BOTH Telugu script and English words exist
    if has_telugu and has_english:
        return LANG_MIXED
    
    # 2. If ONLY Telugu script exists
    if has_telugu:
        return LANG_TELUGU
    
    # 3. If ONLY English words exist
    if has_english:
        return LANG_ENGLISH
    
    # 4. Otherwise, use langdetect() fallback
    try:
        detected = detect(cleaned_text)
        if detected == LANG_TELUGU:
            return LANG_TELUGU
        return LANG_ENGLISH
        
    except LangDetectException:
        # 5. Fallback for exceptions
        return DEFAULT_LANG
    except Exception:
        return DEFAULT_LANG


# ---------------------------------------------------------------------------
# Test suite - runs only when this file is executed directly,
# not when imported by other RAVEN modules.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        ("English sentence", "The RBI reduced the repo rate."),
        ("Pure Telugu sentence", "తెలంగాణ ప్రభుత్వం కొత్త పథకాన్ని ప్రకటించింది"),
        ("Mixed Telugu + English", "PM Modi హైదరాబాద్లో ప్రసంగించారు."),
        ("OCR English text", "BREAKING NEWS"),
        ("OCR Telugu text", "కోవిడ్ వ్యాక్సిన్ సురక్షితం"),
        ("Mixed OCR text", "BREAKING NEWS తెలంగాణ"),
        ("Very short English word", "Hi"),
        ("Very short Telugu word", "హాయ్"),
        ("Empty string", ""),
        ("Numbers only", "123456789"),
        ("Random OCR garbage", "@@##%%^^&&"),
    ]

    print("=" * 60)
    print("  RAVEN - Language Detection Module - Test Suite")
    print("=" * 60)

    for description, sample_text in test_cases:
        result = detect_language(sample_text)
        display_text = repr(sample_text[:50]) + ("..." if len(sample_text) > 50 else "")
        print(f"\n  Test    : {description}")
        print(f"  Input   : {display_text}")
        print(f"  Result  : {result}")

    print("\n" + "=" * 60)
    print("  All test cases completed.")
    print("=" * 60)
