"""
language_detect.py
==================
RAVEN Project - Enhanced Language Detection Module

Responsibility:
    Identifies whether text is English ("en"), Telugu ("te"), or Mixed ("mixed"),
    and computes a reliable confidence score.
    Supports short texts, OCR fragments, and segment-level language identification.

Dependencies:
    pip install langdetect
"""

import logging
import re
from typing import Tuple, Dict, Any
from langdetect import detect_langs, DetectorFactory

logger = logging.getLogger(__name__)

# Deterministic seed for reproducible langdetect results
DetectorFactory.seed = 0

LANG_ENGLISH = "en"
LANG_TELUGU = "te"
LANG_MIXED = "mixed"
DEFAULT_LANG = LANG_ENGLISH

# Regex patterns for scripts
RE_TELUGU = re.compile(r'[\u0C00-\u0C7F]')
RE_ENGLISH = re.compile(r'[a-zA-Z]')
RE_ALPHANUMERIC = re.compile(r'[\w]', re.UNICODE)


def _preprocess_text(text: str) -> str:
    """
    Clean text before language detection:
    - Safe handling of None
    - Whitespace normalization
    """
    if text is None:
        return ""
    text = str(text).strip()
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def count_script_characters(text: str) -> Tuple[int, int]:
    """
    Returns (telugu_char_count, english_char_count).
    """
    telugu_count = len(RE_TELUGU.findall(text))
    english_count = len(RE_ENGLISH.findall(text))
    return telugu_count, english_count


def detect_language_with_confidence(text: str) -> Tuple[str, float]:
    """
    Detect whether the input text is English, Telugu, or Mixed,
    along with a confidence estimate in [0.0, 1.0].

    Logic:
    1. Count Telugu and English characters.
    2. If both are present:
       - If both have significant presence (> 10% each of total alpha chars, or >= 3 chars each): "mixed"
       - Otherwise classified by dominant script with slightly lower confidence.
    3. If only Telugu is present: "te" with high confidence.
    4. If only English is present: "en" with high confidence.
    5. If neither (e.g., numbers, symbols, foreign Latin): use langdetect fallback.
    6. Empty/None text falls back to ("en", 0.0).
    """
    cleaned = _preprocess_text(text)
    if not cleaned:
        return DEFAULT_LANG, 0.0

    te_count, en_count = count_script_characters(cleaned)
    total_alpha = te_count + en_count

    # Case 1: Both scripts present
    if te_count > 0 and en_count > 0:
        te_ratio = te_count / total_alpha
        en_ratio = en_count / total_alpha

        # To be genuinely mixed, both scripts must have at least 10% representation
        # and at least 3 characters. Otherwise, classify by the dominant script.
        if te_ratio >= 0.10 and en_ratio >= 0.10 and te_count >= 3 and en_count >= 3:
            confidence = round(min(0.98, 0.70 + 0.30 * min(1.0, total_alpha / 15.0)), 3)
            return LANG_MIXED, confidence
        elif te_count > en_count:
            confidence = round(min(0.99, 0.75 + 0.25 * te_ratio), 3)
            return LANG_TELUGU, confidence
        else:
            confidence = round(min(0.99, 0.75 + 0.25 * en_ratio), 3)
            return LANG_ENGLISH, confidence

    # Case 2: Only Telugu characters present
    if te_count > 0:
        confidence = round(min(0.99, 0.75 + 0.25 * min(1.0, te_count / 10.0)), 3)
        return LANG_TELUGU, confidence

    # Case 3: Only English characters present
    if en_count > 0:
        confidence = round(min(0.99, 0.75 + 0.25 * min(1.0, en_count / 10.0)), 3)
        return LANG_ENGLISH, confidence

    # Case 4: Neither script detected (numbers, punctuation, symbols, URLs)
    # Check for URLs or domain names
    if "http" in cleaned.lower() or "www." in cleaned.lower() or ".com" in cleaned.lower():
        return LANG_ENGLISH, 0.90

    # Fallback to langdetect for romanized or other characters
    try:
        langs = detect_langs(cleaned)
        if langs:
            top_lang = langs[0]
            if top_lang.lang == "te":
                return LANG_TELUGU, round(float(top_lang.prob), 3)
            elif top_lang.lang == "en":
                return LANG_ENGLISH, round(float(top_lang.prob), 3)
            else:
                return LANG_ENGLISH, 0.50
    except Exception:
        pass

    return DEFAULT_LANG, 0.50


def detect_language(text: str) -> str:
    """
    Backwards-compatible convenience function returning only the language code.
    """
    lang, _ = detect_language_with_confidence(text)
    return lang


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    test_cases = [
        ("English sentence", "The RBI reduced the repo rate by 0.50%."),
        ("Pure Telugu sentence", "తెలంగాణ ప్రభుత్వం కొత్త పథకాన్ని ప్రకటించింది"),
        ("Mixed Telugu + English", "PM Modi హైదరాబాద్లో ప్రసంగించారు."),
        ("OCR English text", "BREAKING NEWS"),
        ("OCR Telugu text", "కోవిడ్ వ్యాక్సిన్ సురక్షితం"),
        ("Mixed OCR text", "BREAKING NEWS తెలంగాణ"),
        ("Very short English word", "Hi"),
        ("Very short Telugu word", "హాయ్"),
        ("Empty string", ""),
        ("Numbers only", "123456789"),
        ("URL input", "https://example.com/news"),
        ("Random OCR garbage", "@@##%%^^&&"),
    ]

    print("=" * 65)
    print("  RAVEN - Enhanced Language Detection Test Suite")
    print("=" * 65)

    for description, sample_text in test_cases:
        lang, conf = detect_language_with_confidence(sample_text)
        print(f"  Test       : {description}")
        print(f"  Input      : {repr(sample_text)}")
        print(f"  Result     : {lang} (confidence: {conf})")
        print("-" * 65)
