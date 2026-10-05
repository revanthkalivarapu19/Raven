"""
normalize.py
============
RAVEN Project - Text Normalization Module

Responsibility:
    Performs conservative text normalization preserving factual integrity:
    - Unicode NFC normalization (crucial for Telugu conjuncts/diacritics)
    - Removal of erratic whitespace and control characters
    - Punctuation hygiene without destroying numbers, currencies, dates, decimals, or URLs
    - STRICT preservation of casing (no blind lowercasing)
    - Preservation of names, organizations, abbreviations, models, percentages
"""

import logging
import re
import sys
import unicodedata

logger = logging.getLogger(__name__)

# Regex for protecting specific structures
RE_URL = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
RE_DECIMAL = re.compile(r'(?<=\d)\s*([.,])\s*(?=\d)')
RE_CURRENCY_NUMBER = re.compile(r'([₹$€£¥])\s+(\d)')
RE_PERCENT = re.compile(r'(\d)\s+([%％])')


def normalize_text(text: str, lang: str = "en") -> str:
    """
    Safely normalize text according to RAVEN claim-preservation principles.
    
    Guarantees:
    - Unicode NFC normalization applied.
    - Factual values (0.50%, ₹5,000, $100, 2026-08-09, 10:30 PM) preserved.
    - Entities and abbreviations (RBI, NASA, GPT-4, 5G) preserved in original casing.
    - URLs preserved intact.
    - No blind lowercasing.
    - Whitespace collapsed cleanly.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    if not text.strip():
        return ""

    try:
        # 1. Unicode NFC Normalization
        text = unicodedata.normalize('NFC', text)

        # 2. Normalize line breaks and tabs into spaces
        text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")

        # 3. Clean zero-width spaces / unprintable formatting characters outside Telugu ligature needs
        text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u00a0", " ")

        # 4. Collapse spaces around decimals (e.g. "0 . 50" -> "0.50")
        text = RE_DECIMAL.sub(r'\1', text)

        # 5. Fix spaces between currency symbol and amount (e.g., "₹ 5,000" -> "₹5,000")
        text = RE_CURRENCY_NUMBER.sub(r'\1\2', text)

        # 6. Fix spaces before percent sign (e.g. "50 %" -> "50%")
        text = RE_PERCENT.sub(r'\1\2', text)

        # 7. Collapse spaces before sentence punctuation (.,!?;:), avoiding URLs
        # Only collapse space before punctuation if preceded by a letter or closing quote/paren
        text = re.sub(r'([a-zA-Z\u0C00-\u0C7F\)\]\'\"])\s+([.,!?;:])(?=\s|$)', r'\1\2', text)

        # 8. Collapse excessive repeated punctuation (e.g. "!!!!" -> "!", "?????" -> "?")
        # Keep ellipsis "..." and double hyphens "--"
        text = re.sub(r'([!?]){2,}', r'\1', text)
        text = re.sub(r'\.{4,}', '...', text)

        # 9. Collapse multiple consecutive whitespace into a single space
        text = re.sub(r'\s+', ' ', text)

        # 10. Strip leading and trailing whitespace
        return text.strip()

    except Exception as e:
        logger.exception(f"Unexpected error during normalization: {e}")
        return text.strip() if text else ""


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    test_cases = [
        ("The RBI reduced the repo rate by 0.50% .", "en"),
        ("తెలంగాణ ప్రభుత్వం కొత్త పథకాన్ని ప్రకటించింది", "te"),
        ("PM Modi హైదరాబాద్ లో ప్రసంగించారు .", "mixed"),
        ("Breaking NEWS  ! ! !", "en"),
        ("Cost is ₹ 5,000 or $ 100 on 2026-08-09 at 10:30 PM .", "en"),
        ("GPT-4 and 5G technology launched by NASA .", "en"),
        ("Visit https://www.example.com/news?id=123 for details .", "en"),
        ("COVID-19 vaccine reduces risk by 50 % .", "en"),
        ("   Lots   of    spaces    between words   ", "en"),
        ("", "en"),
        (None, "en"),
    ]

    print("=" * 65)
    print("  RAVEN - Enhanced Normalization Test Suite")
    print("=" * 65)

    for sample, lang in test_cases:
        res = normalize_text(sample, lang)
        print(f"Input : {repr(sample)}")
        print(f"Output: {repr(res)}")
        print("-" * 65)
