"""
normalize.py
============
RAVEN Project - Text Normalization Module

Responsibility:
    Receives text in its ORIGINAL detected language ("en", "te", or "mixed").
    Performs mechanical text normalization (cleaning whitespace, 
    punctuation, and casing) without changing the meaning.

This module does NOT perform:
    - OCR
    - Language Detection
    - Translation
    - NLP
    - Tokenization
    - Claim Extraction
    - Any LLM reasoning

Usage (from other modules):
    from agents.input_processing.normalize import normalize_text
    clean_text = normalize_text("Hello   World  !!!")
"""

import logging
import re

logger = logging.getLogger(__name__)

def normalize_text(text: str, lang: str) -> str:
    """
    Normalize text in its ORIGINAL detected language ("en", "te", or "mixed") mechanically before the NLP pipeline.
    
    Rules applied:
        1. Guard against None input.
        2. Replace newlines with spaces.
        3. Collapse multiple spaces into one.
        4. Remove spaces before punctuation.
        5. Collapse repeated punctuation.
        6. Convert English/mixed text to lowercase.

    Args:
        text (str): The raw text to normalize.
        lang (str): The original detected language ("en", "te", or "mixed").

    Returns:
        str: A safely normalized string.
    """
    try:
        # 1. Handle None safely. Return an empty string.
        if text is None:
            return ""

        # Convert to string just in case, though type hint says str
        if not isinstance(text, str):
            text = str(text)

        # 2. Replace newlines with spaces (done before space collapse).
        text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")

        # 3. Collapse multiple spaces into one.
        text = re.sub(r"\s+", " ", text)

        # 4. Remove spaces before punctuation.
        # Matches a space followed by any of .,!?;:
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)

        # 5. Collapse repeated punctuation.
        # Matches any of .,!?;: followed by one or more of the same character.
        text = re.sub(r'([.,!?;:])\1+', r"\1", text)

        # 6. Convert English/mixed text to lowercase unconditionally.
        if lang in ("en", "mixed"):
            text = text.lower()

        # Final cleanup for leading/trailing whitespace
        return text.strip()

    except Exception as e:
        logger.exception("Unexpected error during text normalization: %s", e)
        # Return whatever we can safely, or empty string on total failure
        return str(text).strip() if text else ""


# ---------------------------------------------------------------------------
# Test suite - runs only when this file is executed directly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Configure simple console logger for testing
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    test_cases = [
        ("en", "The Telangana Government announced a new scheme."),
        ("te", "తెలంగాణ ప్రభుత్వం కొత్త పథకాన్ని ప్రకటించింది"),
        ("mixed", "PM Modi హైదరాబాద్లో ప్రసంగించారు."),
        ("en", "Breaking NEWS !!!"),
        ("en", "OCR extracted English sentence"),
        ("en", "COVID vaccine is safe."),
        ("en", "RBI reduced the repo rate."),
        ("en", "NASA launched a new satellite."),
        ("en", "Cost is ₹4000."),
        ("en", "Visit https://www.example.com"),
        ("en", "#BreakingNews"),
        ("en", "@PMOIndia"),
        ("en", ""),
        ("en", None),
    ]

    print("=" * 60)
    print("  RAVEN - Normalization Module - Test Suite")
    print("=" * 60)

    for lang, sample_text in test_cases:
        result = normalize_text(sample_text, lang)
        
        # Format the input for display safely
        if sample_text is None:
            display_input = "None"
        else:
            display_input = repr(sample_text[:60]) + ("..." if len(sample_text) > 60 else "")

        print(f"\n  Lang     : {lang}")
        print(f"  Input    : {display_input}")
        print(f"  Result   : {repr(result)}")

    print("\n" + "=" * 60)
    print("  All test cases completed.")
    print("=" * 60)
