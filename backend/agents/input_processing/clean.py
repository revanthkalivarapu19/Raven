import logging
import re
import sys

logger = logging.getLogger(__name__)

def _is_garbage_token(token: str) -> bool:
    """
    Determines if a token is clearly standalone OCR garbage.
    Must be extremely conservative.
    """
    # 1. If it has any alphanumeric character (Unicode letters/numbers), keep it.
    if any(c.isalnum() for c in token):
        return False
        
    # 2. Check for typical OCR garbage characters
    # If the token contains any of these, and has no alphanumeric chars, it's highly likely noise.
    garbage_chars = set("@#^~|\\_")
    if any(c in garbage_chars for c in token):
        return True
        
    # 3. If it's multiple punctuation marks, keep common ones like ..., !!, --
    valid_repeated = {"...", "..", "!!", "!!!", "??", "???", "--", "---", "=="}
    if token in valid_repeated:
        return False
        
    # 4. If it's multiple symbols (and not in valid_repeated), consider it noise.
    if len(token) > 1:
        return True
        
    # 5. Length 1 token that didn't hit garbage_chars (e.g. '%', '$', '+', '-'). Keep it.
    return False

def clean_content(text: str) -> str:
    """
    Perform final, conservative content cleaning to remove obvious OCR noise.
    Does NOT perform lowercasing, full punctuation normalization, or general whitespace cleanup.
    """
    if text is None:
        return ""
        
    try:
        text = str(text)
    except Exception:
        logger.exception("Failed to convert input to string")
        return ""
        
    # Remove Unicode control characters that are unsafe/noisy.
    # Keep \x09 (\t), \x0A (\n), \x0D (\r)
    # Range is \x00-\x08, \x0b, \x0c, \x0e-\x1f, \x7f
    try:
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+', '', text)
    except Exception:
        logger.exception("Regex error removing control chars")

    try:
        # Split by whitespace, preserving the exact whitespace tokens
        parts = re.split(r'(\s+)', text)
        result_parts = []
        skip_next_whitespace = False
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Non-whitespace Token
                if part and _is_garbage_token(part):
                    # It's a garbage token, remove it.
                    # We will also try to clean up the space created by removal.
                    if len(result_parts) == 0:
                        skip_next_whitespace = True
                    else:
                        skip_next_whitespace = True
                else:
                    result_parts.append(part)
            else:  # Whitespace Token
                if skip_next_whitespace:
                    # Keep newlines if they exist in the skipped whitespace
                    if '\n' in part:
                        result_parts.append('\n')
                    skip_next_whitespace = False
                else:
                    result_parts.append(part)
                    
        # Join back together and strip leading/trailing whitespace
        text = "".join(result_parts).strip()
    except Exception:
        logger.exception("Error during token cleaning")
        
    return text

if __name__ == "__main__":
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        
    # Test suite
    test_cases = [
        "The government announced a new policy.",
        "@@## The government announced a new policy ^^",
        "AI and RBI announced new guidelines.",
        "COVID-19 vaccine reduces risk by 50%.",
        "RBI reduced the repo rate by 0.50%.",
        "The product costs ₹5000 or $100.",
        "GPT-4 and 5G technology were discussed.",
        "Meeting scheduled for 2026-08-09.",
        "తెలంగాణ government announced a new policy.",
        "https://example.com/news",
        "#BreakingNews",
        "@PMOIndia announced the update.",
        "@@## ^^ ~~ || !!!@@##",
        "",
        None
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"Test {i}:")
        print(f"Input : {repr(case)}")
        result = clean_content(case)
        print(f"Output: {repr(result)}")
        print("-" * 40)
