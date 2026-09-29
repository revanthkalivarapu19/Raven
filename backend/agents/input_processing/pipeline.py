"""
pipeline.py
===========
RAVEN Project - Input Processing Pipeline

Orchestrates OCR, language detection, normalization, cleaning,
and conditional translation.
"""

import logging
from typing import Optional, Dict, Any

from agents.input_processing.ocr import extract_text_from_image
from agents.input_processing.language_detect import detect_language
from agents.input_processing.normalize import normalize_text
from agents.input_processing.clean import clean_content
from agents.input_processing.translate import translate_to_english

logger = logging.getLogger(__name__)

def process_input(text: Optional[str] = None, image_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Process text and/or an image through the RAVEN input pipeline.
    
    Args:
        text (Optional[str]): User-provided text.
        image_path (Optional[str]): Path to an image file.
        
    Returns:
        dict: A dictionary containing the processed text and metadata.
    """
    had_text = bool(text and text.strip())
    had_image = bool(image_path)
    
    if not had_text and not had_image:
        raise ValueError("Neither text nor image was provided.")
        
    ocr_text = None
    if had_image:
        logger.info(f"Running OCR on image: {image_path}")
        try:
            ocr_text = extract_text_from_image(image_path)
            if ocr_text:
                ocr_text = ocr_text.strip()
        except Exception as e:
            logger.exception("OCR processing failed")
            ocr_text = None
            
    # Combine user text and OCR text
    combined_text = ""
    if had_text and ocr_text:
        logger.info("Combining user text and OCR extracted text.")
        combined_text = f"{text.strip()}\n{ocr_text}"
    elif had_text:
        combined_text = text.strip()
    elif ocr_text:
        combined_text = ocr_text
        
    if not combined_text.strip():
        raise ValueError("No usable text was extracted from inputs.")
        
    # Language Detection
    logger.info("Detecting language...")
    original_language = detect_language(combined_text)
    logger.info(f"Detected language: {original_language}")
    
    # Normalization
    logger.info("Normalizing text...")
    normalized_text = normalize_text(combined_text, original_language)
    
    # Cleaning
    logger.info("Cleaning text...")
    cleaned_text = clean_content(normalized_text)
    
    # Translation
    if original_language in ("te", "mixed"):
        logger.info("Translating text to English...")
        final_text = translate_to_english(cleaned_text)
        was_translated = True
    else:
        logger.info("Skipping translation (language is English).")
        final_text = cleaned_text
        was_translated = False
        
    logger.info("Pipeline processing complete.")
    
    return {
        "processed_text": final_text,
        "original_language": original_language,
        "was_translated": was_translated,
        "had_image": had_image,
        "had_text": had_text,
        "pre_translation_text": cleaned_text
    }


if __name__ == "__main__":
    # Test suite
    import os
    import sys
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    test_image = "agents/input_processing/testimage.png"
    if not os.path.exists(test_image):
        # fallback if run from inside the folder
        test_image = "testimage_tel.png"
    
    test_cases = [
        (
            "English text only", 
            {"text": "The RBI reduced the repo rate by 0.50%.", "image_path": None}
        ),
        (
            "Telugu text only", 
            {"text": "రిజర్వ్ బ్యాంక్ వడ్డీ రేటును తగ్గించింది.", "image_path": None}
        ),
        (
            "Mixed Telugu-English text", 
            {"text": "నేను school కి వెళ్తాను and I study computer science.", "image_path": None}
        ),
        (
            "Image only", 
            {"text": None, "image_path": test_image if os.path.exists(test_image) else None}
        ),
        (
            "Text + image", 
            {"text": "This is a caption for the image.", "image_path": test_image if os.path.exists(test_image) else None}
        ),
        (
            "Neither text nor image", 
            {"text": None, "image_path": None}
        ),
    ]

    print("=" * 60)
    print("  RAVEN - Pipeline Test Suite")
    print("=" * 60)
    
    for name, kwargs in test_cases:
        print(f"\n--- Running Test: {name} ---")
        try:
            # If the test requires an image but we couldn't find one locally, skip gracefully 
            # rather than failing the expected ValueError test.
            if "image" in name.lower() and kwargs["image_path"] is None and name != "Neither text nor image":
                print("Skipped: No test image found.")
                continue
                
            result = process_input(**kwargs)
            print("SUCCESS")
            for k, v in result.items():
                print(f"  {k}: {repr(v)[:100]}")
        except ValueError as e:
            if name == "Neither text nor image":
                print(f"EXPECTED ERROR: {e}")
            else:
                print(f"UNEXPECTED ERROR: {e}")
        except Exception as e:
            print(f"FAILED: {e}")

    print("\n" + "=" * 60)
    print("  All test cases completed.")
    print("=" * 60)
