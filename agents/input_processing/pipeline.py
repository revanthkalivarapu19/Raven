"""
pipeline.py
===========
RAVEN Project - Enhanced Input Processing Pipeline

Orchestrates:
- Independent segmentation for user text vs. OCR image input
- Reading-order aware OCR with bounding boxes and confidence preservation
- Script-aware per-segment language detection with confidence
- Unicode NFC normalization without casing corruption
- Conservative OCR noise cleaning and repeated-line deduplication
- Sentence-aware translation preserving English and protected entities
- Propagation of quality, provenance, and diagnostic flags into shared state
"""

import logging
import os
import sys
from typing import Optional, Dict, Any, List

from agents.input_processing.ocr import extract_ocr_data
from agents.input_processing.language_detect import detect_language_with_confidence
from agents.input_processing.normalize import normalize_text
from agents.input_processing.clean import clean_content_with_flags
from agents.input_processing.translate import translate_with_metadata

logger = logging.getLogger(__name__)


def process_input(text: Optional[str] = None, image_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Process text and/or image inputs through the redesigned RAVEN input pipeline.

    Maintains distinct segment provenance for user-entered text and OCR text,
    processes them independently, and synthesizes a claim-ready representation
    with rich quality and provenance metadata for the shared LangGraph state.
    """
    had_text = bool(text and text.strip())
    had_image = bool(image_path)

    if not had_text and not had_image:
        raise ValueError("Neither text nor image was provided.")

    segments: List[Dict[str, Any]] = []
    quality_flags: List[str] = []
    ocr_data: Dict[str, Any] = {}
    ocr_confidence: Optional[float] = None
    order_counter = 0

    # -------------------------------------------------------------------------
    # 1. Process User Text Segment
    # -------------------------------------------------------------------------
    user_clean = ""
    user_claim_ready = ""
    user_lang = "en"
    user_lang_conf = 1.0

    if had_text:
        raw_user_text = text.strip()
        user_lang, user_lang_conf = detect_language_with_confidence(raw_user_text)
        user_norm = normalize_text(raw_user_text, user_lang)
        user_clean, u_flags = clean_content_with_flags(user_norm)
        quality_flags.extend(u_flags)

        user_translated = None
        if user_lang in ("te", "mixed"):
            trans_meta = translate_with_metadata(user_clean)
            user_translated = trans_meta["translated_text"]
            user_claim_ready = user_translated
        else:
            user_claim_ready = user_clean

        segments.append({
            "source": "user_text",
            "original_text": raw_user_text,
            "detected_language": user_lang,
            "language_confidence": user_lang_conf,
            "normalized_text": user_clean,
            "translated_text": user_translated,
            "ocr_confidence": None,
            "bounding_box": None,
            "order": order_counter,
            "quality_flags": u_flags
        })
        order_counter += 1

    # -------------------------------------------------------------------------
    # 2. Process OCR Image Segment
    # -------------------------------------------------------------------------
    ocr_clean = ""
    ocr_claim_ready = ""
    ocr_lang = "en"
    ocr_lang_conf = 1.0

    if had_image:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Provided image path does not exist: {image_path}")

        logger.info(f"Extracting OCR data from image: {image_path}")
        ocr_data = extract_ocr_data(image_path)
        ocr_raw = ocr_data.get("text", "").strip()
        ocr_confidence = ocr_data.get("average_confidence", 0.0)
        quality_flags.extend(ocr_data.get("quality_flags", []))

        if ocr_raw:
            ocr_lang, ocr_lang_conf = detect_language_with_confidence(ocr_raw)
            ocr_norm = normalize_text(ocr_raw, ocr_lang)
            ocr_clean, o_flags = clean_content_with_flags(ocr_norm)
            quality_flags.extend(o_flags)

            ocr_translated = None
            if ocr_lang in ("te", "mixed"):
                trans_meta = translate_with_metadata(ocr_clean)
                ocr_translated = trans_meta["translated_text"]
                ocr_claim_ready = ocr_translated
            else:
                ocr_claim_ready = ocr_clean

            segments.append({
                "source": "ocr",
                "original_text": ocr_raw,
                "detected_language": ocr_lang,
                "language_confidence": ocr_lang_conf,
                "normalized_text": ocr_clean,
                "translated_text": ocr_translated,
                "ocr_confidence": ocr_confidence,
                "bounding_box": None,
                "blocks": ocr_data.get("blocks", []),
                "order": order_counter,
                "quality_flags": o_flags
            })
            order_counter += 1
        else:
            quality_flags.append("empty_ocr_text")

    # Guard: Ensure at least some text is available
    if not any(s.get("normalized_text") for s in segments):
        raise ValueError("No usable text was extracted from inputs.")

    # -------------------------------------------------------------------------
    # 3. Aggregate Metadata
    # -------------------------------------------------------------------------
    # Aggregate Language
    if had_text and (had_image and ocr_clean):
        if user_lang == "mixed" or ocr_lang == "mixed":
            agg_lang = "mixed"
        elif user_lang != ocr_lang:
            agg_lang = "mixed"
        else:
            agg_lang = user_lang
        agg_lang_conf = round((user_lang_conf + ocr_lang_conf) / 2.0, 3)
    elif had_text:
        agg_lang = user_lang
        agg_lang_conf = user_lang_conf
    else:
        agg_lang = ocr_lang
        agg_lang_conf = ocr_lang_conf

    # Check if translation occurred
    was_translated = any(s.get("translated_text") is not None for s in segments)

    # Build canonical combined pre-translation and claim-ready processed text
    claim_ready_parts = []
    pre_trans_parts = []

    for seg in segments:
        txt = seg.get("translated_text") or seg.get("normalized_text", "")
        if txt:
            claim_ready_parts.append(txt)
        raw_clean = seg.get("normalized_text", "")
        if raw_clean:
            pre_trans_parts.append(raw_clean)

    processed_text = "\n".join(claim_ready_parts).strip()
    pre_translation_text = "\n".join(pre_trans_parts).strip()

    # Deduplicate quality flags preserving order
    unique_quality_flags = list(dict.fromkeys(quality_flags))

    return {
        "processed_text": processed_text,
        "pre_translation_text": pre_translation_text,
        "original_language": agg_lang,
        "language_confidence": agg_lang_conf,
        "was_translated": was_translated,
        "had_image": had_image,
        "had_text": had_text,
        "segments": segments,
        "ocr_data": ocr_data,
        "ocr_confidence": ocr_confidence,
        "quality_flags": unique_quality_flags,
        "translation_info": {
            "was_translated": was_translated,
            "translated_segments": sum(1 for s in segments if s.get("translated_text") is not None)
        }
    }


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    test_image = "agents/input_processing/testimage.png"
    if not os.path.exists(test_image):
        test_image = "testimage_tel.png"

    test_cases = [
        ("English text only", {"text": "The RBI reduced the repo rate by 0.50%.", "image_path": None}),
        ("Telugu text only", {"text": "రిజర్వ్ బ్యాంక్ వడ్డీ రేటును తగ్గించింది.", "image_path": None}),
        ("Mixed Telugu-English text", {"text": "నేను school కి వెళ్తాను and I study computer science.", "image_path": None}),
        ("Image only", {"text": None, "image_path": test_image if os.path.exists(test_image) else None}),
        ("Text + image", {"text": "Is this claim true?", "image_path": test_image if os.path.exists(test_image) else None}),
        ("Neither text nor image", {"text": None, "image_path": None}),
    ]

    print("=" * 65)
    print("  RAVEN - Enhanced Pipeline Test Suite")
    print("=" * 65)

    for name, kwargs in test_cases:
        print(f"\n--- Running Test: {name} ---")
        try:
            res = process_input(**kwargs)
            print("STATUS: SUCCESS")
            print("  processed_text     :", repr(res["processed_text"][:80]))
            print("  original_language  :", res["original_language"], f"(conf: {res['language_confidence']})")
            print("  was_translated     :", res["was_translated"])
            print("  segments count     :", len(res["segments"]))
            for s in res["segments"]:
                print(f"    - Segment [{s['source']}]: {repr(s['normalized_text'][:40])} (lang: {s['detected_language']}, ocr_conf: {s.get('ocr_confidence')})")
            print("  quality_flags      :", res["quality_flags"])
        except ValueError as e:
            if name == "Neither text nor image":
                print(f"STATUS: EXPECTED ERROR -> {e}")
            else:
                print(f"STATUS: UNEXPECTED VALUE ERROR -> {e}")
        except Exception as e:
            print(f"STATUS: FAILED -> {e}")
