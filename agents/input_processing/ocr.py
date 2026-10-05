"""
ocr.py
======
RAVEN Project - Enhanced OCR Module

Strengthened OCR processing with:
- Intelligent CPU-friendly image preprocessing (contrast CLAHE, resizing, noise reduction)
- Reading-order sorting (top-to-bottom, left-to-right line clustering)
- Preservation of bounding boxes, per-block confidence, and aggregate quality metrics
- Backward-compatible extract_text_from_image() helper
"""

import logging
import os
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Lazy singleton for reader to avoid load overhead on import
_READER = None


def get_ocr_reader():
    """
    Lazily initialize and return the EasyOCR reader singleton.
    """
    global _READER
    if _READER is None:
        import easyocr
        import torch
        use_gpu = torch.cuda.is_available()
        logger.info(f"Initializing EasyOCR reader (GPU={use_gpu}, languages=['en', 'te'])...")
        _READER = easyocr.Reader(['en', 'te'], gpu=use_gpu)
    return _READER


def deskew_image(gray: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Performs mild deskewing if a noticeable skew is detected.
    Limits correction to small angles to avoid rotating oriented layouts.
    """
    try:
        # Detect edges for Hough lines
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
        if lines is None:
            return gray

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) <= max_angle and abs(angle) > 0.5:
                angles.append(angle)

        if not angles:
            return gray

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5 or abs(median_angle) > max_angle:
            return gray

        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        m = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception as e:
        logger.debug(f"Deskew skipped due to: {e}")
        return gray


def preprocess_image(image_input) -> np.ndarray:
    """
    Preprocess image for optimal OCR:
    - Grayscale conversion
    - Dynamic resizing (upscale small text, cap oversized images)
    - Contrast Limited Adaptive Histogram Equalization (CLAHE)
    - Mild deskewing
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at path: {image_input}")
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Unable to read image at path: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape[:2]

    # Adaptive scaling:
    # Small screenshots benefit from 1.5x - 2.0x upscale
    # Oversized photos should be capped to prevent CPU/memory bottlenecks
    if h < 600 or w < 600:
        fx = fy = 1.75
        gray = cv2.resize(gray, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    elif h > 2200 or w > 2200:
        scale = 2000.0 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Mild deskewing
    gray = deskew_image(gray)

    # CLAHE contrast enhancement for text clarity
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced


def sort_reading_order(raw_results: List[Tuple[Any, str, float]]) -> List[Dict[str, Any]]:
    """
    Sort OCR blocks into a sensible natural reading order:
    Top-to-bottom line clustering, and left-to-right within lines.
    """
    if not raw_results:
        return []

    parsed_blocks = []
    for item in raw_results:
        box, text, conf = item
        text = str(text).strip()
        if not text:
            continue

        pts = np.array(box)
        min_x = float(np.min(pts[:, 0]))
        max_x = float(np.max(pts[:, 0]))
        min_y = float(np.min(pts[:, 1]))
        max_y = float(np.max(pts[:, 1]))
        center_y = (min_y + max_y) / 2.0
        height = max(1.0, max_y - min_y)

        parsed_blocks.append({
            "text": text,
            "confidence": float(conf),
            "box": [[float(p[0]), float(p[1])] for p in box],
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "center_y": center_y,
            "height": height
        })

    if not parsed_blocks:
        return []

    # Sort vertically first
    parsed_blocks.sort(key=lambda b: b["center_y"])

    # Cluster into lines based on vertical overlap
    lines = []
    current_line = [parsed_blocks[0]]

    for b in parsed_blocks[1:]:
        line_ref = current_line[-1]
        line_height = max(line_ref["height"], b["height"])
        # If vertical distance between centers is within 50% of box height, same line
        if abs(b["center_y"] - line_ref["center_y"]) < line_height * 0.55:
            current_line.append(b)
        else:
            # Sort previous line left-to-right
            current_line.sort(key=lambda x: x["min_x"])
            lines.append(current_line)
            current_line = [b]

    if current_line:
        current_line.sort(key=lambda x: x["min_x"])
        lines.append(current_line)

    ordered_blocks = []
    order_idx = 0
    for line in lines:
        for blk in line:
            ordered_blocks.append({
                "text": blk["text"],
                "confidence": round(blk["confidence"], 4),
                "box": blk["box"],
                "order": order_idx
            })
            order_idx += 1

    return ordered_blocks


def extract_ocr_data(image_path: str) -> Dict[str, Any]:
    """
    Extract structured OCR data from an image including:
    - ordered full text
    - per-block text, confidence, bounding boxes, and reading order
    - average confidence and quality flags
    """
    reader = get_ocr_reader()
    preprocessed = preprocess_image(image_path)
    raw_results = reader.readtext(preprocessed)

    ordered_blocks = sort_reading_order(raw_results)

    if not ordered_blocks:
        return {
            "text": "",
            "blocks": [],
            "average_confidence": 0.0,
            "quality_flags": ["empty_ocr_result"]
        }

    confidences = [b["confidence"] for b in ordered_blocks]
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    low_conf_count = sum(1 for c in confidences if c < 0.40)

    quality_flags = []
    if avg_conf < 0.50:
        quality_flags.append("low_average_ocr_confidence")
    if low_conf_count > len(ordered_blocks) / 3:
        quality_flags.append("multiple_low_confidence_blocks")

    # Combine blocks preserving natural line breaks
    combined_lines = []
    current_line_texts = []
    prev_order = None

    full_text = " ".join(b["text"] for b in ordered_blocks)

    return {
        "text": full_text,
        "blocks": ordered_blocks,
        "average_confidence": round(avg_conf, 4),
        "quality_flags": quality_flags
    }


def extract_text_from_image(image_path: str) -> str:
    """
    Convenience backwards-compatible function returning raw ordered text.
    """
    ocr_data = extract_ocr_data(image_path)
    return ocr_data.get("text", "")


if __name__ == "__main__":
    import sys
    test_img = "agents/input_processing/testimage.png"
    if not os.path.exists(test_img):
        test_img = "testimage.png"

    if os.path.exists(test_img):
        data = extract_ocr_data(test_img)
        print("=" * 60)
        print("OCR DATA RESULT")
        print("Average Confidence:", data["average_confidence"])
        print("Quality Flags:", data["quality_flags"])
        print("Block Count:", len(data["blocks"]))
        print("Extracted Text Preview:", data["text"][:200])
        print("=" * 60)
    else:
        print("No test image found.")
