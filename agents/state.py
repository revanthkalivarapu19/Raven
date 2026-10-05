"""
state.py
========
RAVEN Project - Shared LangGraph State Schema

Defines the shared state contract flowing through:
Input Processing -> Claim Extraction -> Domain Detection -> STOP
"""

from typing import TypedDict, Optional, List, Dict, Any


class TextSegment(TypedDict, total=False):
    """
    Represents an atomic text segment with provenance and processing details.
    """
    source: str                      # "user_text" or "ocr"
    original_text: str               # Raw text prior to cleaning
    detected_language: str           # "en", "te", or "mixed"
    language_confidence: float       # Confidence of language detection (0.0 to 1.0)
    normalized_text: str             # Normalized text (Unicode NFC, cleaned)
    translated_text: Optional[str]   # Translated text if segment was Telugu/mixed
    ocr_confidence: Optional[float]  # OCR confidence score (0.0 to 1.0) if source is "ocr"
    bounding_box: Optional[List[Any]] # Bounding box coordinates if source is "ocr"
    order: int                       # Reading / sequential order position


class RavenState(TypedDict, total=False):
    """
    Canonical shared state for the active RAVEN pipeline:
    Input Processing -> Claim Extraction -> Domain Detection
    """
    # Raw Inputs
    text: Optional[str]
    image_path: Optional[str]
    
    # Flags on input existence
    had_text: bool
    had_image: bool

    # Segments & Provenance
    segments: List[Dict[str, Any]]

    # Aggregate Language Data
    original_language: str           # "en", "te", or "mixed"
    language_confidence: float       # Overall language confidence (0.0 to 1.0)

    # Aggregate OCR Data
    ocr_data: Dict[str, Any]         # Details on boxes, blocks, raw OCR text
    ocr_confidence: Optional[float]  # Weighted average OCR confidence

    # Aggregate Translation Data
    was_translated: bool             # True if any Telugu/mixed content was translated
    translation_info: Dict[str, Any] # Metadata on translated segments
    pre_translation_text: str        # Canonical text prior to translation

    # Final Canonical Processed Text (Claim-Ready)
    processed_text: str

    # Quality and Diagnostics Flags
    quality_flags: List[str]         # e.g., ["low_ocr_confidence", "repeated_ocr_lines"]
    errors: List[str]                # Recoverable error messages

    # Downstream Agent Outputs
    claim: str                       # Output of Claim Extraction
    domain: Dict[str, Any]           # Output of Domain Detection: {"domain", "confidence", "reason"}
