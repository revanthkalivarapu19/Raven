# src/extractor/__init__.py
"""Extraction module for parsing raw documents into ProcessedDocument objects."""

from .base_extractor import BaseExtractor
from .html_cleaner import HTMLCleaner
from .pdf_extractor import PDFExtractor
from .text_extractor import TextExtractor

def get_extractor(document_type: str) -> BaseExtractor:
    """Factory to return the appropriate extractor based on document type.

    Args:
        document_type (str): The file extension or type (e.g., 'pdf', 'html', '.txt')

    Returns:
        BaseExtractor: An instance of the corresponding extractor.

    Raises:
        ValueError: If the document type is unsupported.
    """
    ext = document_type.lower().strip('.')
    if ext in ['html', 'htm']:
        return HTMLCleaner()
    elif ext == 'pdf':
        return PDFExtractor()
    elif ext in ['txt', 'md']:
        return TextExtractor()
    else:
        raise ValueError(f"Unsupported document type: {document_type}")

__all__ = ['BaseExtractor', 'HTMLCleaner', 'PDFExtractor', 'TextExtractor', 'get_extractor']
