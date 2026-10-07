# src/extractor/html_cleaner.py
import re
from pathlib import Path
from typing import Optional
from datetime import datetime

from bs4 import BeautifulSoup

from rag_ingestion.metadata.processed import ProcessedDocument
from rag_ingestion.extractor.base_extractor import BaseExtractor
from rag_ingestion.utils.logger import get_logger

class HTMLCleaner(BaseExtractor):
    """Extractor for HTML documents. Removes boilerplate and extracts plain text."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def extract(self, file_path: Path, source_metadata: dict) -> Optional[ProcessedDocument]:
        try:
            content = file_path.read_bytes()
            return self.extract_from_content(content, source_metadata, file_path=file_path)
        except Exception as e:
            self.logger.error("Error reading HTML file", extra={"path": str(file_path), "error": str(e)})
            return None

    def extract_from_content(self, content: bytes, source_metadata: dict, file_path: Optional[Path] = None) -> Optional[ProcessedDocument]:
        try:
            soup = BeautifulSoup(content, 'html.parser')

            # Remove scripts, styles, and common boilerplate tags
            for element in soup(["script", "style", "noscript", "meta", "link", "header", "footer", "nav", "aside"]):
                element.decompose()

            # Extract title if available
            extracted_title = None
            if soup.title and soup.title.string:
                extracted_title = soup.title.string.strip()

            title = extracted_title or source_metadata.get("title", "Unknown Title")

            # Get text and clean up whitespace
            text = soup.get_text(separator=' ')
            cleaned_text = re.sub(r'\s+', ' ', text).strip()

            if not cleaned_text:
                self.logger.warning("No text extracted from HTML", extra={"path": str(file_path) if file_path else "in-memory"})
                return None

            sha256 = self.compute_sha256(content)

            return ProcessedDocument(
                title=title,
                url=source_metadata.get("url", ""),
                source=source_metadata.get("source", "Unknown"),
                domain=source_metadata.get("domain", "Unknown"),
                publication_date=source_metadata.get("publication_date"),
                retrieved_date=source_metadata.get("retrieved_date", datetime.utcnow()),
                document_type="html",
                language=source_metadata.get("language", "en"),
                sha256=sha256,
                content_hash=sha256,
                text=cleaned_text
            )
        except Exception as e:
            self.logger.exception(f"Error extracting HTML: {e}")
            return None
