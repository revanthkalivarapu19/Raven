# src/extractor/html_cleaner.py
import re
from pathlib import Path
from typing import Optional
from datetime import datetime

from bs4 import BeautifulSoup

from rag_ingestion.metadata.processed import ProcessedDocument
from rag_ingestion.evidence_quality import has_substantive_text
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

            # Prefer semantic article content where available. Some NASA pages
            # implement section navigation as a styled div rather than a nav
            # element, so remove those explicit navigation containers too.
            for element in soup.select(
                '[role="navigation"], [aria-label*="navigation" i], '
                '[class*="secondary-navigation"], [class*="breadcrumb"]'
            ):
                element.decompose()

            # Extract title if available
            extracted_title = None
            if soup.title and soup.title.string:
                extracted_title = soup.title.string.strip()

            title = extracted_title or source_metadata.get("title", "Unknown Title")

            # Modern NASA article pages place the useful body in main/article;
            # extracting the whole document otherwise leads with site menus.
            content_root = soup.select_one("main article") or soup.select_one("main") or soup.select_one("article") or soup.body or soup

            # Preserve descriptive image alt text as source-authored content.
            # Short labels/logos are excluded to avoid reintroducing navigation.
            for image in content_root.find_all("img", alt=True):
                alt = re.sub(r"\s+", " ", image.get("alt", "")).strip()
                if len(alt) >= 40:
                    image.replace_with(soup.new_string(f" Image description: {alt} "))
                else:
                    image.decompose()

            text = content_root.get_text(separator=' ')
            cleaned_text = re.sub(r'\s+', ' ', text).strip()

            if not has_substantive_text(cleaned_text):
                self.logger.warning("No substantive text extracted from HTML", extra={"path": str(file_path) if file_path else "in-memory"})
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
