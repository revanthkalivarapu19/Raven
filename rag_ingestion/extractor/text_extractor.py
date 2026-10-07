# src/extractor/text_extractor.py
import re
from pathlib import Path
from typing import Optional
from datetime import datetime

from rag_ingestion.metadata.processed import ProcessedDocument
from rag_ingestion.extractor.base_extractor import BaseExtractor
from rag_ingestion.utils.logger import get_logger

class TextExtractor(BaseExtractor):
    """Extractor for plain text and markdown documents."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def extract(self, file_path: Path, source_metadata: dict) -> Optional[ProcessedDocument]:
        try:
            content = file_path.read_bytes()
            text = content.decode('utf-8', errors='replace')

            # Simple whitespace cleanup
            cleaned_text = re.sub(r'\s+', ' ', text).strip()

            if not cleaned_text:
                self.logger.warning("No text extracted from TXT/MD", extra={"path": str(file_path)})
                return None

            sha256 = self.compute_sha256(content)

            # For pure text files, we usually have to rely heavily on source metadata for title
            return ProcessedDocument(
                title=source_metadata.get("title", "Unknown Title"),
                url=source_metadata.get("url", ""),
                source=source_metadata.get("source", "Unknown"),
                domain=source_metadata.get("domain", "Unknown"),
                publication_date=source_metadata.get("publication_date"),
                retrieved_date=source_metadata.get("retrieved_date", datetime.utcnow()),
                document_type=source_metadata.get("document_type", "txt"),
                language=source_metadata.get("language", "en"),
                sha256=sha256,
                content_hash=sha256,
                text=cleaned_text
            )
        except Exception as e:
            self.logger.error("Error extracting Text", extra={"path": str(file_path), "error": str(e)})
            return None
