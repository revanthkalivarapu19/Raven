# src/extractor/pdf_extractor.py
import re
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from rag_ingestion.metadata.processed import ProcessedDocument
from rag_ingestion.extractor.base_extractor import BaseExtractor
from rag_ingestion.utils.logger import get_logger

class PDFExtractor(BaseExtractor):
    """Extractor for PDF documents using PyMuPDF."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def extract(self, file_path: Path, source_metadata: dict) -> Optional[ProcessedDocument]:
        if fitz is None:
            self.logger.error("PyMuPDF (fitz) is not installed. Cannot extract PDF.")
            return None

        try:
            content = file_path.read_bytes()

            text_blocks = []
            title = source_metadata.get("title", "Unknown Title")

            with fitz.open(stream=content, filetype="pdf") as doc:
                # Try to extract title from PDF metadata if not provided
                if doc.metadata and doc.metadata.get("title"):
                    pdf_title = doc.metadata.get("title").strip()
                    if pdf_title:
                        title = pdf_title

                for page in doc:
                    text_blocks.append(page.get_text())

            raw_text = "\n".join(text_blocks)
            cleaned_text = re.sub(r'\s+', ' ', raw_text).strip()

            if not cleaned_text:
                self.logger.warning("No text extracted from PDF", extra={"path": str(file_path)})
                return None

            sha256 = self.compute_sha256(content)

            return ProcessedDocument(
                title=title,
                url=source_metadata.get("url", ""),
                source=source_metadata.get("source", "Unknown"),
                domain=source_metadata.get("domain", "Unknown"),
                publication_date=source_metadata.get("publication_date"),
                retrieved_date=source_metadata.get("retrieved_date", datetime.utcnow()),
                document_type="pdf",
                language=source_metadata.get("language", "en"),
                sha256=sha256,
                text=cleaned_text
            )
        except Exception as e:
            self.logger.error("Error extracting PDF", extra={"path": str(file_path), "error": str(e)})
            return None
