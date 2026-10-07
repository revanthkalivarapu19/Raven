# src/extractor/base_extractor.py
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from rag_ingestion.metadata.processed import ProcessedDocument

class BaseExtractor(ABC):
    """Abstract base class for all document extractors."""

    @abstractmethod
    def extract(self, file_path: Path, source_metadata: dict) -> Optional[ProcessedDocument]:
        """Extract text and metadata from the given file path.

        Args:
            file_path (Path): Path to the raw downloaded file.
            source_metadata (dict): Metadata collected during the crawl phase
                (e.g., url, source name, domain).

        Returns:
            Optional[ProcessedDocument]: The cleaned document with metadata, or None if extraction fails.
        """
        pass

    def compute_sha256(self, content: bytes) -> str:
        """Compute the SHA-256 hash of the content."""
        return hashlib.sha256(content).hexdigest()
