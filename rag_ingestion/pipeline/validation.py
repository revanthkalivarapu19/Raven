# src/pipeline/validation.py
import os
import mimetypes
from pathlib import Path
from rag_ingestion.config.config import config
from rag_ingestion.utils.logger import get_errors_logger

class ValidationLayer:
    """Validates downloaded raw files before extraction."""

    def __init__(self):
        self.logger = get_errors_logger(self.__class__.__name__)
        # Ensure mimetypes are initialized
        mimetypes.init()

    def validate(self, file_path: Path, expected_ext: str = None) -> bool:
        """Run all validation checks on the file.

        Args:
            file_path: Path to the downloaded file.
            expected_ext: Optional expected file extension (e.g. '.pdf')

        Returns:
            bool: True if valid, False otherwise.
        """
        # 1. File exists
        if not file_path.exists() or not file_path.is_file():
            self.logger.error("Validation failed: File does not exist", extra={"path": str(file_path)})
            return False

        # 2. Reasonable file size & non-empty
        size = file_path.stat().st_size
        if size == 0:
            self.logger.error("Validation failed: File is empty", extra={"path": str(file_path)})
            return False

        if size > config.max_file_size_bytes:
            self.logger.error(
                "Validation failed: File exceeds maximum allowed size",
                extra={"path": str(file_path), "size": size, "max_size": config.max_file_size_bytes}
            )
            return False

        # 3. Supported MIME type
        # Guess mime type based on file extension
        mime_type, _ = mimetypes.guess_type(str(file_path))
        ext = file_path.suffix.lower().lstrip('.')
        if ext not in config.supported_document_types:
            self.logger.error("Validation failed: Unsupported document type", extra={"path": str(file_path), "ext": ext})
            return False

        # 4. Valid PDF structure (if PDF)
        if ext == 'pdf':
            if not self._is_valid_pdf(file_path):
                self.logger.error("Validation failed: Invalid or corrupted PDF structure", extra={"path": str(file_path)})
                return False

        return True

    def _is_valid_pdf(self, file_path: Path) -> bool:
        """Basic check for PDF magic number."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(5)
                return header == b'%PDF-'
        except Exception as e:
            self.logger.error("Failed to read PDF header", extra={"path": str(file_path), "error": str(e)})
            return False
