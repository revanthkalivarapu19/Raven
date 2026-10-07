# src/pipeline/ingestion_pipeline.py
from typing import List, Tuple, Optional
from pathlib import Path

from rag_ingestion.pipeline.storage_manager import StorageManager
from rag_ingestion.pipeline.validation import ValidationLayer
from rag_ingestion.extractor import get_extractor
from rag_ingestion.metadata.processed import ProcessedDocument
from rag_ingestion.utils.logger import get_ingestion_logger
from rag_ingestion.chunking.chunk_manager import ChunkManager
from rag_ingestion.embeddings.embedding_manager import EmbeddingManager
from rag_ingestion.vectordb.index_manager import IndexManager

class IngestionPipeline:
    """Orchestrates the validation, extraction, and chunking of downloaded documents."""

    def __init__(self, storage_manager: StorageManager, chunk_manager: ChunkManager = None, embedding_manager: EmbeddingManager = None, index_manager: IndexManager = None):
        self.storage = storage_manager
        self.validator = ValidationLayer()
        self.chunk_manager = chunk_manager
        self.embedding_manager = embedding_manager
        self.index_manager = index_manager
        self.logger = get_ingestion_logger(self.__class__.__name__)

    def run(self, raw_documents: List[Tuple[Path, dict]]) -> List[ProcessedDocument]:
        processed_results = []

        for raw_path, metadata in raw_documents:
            try:
                self.logger.info("Starting ingestion for document", extra={"path": str(raw_path)})

                # 1. Validation
                ext = metadata.get("document_type", raw_path.suffix.lstrip('.'))
                if not self.validator.validate(raw_path, expected_ext=ext):
                    self.logger.warning("Document failed validation, skipping", extra={"path": str(raw_path)})
                    continue

                # 2. Extraction
                try:
                    extractor = get_extractor(ext)
                except ValueError as e:
                    self.logger.error("No extractor found", extra={"ext": ext, "error": str(e)})
                    continue

                processed_doc = extractor.extract(raw_path, metadata)
                if not processed_doc:
                    self.logger.warning("Extraction returned None, skipping", extra={"path": str(raw_path)})
                    continue

                # Fill in missing metadata fields
                processed_doc.file_size = raw_path.stat().st_size

                # 3. HOOK (Chunking Phase 3A)
                processed_doc = self._run_hooks(processed_doc)

                # 4. Storage & Registration
                self.storage.save_processed(processed_doc)
                self.storage.register_hash(processed_doc, raw_path)

                processed_results.append(processed_doc)
                self.logger.info("Document successfully ingested", extra={"id": processed_doc.document_id})

            except Exception as e:
                self.logger.error("Error during ingestion", extra={"path": str(raw_path), "error": str(e)})
                continue

        return processed_results

    def _run_hooks(self, doc: ProcessedDocument) -> ProcessedDocument:
        """Extension point for chunking phase."""
        if self.chunk_manager:
            chunks = self.chunk_manager.process(doc)
            if self.embedding_manager and chunks:
                self.embedding_manager.process(doc, chunks)
                if self.index_manager:
                    self.index_manager.process_document(doc)
        return doc
