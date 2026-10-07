# src/chunking/chunk_manager.py
import time
from typing import List
from rag_ingestion.metadata.processed import ProcessedDocument
from rag_ingestion.chunking.chunk_metadata import Chunk
from rag_ingestion.chunking.semantic_chunker import SemanticChunker
from rag_ingestion.chunking.fallback_chunker import FallbackChunker
from rag_ingestion.chunking.chunk_validator import ChunkValidator
from rag_ingestion.utils.logger import get_logger

class ChunkManager:
    """Orchestrates chunkers, validates output, and yields chunks."""

    def __init__(self, storage_manager):
        self.storage = storage_manager
        self.semantic_chunker = SemanticChunker()
        self.fallback_chunker = FallbackChunker()
        self.validator = ChunkValidator()
        self.logger = get_logger(self.__class__.__name__)

        # Stats
        self.docs_processed = 0
        self.total_chunks = 0
        self.total_words = 0
        self.largest_chunk = 0
        self.smallest_chunk = float('inf')

    def process(self, document: ProcessedDocument) -> List[Chunk]:
        start_time = time.time()

        chunks = []
        try:
            chunks = self.semantic_chunker.chunk(document)
        except Exception as e:
            self.logger.error("Semantic chunking failed, falling back", extra={"doc_id": document.document_id, "error": str(e)})
            chunks = self.fallback_chunker.chunk(document)

        valid_chunks = self.validator.validate(chunks)

        if not valid_chunks and document.text.strip():
            self.logger.warning("All semantic chunks invalid, forcing fallback", extra={"doc_id": document.document_id})
            chunks = self.fallback_chunker.chunk(document)
            valid_chunks = self.validator.validate(chunks)

        # Update metadata links and total count
        total_chunks = len(valid_chunks)
        for i, chunk in enumerate(valid_chunks):
            chunk.metadata.total_chunks = total_chunks
            if i > 0:
                chunk.metadata.previous_chunk_id = valid_chunks[i-1].chunk_id
            if i < total_chunks - 1:
                chunk.metadata.next_chunk_id = valid_chunks[i+1].chunk_id

            # Stats
            wc = chunk.word_count
            self.total_words += wc
            if wc > self.largest_chunk: self.largest_chunk = wc
            if wc < self.smallest_chunk: self.smallest_chunk = wc

        self.docs_processed += 1
        self.total_chunks += total_chunks

        # Persist chunks and manifest via StorageManager
        self.storage.save_chunks(document, valid_chunks)

        proc_time = time.time() - start_time
        self.logger.info("Chunking complete", extra={
            "doc_id": document.document_id,
            "total_chunks": total_chunks,
            "time": proc_time
        })

        return valid_chunks

    def get_summary(self, processing_time: float) -> str:
        avg_size = (self.total_words / self.total_chunks) if self.total_chunks > 0 else 0
        smallest = self.smallest_chunk if self.smallest_chunk != float('inf') else 0
        return f"""
====================================
CHUNK SUMMARY
====================================
Documents Processed: {self.docs_processed}
Chunks Generated: {self.total_chunks}
Average Chunk Size: {avg_size:.1f} words
Largest Chunk: {self.largest_chunk} words
Smallest Chunk: {smallest} words
Processing Time: {processing_time:.2f} seconds
====================================
"""
