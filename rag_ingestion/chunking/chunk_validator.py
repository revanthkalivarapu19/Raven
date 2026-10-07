# src/chunking/chunk_validator.py
from typing import List
from rag_ingestion.config.config import config
from rag_ingestion.chunking.chunk_metadata import Chunk
from rag_ingestion.utils.logger import get_logger

class ChunkValidator:
    """Validates individual chunks against predefined boundaries."""
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.min_size = config.min_chunk_size
        self.max_size = config.max_chunk_size

    def validate(self, chunks: List[Chunk]) -> List[Chunk]:
        valid_chunks = []
        seen_texts = set()

        for chunk in chunks:
            if not chunk.text or not chunk.text.strip():
                self.logger.warning("Chunk validation failed: Empty chunk", extra={"chunk_id": chunk.chunk_id})
                continue

            try:
                chunk.text.encode('utf-8')
            except UnicodeEncodeError:
                self.logger.warning("Chunk validation failed: Invalid UTF-8", extra={"chunk_id": chunk.chunk_id})
                continue

            if chunk.word_count < self.min_size:
                self.logger.warning("Chunk validation failed: Below minimum size", extra={"chunk_id": chunk.chunk_id, "size": chunk.word_count})
                continue

            if chunk.word_count > self.max_size * 2:
                self.logger.warning("Chunk validation failed: Exceeds maximum size significantly", extra={"chunk_id": chunk.chunk_id, "size": chunk.word_count})
                continue

            if chunk.text in seen_texts:
                self.logger.warning("Chunk validation failed: Duplicate chunk in document", extra={"chunk_id": chunk.chunk_id})
                continue

            seen_texts.add(chunk.text)
            valid_chunks.append(chunk)

        return valid_chunks
