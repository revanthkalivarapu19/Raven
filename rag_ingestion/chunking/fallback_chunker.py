# src/chunking/fallback_chunker.py
import re
from typing import List

from rag_ingestion.config.config import config
from rag_ingestion.metadata.processed import ProcessedDocument
from rag_ingestion.chunking.chunk_metadata import Chunk
from rag_ingestion.chunking.base_chunker import BaseChunker

class FallbackChunker(BaseChunker):
    """Word-count based chunker with fixed overlap."""

    def __init__(self):
        self.max_words = config.max_chunk_size
        self.overlap = config.chunk_overlap

    def chunk(self, document: ProcessedDocument) -> List[Chunk]:
        text = document.text
        # We split by whitespace to count words and build chunks
        words = text.split()

        chunks = []
        chunk_index = 0
        start_idx = 0

        # Approximate offset tracking
        current_char_offset = 0

        while start_idx < len(words):
            end_idx = min(start_idx + self.max_words, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)

            # Recompute accurate offset based on finding the text in the original document
            # To be efficient, we search forward from previous offset
            found_idx = text.find(chunk_text[:50], current_char_offset)
            if found_idx != -1:
                chunk_start = found_idx
            else:
                chunk_start = current_char_offset

            chunk_end = chunk_start + len(chunk_text)
            current_char_offset = chunk_start + (len(" ".join(words[start_idx:start_idx + (self.max_words - self.overlap)]))) if start_idx + self.max_words < len(words) else chunk_end

            chunks.append(self._create_chunk(document, chunk_text, chunk_index, chunk_start, chunk_end))

            chunk_index += 1
            start_idx += (self.max_words - self.overlap)

            # Prevent infinite loop if overlap >= max_words
            if self.overlap >= self.max_words:
                break

        return chunks
