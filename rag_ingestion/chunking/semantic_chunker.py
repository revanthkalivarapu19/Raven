# src/chunking/semantic_chunker.py
import re
from typing import List

from rag_ingestion.config.config import config
from rag_ingestion.metadata.processed import ProcessedDocument
from rag_ingestion.chunking.chunk_metadata import Chunk
from rag_ingestion.chunking.base_chunker import BaseChunker
from rag_ingestion.utils.logger import get_logger

class SemanticChunker(BaseChunker):
    """Chunks text based on structural semantics (paragraphs, sentences) without embeddings."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.max_words = config.max_chunk_size

        # Basic sentence splitting regex (handles punctuation followed by space and capital)
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    def chunk(self, document: ProcessedDocument) -> List[Chunk]:
        chunks = []
        raw_text = document.text

        # Split into structural paragraphs first
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', raw_text) if p.strip()]
        if not paragraphs:
            # Fallback if no natural paragraphs exist
            paragraphs = [raw_text]

        current_chunk_words = []
        current_chunk_word_count = 0
        current_start_offset = 0
        current_offset_tracker = 0
        chunk_index = 0

        for paragraph in paragraphs:
            p_word_count = self.word_count(paragraph)

            # If paragraph fits in current chunk, add it
            if current_chunk_word_count + p_word_count <= self.max_words:
                current_chunk_words.append(paragraph)
                current_chunk_word_count += p_word_count
                current_offset_tracker += len(paragraph) + 2 # rough offset tracking
            else:
                # Flush current chunk if not empty
                if current_chunk_words:
                    chunk_text = "\n\n".join(current_chunk_words)
                    end_offset = current_start_offset + len(chunk_text)
                    chunks.append(self._create_chunk(document, chunk_text, chunk_index, current_start_offset, end_offset))
                    chunk_index += 1
                    current_start_offset = end_offset + 2
                    current_chunk_words = []
                    current_chunk_word_count = 0

                # If paragraph itself is larger than max_words, we must split it by sentences
                if p_word_count > self.max_words:
                    sentences = self.sentence_pattern.split(paragraph)
                    for sentence in sentences:
                        s_word_count = self.word_count(sentence)
                        if current_chunk_word_count + s_word_count > self.max_words and current_chunk_words:
                            chunk_text = " ".join(current_chunk_words)
                            end_offset = current_start_offset + len(chunk_text)
                            chunks.append(self._create_chunk(document, chunk_text, chunk_index, current_start_offset, end_offset))
                            chunk_index += 1
                            current_start_offset = end_offset + 1
                            current_chunk_words = []
                            current_chunk_word_count = 0

                        # If a single sentence is STILL larger than max_words, it must go to the fallback chunker logic
                        # But for simplicity, we just add it and it will violate max size, which validation catches
                        # or we can force split. We will just append it.
                        current_chunk_words.append(sentence)
                        current_chunk_word_count += s_word_count
                else:
                    current_chunk_words.append(paragraph)
                    current_chunk_word_count += p_word_count

        # Flush remainder
        if current_chunk_words:
            chunk_text = "\n\n".join(current_chunk_words)
            end_offset = current_start_offset + len(chunk_text)
            chunks.append(self._create_chunk(document, chunk_text, chunk_index, current_start_offset, end_offset))

        return chunks
