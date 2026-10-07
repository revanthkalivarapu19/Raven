# src/chunking/base_chunker.py
import re
from abc import ABC, abstractmethod
from typing import List

from rag_ingestion.metadata.processed import ProcessedDocument
from rag_ingestion.chunking.chunk_metadata import Chunk, ChunkMetadata

class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, document: ProcessedDocument) -> List[Chunk]:
        """Convert a ProcessedDocument into a list of Chunks."""
        pass

    def word_count(self, text: str) -> int:
        """Utility to accurately count words in a string."""
        return len(re.findall(r'\b\w+\b', text))

    def _create_chunk(self,
                      document: ProcessedDocument,
                      text: str,
                      chunk_index: int,
                      start_offset: int,
                      end_offset: int,
                      section_title: str = None) -> Chunk:
        """Helper method to instantiate a Chunk object with deterministic ID."""
        chunk_id = f"{document.document_id}_chunk_{chunk_index}"

        metadata = ChunkMetadata(
            document_id=document.document_id,
            chunk_id=chunk_id,
            source=document.source,
            domain=document.domain,
            url=document.url,
            publication_date=document.publication_date,
            authority_score=document.authority_score,
            language=document.language,
            section_title=section_title or document.title,
            chunk_number=chunk_index,
            total_chunks=0 # Updated later by ChunkManager
        )

        return Chunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            chunk_index=chunk_index,
            text=text,
            word_count=self.word_count(text),
            char_count=len(text),
            start_offset=start_offset,
            end_offset=end_offset,
            section_title=section_title or document.title,
            source=document.source,
            domain=document.domain,
            language=document.language,
            metadata=metadata
        )
