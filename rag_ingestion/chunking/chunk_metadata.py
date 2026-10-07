# src/chunking/chunk_metadata.py
"""Pydantic models for representing text chunks and their associated metadata."""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class ChunkMetadata(BaseModel):
    """Metadata embedded within each Chunk."""
    document_id: str = Field(..., description="Permanent unique identifier for the parent document")
    chunk_id: str = Field(..., description="Deterministic chunk ID (e.g. document_id_chunk_index)")
    source: str = Field(..., description="Human-readable source name")
    domain: str = Field(..., description="Top-level domain of the source")
    url: str = Field(..., description="Original URL of the document")
    publication_date: Optional[datetime] = Field(None, description="Original publication date")
    authority_score: float = Field(1.0, description="Trust score of the source")
    language: str = Field("en", description="Language of the text")
    section_title: Optional[str] = Field(None, description="Title of the section this chunk belongs to")
    chunk_number: int = Field(..., description="Sequential index of this chunk")
    total_chunks: int = Field(0, description="Total chunks generated for this document")

    # Linked list style pointers for context reconstruction
    previous_chunk_id: Optional[str] = Field(None, description="ID of the previous chunk in the document")
    next_chunk_id: Optional[str] = Field(None, description="ID of the next chunk in the document")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

class Chunk(BaseModel):
    """Represents a discrete piece of text extracted from a ProcessedDocument."""
    chunk_id: str = Field(..., description="Unique deterministic identifier for this chunk")
    document_id: str = Field(..., description="Parent document identifier")
    chunk_index: int = Field(..., description="Index of this chunk within the document")

    text: str = Field(..., description="The text content of the chunk")
    word_count: int = Field(..., description="Number of words in this chunk")
    char_count: int = Field(..., description="Number of characters in this chunk")

    start_offset: int = Field(..., description="Character offset start within the parent document")
    end_offset: int = Field(..., description="Character offset end within the parent document")

    section_title: Optional[str] = Field(None, description="Heading or section title for context")
    source: str = Field(..., description="Source name")
    domain: str = Field(..., description="Domain name")
    language: str = Field("en", description="Language code")

    metadata: ChunkMetadata = Field(..., description="Full metadata object for the chunk")
