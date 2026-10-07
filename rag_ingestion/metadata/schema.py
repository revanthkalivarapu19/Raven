# src/metadata/schema.py
"""Pydantic model that defines the metadata for each text chunk stored in the vector database.

The fields capture provenance, content characteristics and chunk positioning.  All
values are serialisable to JSON so they can be persisted alongside the vector
embeddings.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, validator

class ChunkMetadata(BaseModel):
    """Metadata for a single document chunk.

    Attributes
    ----------
    source: str
        Human‑readable name of the source (e.g. "WHO").
    title: str
        Title of the original document.
    url: str
        Original URL where the document was retrieved.
    domain: str
        Top‑level domain (e.g. "who.int").
    authority_score: float
        Trust score between 0 and 1.
    publication_date: Optional[datetime]
        When the source document was originally published.
    retrieved_date: datetime
        Timestamp when the document was crawled.
    language: str
        Detected language of the chunk (ISO‑639‑1 code).
    original_language: Optional[str]
        Language of the original document if different from ``language``.
    document_type: str
        Type of the original document (pdf, html, txt, markdown, etc.).
    chunk_number: int
        Sequential number of the chunk within the source document (starting at 0).
    chunk_start: int
        Word index where the chunk starts (inclusive).
    chunk_end: int
        Word index where the chunk ends (exclusive).
    hash: str
        SHA‑256 hash of the raw chunk text – used for duplicate detection.
    """

    source: str = Field(..., description="Human‑readable source name")
    title: str = Field(..., description="Document title")
    url: str = Field(..., description="Original document URL")
    domain: str = Field(..., description="Top‑level domain, e.g. 'who.int'")
    authority_score: float = Field(..., ge=0.0, le=1.0, description="Trust score (0‑1)")
    publication_date: Optional[datetime] = Field(None, description="Original publication date")
    retrieved_date: datetime = Field(default_factory=datetime.utcnow, description="Crawl timestamp")
    language: str = Field(..., description="Detected language ISO‑639‑1 code")
    original_language: Optional[str] = Field(None, description="Original document language if different")
    document_type: str = Field(..., description="Document format such as pdf, html, txt")
    chunk_number: int = Field(..., ge=0, description="Zero‑based index of the chunk within the document")
    chunk_start: int = Field(..., ge=0, description="Word index where the chunk starts (inclusive)")
    chunk_end: int = Field(..., gt=0, description="Word index where the chunk ends (exclusive)")
    hash: str = Field(..., description="SHA‑256 hash of the chunk text for deduplication")

    @validator("hash")
    def hash_is_hex(cls, v: str) -> str:
        if len(v) != 64 or not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError("hash must be a 64‑character hexadecimal string")
        return v.lower()

    class Config:
        orm_mode = True
        json_encoders = {datetime: lambda v: v.isoformat()}
