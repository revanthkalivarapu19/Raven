# src/metadata/processed.py
"""Pydantic model representing a fully processed document after extraction and cleaning.
It contains the cleaned text and all relevant metadata required for downstream RAG steps.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional

class ProcessedDocument(BaseModel):
    """Model for a document that has been downloaded, cleaned, and enriched."""

    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Permanent unique identifier for the document")
    title: str = Field(..., description="Document title (may be extracted from PDF or HTML heading)")
    source: str = Field(..., description="Human‑readable source name (e.g., 'WHO')")
    domain: str = Field(..., description="Top‑level domain, e.g. 'who.int'")
    url: str = Field(..., description="Original source URL")
    publication_date: Optional[datetime] = Field(None, description="Original publication date if known")
    retrieved_date: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the document was fetched")
    document_type: str = Field(..., description="File format: pdf, html, txt, markdown")
    language: str = Field("en", description="ISO‑639‑1 language code (default English)")
    authority_score: float = Field(1.0, description="Trust score of the source")
    version: str = Field("1.0", description="Schema/Processing version")
    content_hash: str = Field(..., description="SHA‑256 hash of the raw content – used for duplicate detection")
    processing_status: str = Field("processed", description="Status of the document ingestion")
    file_size: int = Field(0, description="Size of the raw file in bytes")
    mime_type: str = Field("application/octet-stream", description="MIME type of the raw document")

    # Keeping text as the main content payload
    text: str = Field(..., description="Cleaned, plain‑text content of the document")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
