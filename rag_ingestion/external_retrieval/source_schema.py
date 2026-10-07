# src/external_retrieval/source_schema.py
"""Schema for external evidence.
Defines the normalized structure returned by all external providers.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field


class ExternalEvidence(BaseModel):
    """Normalized evidence returned by an external source provider.

    This schema is designed to be compatible with downstream conversion
    into the core `Evidence` schema used by the verification pipeline.
    """

    # Required fields
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    source: str
    title: str
    url: str
    domain: str
    retrieved_date: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Optional fields
    publication_date: Optional[datetime] = None
    language: Optional[str] = None
    authority_score: Optional[float] = None
    source_type: Optional[str] = None
