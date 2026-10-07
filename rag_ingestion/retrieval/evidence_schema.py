from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

class Evidence(BaseModel):
    """Schema representing a retrieved evidence chunk.

    Required fields are mandated by the retrieval specification.
    Optional fields may be missing when the source document does not contain them.
    """

    # Required fields
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunk_id: str
    document_id: str
    text: str
    domain: str
    source: str
    language: str
    retrieved_date: Optional[datetime] = Field(default_factory=datetime.utcnow)
    # None means no semantic similarity was calculated (for example, external API evidence).
    similarity_score: Optional[float] = None
    retrieval_method: Literal["faiss_local", "external"] = "faiss_local"
    metadata: Dict[str, Any]

    # Optional fields – may be None if not present in source metadata
    claim_id: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    publication_date: Optional[datetime] = None
    authority_score: Optional[float] = None
