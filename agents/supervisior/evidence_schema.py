"""
NOTE: This is a local copy of agents/retrieval's real evidence_schema.py,
included here for standalone development/testing of the Supervisor
Agent. Replace with a shared import once modules are merged.
"""

from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

class Evidence(BaseModel):
    """Schema representing a retrieved evidence chunk.
    NOTE: This is a local copy of agents/retrieval's real evidence_schema.py,
    included here for standalone development/testing of the Supervisor
    Agent. Replace with a shared import once modules are merged.
    """
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunk_id: str
    document_id: str
    text: str
    domain: str
    source: str
    language: str
    retrieved_date: datetime = Field(default_factory=datetime.utcnow)
    similarity_score: float
    retrieval_method: Literal["faiss_local"] = "faiss_local"
    metadata: Dict[str, Any]
    claim_id: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    publication_date: Optional[datetime] = None
    authority_score: Optional[float] = None
