# src/embeddings/embedding_metadata.py
"""Pydantic models for embedding metadata.
The embedding layer works with Chunk objects only; the embedding manager
produces a document‑level embedding matrix and associated metadata.
"""

from datetime import datetime
from typing import List
from pydantic import BaseModel, Field


class ChunkEmbeddingMetadata(BaseModel):
    """Metadata for a single chunk embedding within a document matrix."""
    document_id: str = Field(..., description="Parent document identifier")
    chunk_id: str = Field(..., description="Chunk identifier, deterministic (docid_idx)")
    embedding_id: str = Field(..., description="Unique identifier for the embedding (often same as chunk_id)")
    embedding_model: str = Field(..., description="Model name used for embedding, e.g. BAAI/bge-large-en-v1.5")
    model_version: str = Field(..., description="Version string supplied via config, used for cache key")
    embedding_dimension: int = Field(..., description="Dimensionality of the vector")
    normalized: bool = Field(..., description="Whether L2‑normalization was applied")
    domain: str = Field(..., description="Domain of the source document")
    source: str = Field(..., description="Human readable source name")
    language: str = Field(..., description="Language code, e.g., 'en'")
    created_date: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of embedding creation")


class DocumentEmbeddingMetadata(BaseModel):
    """Document‑level metadata summarising the embedding matrix.
    This is stored as ``embedding_metadata.json`` next to the ``embeddings.npy`` file.
    """
    document_id: str = Field(..., description="Parent document identifier")
    embedding_model: str = Field(..., description="Model name used for embedding")
    model_version: str = Field(..., description="Version string from config")
    embedding_dimension: int = Field(..., description="Vector dimension")
    normalized: bool = Field(..., description="Whether vectors were normalized")
    total_chunks: int = Field(..., description="Number of rows (chunks) in the matrix")
    average_vector_norm: float = Field(..., description="Mean L2 norm of vectors (should be 1.0 when normalized)")
    created_date: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of matrix creation")
    chunk_metadata: List[ChunkEmbeddingMetadata] = Field(..., description="Metadata for each chunk in order")
