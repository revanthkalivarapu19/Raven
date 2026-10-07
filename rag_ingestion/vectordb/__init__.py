# src/vectordb/__init__.py
"""VectorDB package exposing public classes.
Provides:
- BaseVectorStore
- FAISSStore
- IndexManager
"""

from .base_vector_store import BaseVectorStore
from .faiss_store import FAISSStore
from .index_manager import IndexManager
