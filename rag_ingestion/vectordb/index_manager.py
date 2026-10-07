# src/vectordb/index_manager.py
from rag_ingestion.config.config import config
"""IndexManager orchestrates per‑domain FAISS indexing.
It receives processed documents (with embeddings already saved) and
creates/updates a FAISS index for the document's domain.
"""

from pathlib import Path
from datetime import datetime
import json

from rag_ingestion.pipeline.storage_manager import StorageManager
from rag_ingestion.metadata.processed import ProcessedDocument
from .faiss_store import FAISSStore
from .vector_metadata import IndexMetadata
from .vector_validator import VectorValidator

class IndexManager:
    """Manages FAISS indexes for each domain.

    Steps performed for each ProcessedDocument:
    1. Load existing index (or create new if missing).
    2. Load document's embedding matrix from ``data/embeddings/<doc_id>/embeddings.npy``.
    3. Validate dimension & model via ``VectorValidator``.
    4. Append vectors, update lookup, statistics, and metadata.
    5. Persist all artifacts through ``StorageManager``.
    """

    def __init__(self, storage: StorageManager):
        self.storage = storage
        # In‑memory registry used to track versions per domain
        self.registry = self.storage.load_index_registry()

    def _ensure_index(self, domain: str, dimension: int) -> FAISSStore:
        """Load or create a FAISSStore for ``domain`` with ``dimension``.
        """
        try:
            index, lookup = self.storage.load_faiss_store(domain)
            store = FAISSStore(dimension)
            store.index = index
            store.id_to_idx = lookup.get("id_to_idx", {})
            store.idx_to_id = {int(k): v for k, v in lookup.get("idx_to_id", {}).items()}
            return store
        except FileNotFoundError:
            # New index
            return FAISSStore(dimension)

    def _update_registry(self, domain: str, version: int, index_dir: Path) -> None:
        self.registry[domain] = {"version": version, "index_dir": str(index_dir)}
        self.storage.save_index_registry(self.registry)

    def process_document(self, doc: ProcessedDocument) -> None:
        """Create or update FAISS index for the document's domain.
        Consumes the embedding matrix and metadata already saved by Phase 3B.
        """
        domain = doc.domain.lower()
        source = doc.source
        doc_id = doc.document_id

        # Load embedding metadata
        doc_metadata = self.storage.load_embedding_metadata(domain, source, doc_id)
        dimension = doc_metadata.get("embedding_dimension")
        model_name = doc_metadata.get("embedding_model")

        if not dimension or not model_name:
            raise ValueError(f"Incomplete embedding metadata for {doc_id}")

        chunk_metadatas = doc_metadata.get("chunk_metadata", [])
        chunk_ids = [cmd.get("chunk_id") for cmd in chunk_metadatas]

        # Load embeddings matrix
        import numpy as np
        vectors = self.storage.load_embedding_matrix(domain, source, doc_id)

        if vectors.shape[0] != len(chunk_ids):
            raise ValueError(f"Mismatch between vector count ({vectors.shape[0]}) and chunk count ({len(chunk_ids)})")

        if np.isnan(vectors).any() or np.isinf(vectors).any():
            raise ValueError(f"Embeddings for {doc_id} contain NaN or Inf values")

        # Metadata
        metadata = IndexMetadata(
            version=self.registry.get(domain, {}).get("version", 1),
            model_name=model_name,
            dimension=dimension,
            created_at=datetime.utcnow().isoformat()
        )

        # Load existing index if present
        store = self._ensure_index(domain, dimension)

        # Validate existing index metadata if present
        try:
            existing_meta = self.storage.load_index_metadata(domain)
            VectorValidator.validate_metadata(IndexMetadata.from_dict(existing_meta), model_name, dimension)
        except FileNotFoundError:
            pass

        # Append vectors (handles duplicate prevention internally)
        store.add_vectors(chunk_ids, vectors)

        # Persist via StorageManager
        lookup = {"id_to_idx": store.id_to_idx, "idx_to_id": store.idx_to_id}
        self.storage.save_faiss_store(domain, store.index, lookup)
        self.storage.save_index_metadata(domain, metadata.to_dict())

        stats = VectorValidator.generate_statistics(vectors.tolist())
        # We might want to aggregate statistics across the whole index, but for now we'll just save the latest document's stats
        # or calculate total stats using store.index.ntotal
        stats["total_vectors_in_index"] = store.index.ntotal
        self.storage.save_statistics(domain, stats)

        # Update registry version (increment if index already existed)
        new_version = (self.registry.get(domain, {}).get("version", 0) + 1)
        self._update_registry(domain, new_version, self.storage._faiss_dir(domain))
