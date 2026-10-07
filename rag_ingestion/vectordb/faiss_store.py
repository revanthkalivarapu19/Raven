import faiss
import numpy as np
from pathlib import Path
from typing import Iterable, Tuple, Any

from .base_vector_store import BaseVectorStore

class FAISSStore(BaseVectorStore):
    """Concrete FAISS implementation.

    Uses IndexFlatIP with optional L2 normalization (handled upstream).
    """

    def __init__(self, dimension: int):
        self.dimension = int(dimension)
        self.index = faiss.IndexFlatIP(int(dimension))
        # Mapping from vector ID (string) to integer position in the index
        self.id_to_idx: dict[str, int] = {}
        self.idx_to_id: dict[int, str] = {}

    def add_vectors(self, ids: Iterable[Any], vectors: Iterable[Iterable[float]]) -> None:
        """Add a batch of vectors.

        Parameters
        ----------
        ids: iterable of hashable identifiers (e.g., chunk IDs)
        vectors: iterable of float sequences matching ``dimension``
        """
        ids_list = list(ids)
        vec_np = np.asarray(list(vectors), dtype="float32")
        if vec_np.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vec_np.shape[1]}")

        # Duplicate prevention
        new_ids = []
        new_vectors = []
        for i, vid in enumerate(ids_list):
            if str(vid) not in self.id_to_idx:
                new_ids.append(vid)
                new_vectors.append(vec_np[i])

        if not new_ids:
            return # All vectors already exist

        new_vec_np = np.array(new_vectors, dtype="float32")

        # Append vectors to FAISS index
        start = self.index.ntotal
        self.index.add(new_vec_np)
        # Record ID mappings
        for offset, vid in enumerate(new_ids):
            idx = start + offset
            self.id_to_idx[str(vid)] = idx
            self.idx_to_id[idx] = str(vid)

    def get_index(self) -> Any:
        """Return the underlying FAISS index object."""
        return self.index

    def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search the index for the nearest vectors.

        * ``query_vector`` must be a 1‑D ``float32`` array of length ``self.dimension``.
        * ``k`` is the number of results requested; it is clamped to the size of the index.
        * Returns ``(ids, scores)`` where ``ids`` are the FAISS internal integer IDs and
          ``scores`` are the raw inner‑product (cosine similarity) values.
        """
        if query_vector.ndim != 1:
            raise ValueError("Query vector must be 1‑D")
        if query_vector.shape[0] != self.dimension:
            raise ValueError(f"Query dimension mismatch: expected {self.dimension}, got {query_vector.shape[0]}")
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)
        if np.isnan(query_vector).any() or np.isinf(query_vector).any():
            raise ValueError("Query vector contains NaN or Inf")
        if self.index.ntotal == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        k = min(k, self.index.ntotal)
        query = query_vector.reshape(1, -1)
        scores, ids = self.index.search(query, k)
        return ids[0], scores[0]
