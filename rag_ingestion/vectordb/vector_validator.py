import json
from pathlib import Path
from datetime import datetime

from .vector_metadata import IndexMetadata

class VectorValidator:
    """Validate FAISS index metadata and vector dimensions.

    Used by ``IndexManager`` before persisting a new index.
    """

    @staticmethod
    def validate_metadata(metadata: IndexMetadata, expected_model: str, expected_dim: int) -> bool:
        """Check that the metadata matches the current embedding config.

        Returns ``True`` if valid, otherwise raises ``ValueError``.
        """
        if metadata.model_name != expected_model:
            raise ValueError(
                f"Embedding model mismatch: index uses {metadata.model_name}, "
                f"but current config expects {expected_model}"
            )
        if metadata.dimension != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: index dimension {metadata.dimension}, "
                f"config expects {expected_dim}"
            )
        return True

    @staticmethod
    def validate_index_file(index_path: Path) -> bool:
        """Simple sanity check that a FAISS index file can be loaded.

        Returns ``True`` if loading succeeds; raises ``IOError`` otherwise.
        """
        import faiss
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found at {index_path}")
        # Attempt to read the index; this will raise if corrupted.
        try:
            faiss.read_index(str(index_path))
        except Exception as exc:
            raise IOError(f"Failed to load FAISS index at {index_path}: {exc}")
        return True

    @staticmethod
    def generate_statistics(vectors: list[list[float]]) -> dict:
        """Return simple statistics for a batch of vectors.

        Includes ``dimension``, ``total_vectors`` and ``average_norm``.
        """
        import numpy as np
        if not vectors:
            return {"dimension": 0, "total_vectors": 0, "average_norm": 0.0}
        arr = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(arr, axis=1)
        return {
            "dimension": arr.shape[1],
            "total_vectors": arr.shape[0],
            "average_norm": float(np.mean(norms)),
        }
