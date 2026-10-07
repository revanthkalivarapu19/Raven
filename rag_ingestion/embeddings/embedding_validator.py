# src/embeddings/embedding_validator.py
"""Validation utilities for embedding matrices.
Ensures dimensionality, NaN/Inf absence, and optional L2 normalization.
"""

import numpy as np

from rag_ingestion.config.config import config


class EmbeddingValidator:
    """Validate a NumPy embedding matrix.

    - `expected_dim` is inferred from `config.embedding_model` via the embedder.
    - Checks for NaN, infinite values.
    - If `config.normalize_embeddings` is True, verifies that each row has norm ≈ 1.0.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

    def validate(self, matrix: np.ndarray) -> bool:
        if matrix.ndim != 2:
            raise ValueError("Embedding matrix must be 2‑dimensional")
        if matrix.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {matrix.shape[1]}")
        if np.isnan(matrix).any():
            raise ValueError("Embedding matrix contains NaN values")
        if np.isinf(matrix).any():
            raise ValueError("Embedding matrix contains infinite values")
        if config.normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                raise ValueError("Embeddings are not properly L2‑normalized")
        return True
