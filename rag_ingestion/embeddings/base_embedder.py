# src/embeddings/base_embedder.py
"""Abstract base class for embedding models.
All embedder implementations must inherit from this class and implement the
`embed_chunks` method. The `embed_query` method is defined for future retrieval
layers but may remain unimplemented for now.
"""

import abc
from typing import List
import numpy as np

from rag_ingestion.chunking.chunk_metadata import Chunk
from rag_ingestion.config.config import config


class BaseEmbedder(abc.ABC):
    """Base interface for embedding models.

    * `embed_chunks(chunks)`: Accepts a list of :class:`Chunk` objects and returns
      a 2‑D ``np.ndarray`` where each row corresponds to a chunk embedding.
    * `embed_query(text)`: Stub for future query embedding – raises
      ``NotImplementedError`` by default.
    """

    def __init__(self):
        self._model = None
        self._dimension = None

    @property
    def dimension(self):
        self._load_model_if_needed()
        return self._dimension

    @abc.abstractmethod
    def _load_model(self):
        """Load the underlying model and set ``self._model`` and ``self._dimension``.
        Implementations should perform lazy loading – only called once.
        """

    def _load_model_if_needed(self):
        if self._model is None:
            self._load_model()
            if hasattr(self._model, "get_sentence_embedding_dimension"):
                self._dimension = self._model.get_sentence_embedding_dimension()
            else:
                # Fallback: infer from a single embedding
                test_vec = self._model.encode(["test"])
                self._dimension = test_vec.shape[1]

    def embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        """Generate embeddings for a batch of chunks.
        Returns a NumPy matrix of shape ``(len(chunks), dimension)``.
        """
        self._load_model_if_needed()
        texts = [c.text for c in chunks]
        # Batch processing – concrete class decides the actual batch size handling
        embeddings = self._embed_batch(texts)
        if config.normalize_embeddings:
            embeddings = self._normalize_vectors(embeddings)
        return embeddings

    @abc.abstractmethod
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into a NumPy matrix. Must respect batch size.
        """

    def embed_query(self, text: str) -> np.ndarray:
        """Encode a single query string. Not required for Phase 3B.
        Sub‑classes may override when retrieval is added.
        """
        raise NotImplementedError("Query embedding is not implemented in Phase 3B")

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """L2‑normalize each row vector.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero – if norm is zero, leave as zero vector
        norms[norms == 0] = 1.0
        return vectors / norms
