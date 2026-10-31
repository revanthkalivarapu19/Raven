# src/embeddings/bge_embedder.py
"""Concrete embedder using BAAI/bge-large-en-v1.5 via sentence-transformers.
The model is loaded lazily and respects the configured batch size.
"""

from typing import List
import threading
import numpy as np

# Optional import of SentenceTransformer; load failure is explicit in _load_model.
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

from rag_ingestion.embeddings.base_embedder import BaseEmbedder
from rag_ingestion.config.config import config


class BGEModelLoadError(RuntimeError):
    """Raised when the configured BGE model cannot be loaded."""


_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()
BGE_INITIALIZATION_COUNT = 0


class BGEEmbedder(BaseEmbedder):
    """Embedder for BGE large model.

    It loads the model on first use, reads the dimension dynamically, and
    processes texts in batches defined by ``config.embedding_batch_size``.
    """

    def _load_model(self):
        """Load the configured BGE model or fail explicitly."""
        global BGE_INITIALIZATION_COUNT
        try:
            if SentenceTransformer is None:
                raise ImportError("SentenceTransformer not installed")

            cache_key = (
                config.embedding_model,
                config.embedding_local_files_only,
                id(SentenceTransformer),
            )
            with _MODEL_CACHE_LOCK:
                model = _MODEL_CACHE.get(cache_key)
                if model is None:
                    model = SentenceTransformer(
                        config.embedding_model,
                        device="cpu",
                        local_files_only=config.embedding_local_files_only,
                    )
                    _MODEL_CACHE[cache_key] = model
                    BGE_INITIALIZATION_COUNT += 1

            # Verify interface
            if not callable(getattr(model, "encode", None)) or not callable(getattr(model, "get_sentence_embedding_dimension", None)):
                raise ValueError("Model does not implement required interface")
            self._model = model
        except Exception as exc:
            raise BGEModelLoadError(
                f"Unable to load embedding model '{config.embedding_model}'"
                f" (local_files_only={config.embedding_local_files_only}): {exc}"
            ) from exc
        # Dimension will be set by BaseEmbedder after loading


    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts respecting the configured batch size.
        Returns a NumPy array of shape (len(texts), dimension).
        """
        batch_size = config.embedding_batch_size
        embeddings_list = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_emb = self._model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
            embeddings_list.append(np.asarray(batch_emb))
        # Concatenate all batches
        return np.vstack(embeddings_list)
    def embed_query(self, text: str) -> np.ndarray:
        """Encode a single query string using the loaded model.

        Returns a normalized ``float32`` vector of shape ``(dimension,)``.
        Raises ``ValueError`` if the resulting vector contains NaN/Inf or
        does not match the expected dimension.
        """
        self._load_model_if_needed()
        vec = np.asarray(self._model.encode([text], normalize_embeddings=True, show_progress_bar=False))[0]
        if vec.ndim != 1:
            raise ValueError("Query embedding must be a 1-D vector")
        if vec.shape[0] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {vec.shape[0]}")
        if np.isnan(vec).any() or np.isinf(vec).any():
            raise ValueError("Embedding contains NaN or Inf values")
        return vec.astype(np.float32)
