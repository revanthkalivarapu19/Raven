# src/pipeline/storage_manager.py
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import faiss
import numpy as np
from rag_ingestion.config.config import config
from rag_ingestion.metadata.processed import ProcessedDocument
from rag_ingestion.utils.logger import get_ingestion_logger

class StorageManager:
    """Manages all filesystem operations for the ingestion pipeline.
    Handles storing raw files, processed text, metadata, and maintaining the deduplication index.
    """

    def __init__(self):
        self.logger = get_ingestion_logger(self.__class__.__name__)

        self.raw_dir = config.raw_dir
        self.processed_dir = config.processed_dir
        self.metadata_dir = config.metadata_dir
        # New directories for embeddings and vector stores
        self.embeddings_dir = config.data_dir / "embeddings"
        self.vector_store_root = config.data_dir / "vector_store"

        self.index_file = self.metadata_dir / "ingestion_index.json"

        # In-memory index of {content_hash: {metadata_dict}}
        self._index: Dict[str, dict] = {}

        self.initialize_folders()
        self.load_index()

    def initialize_folders(self) -> None:
        """Create necessary directories if they do not exist."""
        for d in [self.raw_dir, self.processed_dir, self.metadata_dir,
                  self.embeddings_dir, self.vector_store_root]:
            d.mkdir(parents=True, exist_ok=True)

    def load_index(self) -> None:
        """Load the duplicate index from disk."""
        if self.index_file.exists():
            try:
                with self.index_file.open("r", encoding="utf-8") as f:
                    self._index = json.load(f)
            except Exception as e:
                self.logger.error("Failed to load ingestion index", extra={"error": str(e)})
                self._index = {}
        else:
            self._index = {}

    def save_index(self) -> None:
        """Persist the duplicate index to disk."""
        try:
            with self.index_file.open("w", encoding="utf-8") as f:
                json.dump(self._index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error("Failed to save ingestion index", extra={"error": str(e)})

    def exists_hash(self, content_hash: str) -> bool:
        """Check if a content hash already exists in the index."""
        return content_hash in self._index

    def register_hash(self, doc: ProcessedDocument, raw_file_path: Path) -> None:
        """Register a processed document into the index."""
        self._index[doc.content_hash] = {
            "document_id": doc.document_id,
            "source": doc.source,
            "url": doc.url,
            "processed_date": doc.retrieved_date.isoformat(),
            "file_path": str(raw_file_path),
            "status": "processed"
        }
        self.save_index()

    def lookup_document(self, content_hash: str) -> Optional[dict]:
        """Lookup an existing document by hash."""
        return self._index.get(content_hash)

    def save_raw(self, content: bytes, filename: str) -> Path:
        """Save a raw document to the raw directory."""
        file_path = self.raw_dir / filename
        file_path.write_bytes(content)
        return file_path

    def save_processed(self, doc: ProcessedDocument) -> Path:
        """Save the processed plain text and its corresponding metadata JSON."""
        # Save text
        text_filename = f"{doc.document_id}.txt"
        text_path = self.processed_dir / text_filename
        text_path.write_text(doc.text, encoding="utf-8")

        # Save metadata JSON (excluding text to save space)
        meta_filename = f"{doc.document_id}.json"
        meta_path = self.metadata_dir / meta_filename
        meta_dict = json.loads(doc.json())
        meta_dict.pop("text", None)

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2, ensure_ascii=False)
        return text_path

    def save_chunks(self, document: ProcessedDocument, chunks: list) -> None:
        """Save chunks and manifest to data/domain/source/chunks/."""
        import re
        domain_str = re.sub(r"[^a-zA-Z0-9_-]", "_", document.domain)
        source_str = re.sub(r"[^a-zA-Z0-9_-]", "_", document.source.lower())

        chunk_dir = config.data_dir / domain_str / source_str / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_ids = []
        for chunk in chunks:
            chunk_filename = f"{chunk.chunk_id}.json"
            chunk_path = chunk_dir / chunk_filename
            with chunk_path.open("w", encoding="utf-8") as f:
                json.dump(json.loads(chunk.json()), f, indent=2, ensure_ascii=False)
            chunk_ids.append(chunk.chunk_id)

        manifest = {
            "document_id": document.document_id,
            "total_chunks": len(chunks),
            "chunk_ids": chunk_ids
        }
        manifest_path = chunk_dir / f"{document.document_id}_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ----- Embedding persistence methods -----
    def get_embedding_cache_path(self, domain: str, source: str) -> Path:
        import re
        domain_str = re.sub(r"[^a-zA-Z0-9_-]", "_", domain)
        source_str = re.sub(r"[^a-zA-Z0-9_-]", "_", source.lower())
        return self.embeddings_dir / domain_str / source_str / "embedding_cache.json"

    def load_embedding_cache(self, domain: str, source: str) -> dict:
        """Load the embedding cache for a source."""
        cache_path = self.get_embedding_cache_path(domain, source)
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_embedding_cache(self, domain: str, source: str, cache: dict) -> None:
        """Save the embedding cache for a source."""
        cache_path = self.get_embedding_cache_path(domain, source)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)

    def save_embedding_matrix(self, domain: str, source: str, doc_id: str, matrix: np.ndarray) -> None:
        """Save embedding matrix as .npy"""
        import re
        domain_str = re.sub(r"[^a-zA-Z0-9_-]", "_", domain)
        source_str = re.sub(r"[^a-zA-Z0-9_-]", "_", source.lower())
        matrix_dir = self.embeddings_dir / domain_str / source_str / "matrices"
        matrix_dir.mkdir(parents=True, exist_ok=True)
        matrix_path = matrix_dir / f"{doc_id}.npy"
        np.save(matrix_path, matrix)

    def load_embedding_matrix(self, domain: str, source: str, doc_id: str) -> np.ndarray:
        """Load embedding matrix .npy"""
        import re
        domain_str = re.sub(r"[^a-zA-Z0-9_-]", "_", domain)
        source_str = re.sub(r"[^a-zA-Z0-9_-]", "_", source.lower())
        matrix_path = self.embeddings_dir / domain_str / source_str / "matrices" / f"{doc_id}.npy"
        if not matrix_path.exists():
            raise FileNotFoundError(f"Embedding matrix not found for {doc_id} at {matrix_path}")
        return np.load(matrix_path)

    def save_embedding_metadata(self, domain: str, source: str, doc_id: str, metadata: dict) -> None:
        """Save embedding metadata JSON."""
        import re
        domain_str = re.sub(r"[^a-zA-Z0-9_-]", "_", domain)
        source_str = re.sub(r"[^a-zA-Z0-9_-]", "_", source.lower())
        meta_dir = self.embeddings_dir / domain_str / source_str / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / f"{doc_id}.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_embedding_metadata(self, domain: str, source: str, doc_id: str) -> dict:
        """Load embedding metadata JSON."""
        import re
        domain_str = re.sub(r"[^a-zA-Z0-9_-]", "_", domain)
        source_str = re.sub(r"[^a-zA-Z0-9_-]", "_", source.lower())
        meta_path = self.embeddings_dir / domain_str / source_str / "metadata" / f"{doc_id}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Embedding metadata not found for {doc_id} at {meta_path}")
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # ----- FAISS persistence methods -----
    def _faiss_dir(self, domain: str) -> Path:
        """Directory for a domain's FAISS index."""
        return self.vector_store_root / domain.lower()

    def save_faiss_store(self, domain: str, index: Any, lookup: dict) -> None:
        """Save FAISS binary and lookup mappings for a domain."""
        domain_dir = self._faiss_dir(domain)
        domain_dir.mkdir(parents=True, exist_ok=True)

        # Save binary
        faiss_file = domain_dir / "index.faiss"
        faiss.write_index(index, str(faiss_file))

        # Save lookup
        lookup_file = domain_dir / "vector_lookup.json"
        with lookup_file.open("w", encoding="utf-8") as f:
            json.dump(lookup, f, indent=2, ensure_ascii=False)

    def load_faiss_store(self, domain: str, expected_dimension: int | None = None) -> tuple[Any, dict]:
        """Load FAISS binary and lookup mappings for a domain.
        Returns a tuple of (faiss.Index, dict).
        Raises FileNotFoundError if index.faiss does not exist.
        """
        domain_dir = self._faiss_dir(domain)
        faiss_file = domain_dir / "index.faiss"
        lookup_file = domain_dir / "vector_lookup.json"

        if not faiss_file.exists():
            raise FileNotFoundError(f"FAISS index for domain '{domain}' not found at {faiss_file}.")

        index = faiss.read_index(str(faiss_file))

        if not lookup_file.exists():
            raise FileNotFoundError(f"FAISS lookup metadata not found at {lookup_file}.")
        with lookup_file.open("r", encoding="utf-8") as f:
            lookup = json.load(f)

        metadata_file = domain_dir / "index_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"FAISS index metadata not found at {metadata_file}.")
        with metadata_file.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        if not isinstance(metadata, dict) or not isinstance(metadata.get("dimension"), int):
            raise ValueError(f"Invalid FAISS index metadata at {metadata_file}.")
        if metadata.get("model_name") != config.embedding_model:
            raise ValueError(
                f"FAISS embedding model mismatch: index uses {metadata.get('model_name')}, "
                f"current config expects {config.embedding_model}"
            )
        if metadata["dimension"] != index.d:
            raise ValueError("FAISS index dimension does not match index metadata.")
        if expected_dimension is not None and index.d != expected_dimension:
            raise ValueError(
                f"FAISS index dimension mismatch: expected {expected_dimension}, got {index.d}"
            )

        id_to_idx = lookup.get("id_to_idx")
        idx_to_id = lookup.get("idx_to_id")
        if not isinstance(id_to_idx, dict) or not isinstance(idx_to_id, dict):
            raise ValueError("FAISS lookup metadata must contain id_to_idx and idx_to_id mappings.")
        if len(id_to_idx) != index.ntotal or len(idx_to_id) != index.ntotal:
            raise ValueError("FAISS vector count does not match lookup metadata.")
        for identifier, position in id_to_idx.items():
            if str(idx_to_id.get(str(position))) != str(identifier):
                raise ValueError("FAISS lookup metadata mappings are inconsistent.")

        return index, lookup

    def save_index_metadata(self, domain: str, metadata: dict) -> None:
        """Save JSON metadata for a domain index."""
        meta_path = self._faiss_dir(domain) / "index_metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_index_metadata(self, domain: str) -> dict:
        """Load JSON metadata for a domain index."""
        meta_path = self._faiss_dir(domain) / "index_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata for domain '{domain}' not found.")
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save_statistics(self, domain: str, stats: dict) -> None:
        """Save statistics JSON for a domain index."""
        stats_path = self._faiss_dir(domain) / "statistics.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    def load_statistics(self, domain: str) -> dict:
        """Load statistics JSON for a domain index."""
        stats_path = self._faiss_dir(domain) / "statistics.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"Statistics for domain '{domain}' not found.")
        with stats_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save_index_registry(self, registry: dict) -> None:
        """Save a registry mapping domains to index directories and versions."""
        reg_path = self.vector_store_root / "index_registry.json"
        with reg_path.open("w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

    def load_index_registry(self) -> dict:
        """Load the index registry."""
        reg_path = self.vector_store_root / "index_registry.json"
        if not reg_path.exists():
            return {}
        with reg_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    def _chunk_path(self, domain: str, source: str, chunk_id: str) -> Path:
        """Return the exact path to a stored chunk JSON file.
        Used by retrieval code to locate a chunk's file.
        """
        import re
        domain_str = re.sub(r"[^a-zA-Z0-9_-]", "_", domain)
        source_str = re.sub(r"[^a-zA-Z0-9_-]", "_", source.lower())
        return config.data_dir / domain_str / source_str / "chunks" / f"{chunk_id}_chunk_0.json"

    def find_chunk_path(self, domain: str, chunk_id: str) -> Path:
        """Search all source sub‑directories under a domain for a given chunk ID.
        Returns the Path to the JSON file or raises ``FileNotFoundError``.
        """
        import re
        domain_str = re.sub(r"[^a-zA-Z0-9_-]", "_", domain)
        domain_root = config.data_dir / domain_str
        if not domain_root.is_dir():
            raise FileNotFoundError(f"Domain directory not found: {domain_root}")
        for source_dir in domain_root.iterdir():
            if not source_dir.is_dir():
                continue
            candidate = source_dir / "chunks" / f"{chunk_id}.json"
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Chunk {chunk_id} not found under domain {domain}")
