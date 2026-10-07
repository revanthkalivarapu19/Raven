"""Retrieval manager that orchestrates query embedding, FAISS search, and evidence construction.

Public API:
    RetrievalManager.search(claim: str, domain: str, top_k: int = 5) -> List[Evidence]
"""

from __future__ import annotations
import json

import uuid
from datetime import datetime
from typing import List, Tuple, Any

from rag_ingestion.embeddings.bge_embedder import BGEEmbedder
from rag_ingestion.embeddings.base_embedder import BaseEmbedder
from rag_ingestion.pipeline.storage_manager import StorageManager
from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.retrieval.faiss_query_service import FAISSQueryService
from rag_ingestion.config.domains import canonical_domain


class RetrievalManager:
    """High‑level service for local evidence retrieval.

    It uses a ``BaseEmbedder`` (BGEEmbedder) to embed the claim, a ``FAISSQueryService``
    to perform similarity search, and the ``StorageManager`` to resolve chunk
    identifiers to concrete JSON files. The resulting ``Evidence`` objects contain
    all required fields and optional metadata.
    """

    def __init__(self, storage_manager: StorageManager | None = None, embedder: BaseEmbedder | None = None):
        self.storage_manager = storage_manager or StorageManager()
        self.embedder = embedder or BGEEmbedder()
        self.query_service = FAISSQueryService(self.storage_manager, self.embedder)

    def _load_chunk(self, domain: str, chunk_id: str) -> dict:
        chunk_path = self.storage_manager.find_chunk_path(domain, chunk_id)
        with open(chunk_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_evidence(self, claim: str, domain: str, chunk_dict: dict, similarity_score: float) -> Evidence:
        """Construct an :class:`Evidence` object from a chunk JSON dictionary.

        ``chunk_dict`` follows the schema produced by ``StorageManager.save_chunks`` –
        it contains the fields of :class:`rag_ingestion.chunking.chunk_metadata.Chunk` plus a
        nested ``metadata`` object.
        """
        # Extract required fields
        chunk_id = chunk_dict.get("chunk_id")
        document_id = chunk_dict.get("document_id")
        text = chunk_dict.get("text")
        language = chunk_dict.get("language")
        source = chunk_dict.get("source")
        # ``metadata`` holds the ChunkMetadata instance
        meta = chunk_dict.get("metadata", {})
        url = meta.get("url")
        title = None  # Title is not stored at chunk level; left optional
        publication_date = meta.get("publication_date")
        authority_score = meta.get("authority_score")

        evidence = Evidence(
            evidence_id=str(uuid.uuid4()),
            claim_id=None,
            chunk_id=chunk_id,
            document_id=document_id,
            text=text,
            domain=domain.lower(),
            source=source,
            language=language,
            url=url,
            title=title,
            publication_date=publication_date,
            authority_score=authority_score,
            retrieved_date=datetime.utcnow(),
            similarity_score=similarity_score,
            retrieval_method="faiss_local",
            metadata=chunk_dict,
        )
        return evidence

    def search(self, claim: str, domain: str, top_k: int = 5) -> List[Evidence]:
        """Run a retrieval query and return a ranked list of ``Evidence`` objects.

        Steps:
        1. Normalise the domain name (lower‑case) and validate that a FAISS store
           exists for it.
        2. Embed the claim using ``embed_query``.
        3. Perform a FAISS search via ``FAISSQueryService``.
        4. Resolve each returned ``chunk_id`` to its JSON representation.
        5. Build ``Evidence`` objects preserving the raw similarity score.
        """
        norm_domain = canonical_domain(domain)
        # Validate that the domain index exists – ``load_faiss_store`` will raise if missing
        _ = self.storage_manager.load_faiss_store(norm_domain)

        # Perform the search – returns list of (chunk_id, score)
        results: List[Tuple[str, float]] = self.query_service.run_query(claim, norm_domain, top_k)

        evidences: List[Evidence] = []
        for chunk_id, score in results:
            try:
                chunk_dict = self._load_chunk(norm_domain, chunk_id)
                ev = self._build_evidence(claim, norm_domain, chunk_dict, score)
                evidences.append(ev)
            except FileNotFoundError:
                # Skip missing chunks – should not happen in a consistent store
                continue
        return evidences
