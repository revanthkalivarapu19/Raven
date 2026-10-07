from typing import List, Tuple

from rag_ingestion.pipeline.storage_manager import StorageManager
from rag_ingestion.embeddings.base_embedder import BaseEmbedder

class FAISSQueryService:
    """Service that performs FAISS similarity search for a query claim.

    It loads the domain-specific FAISS index via ``StorageManager`` and uses the
    provided ``embedder`` to obtain a query embedding. The ``search`` method
    returns a list of ``(chunk_id, similarity_score)`` tuples ordered by score
    descending.
    """

    def __init__(self, storage_manager: StorageManager, embedder: BaseEmbedder):
        self.storage_manager = storage_manager
        self.embedder = embedder

    def run_query(self, claim: str, domain: str, top_k: int = 5) -> List[Tuple[str, float]]:
        # Normalise domain name
        norm_domain = domain.lower()
        # Load FAISS index and lookup mapping
        index, lookup = self.storage_manager.load_faiss_store(
            norm_domain, expected_dimension=self.embedder.dimension
        )
        # Embed the query
        query_vec = self.embedder.embed_query(claim)
        # Perform search using the FAISSStore interface (index is a faiss.Index)
        # We instantiate a temporary FAISSStore wrapper to reuse its search method
        from rag_ingestion.vectordb.faiss_store import FAISSStore
        faiss_store = FAISSStore(dimension=self.embedder.dimension)
        # Inject loaded index
        faiss_store.index = index

        ids, scores = faiss_store.search(query_vec, top_k)
        # Map FAISS internal ids to chunk identifiers using the lookup dict
        idx_to_id = lookup.get("idx_to_id", {})
        results: List[Tuple[str, float]] = []
        for idx, score in zip(ids.tolist(), scores.tolist()):
            chunk_id = idx_to_id.get(str(idx))
            if chunk_id:
                results.append((chunk_id, score))
        # Already ordered by FAISS (highest score first)
        return results
