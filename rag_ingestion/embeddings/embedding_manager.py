# src/embeddings/embedding_manager.py
import time
import hashlib
import numpy as np
from typing import List, Dict, Any

from rag_ingestion.config.config import config
from rag_ingestion.metadata.processed import ProcessedDocument
from rag_ingestion.chunking.chunk_metadata import Chunk
from rag_ingestion.embeddings.bge_embedder import BGEEmbedder
from rag_ingestion.embeddings.embedding_validator import EmbeddingValidator
from rag_ingestion.embeddings.embedding_metadata import DocumentEmbeddingMetadata, ChunkEmbeddingMetadata
from rag_ingestion.utils.logger import get_logger


class EmbeddingManager:
    """Manages the generation, validation, caching, and persistence of embeddings."""

    def __init__(self, storage_manager):
        self.storage = storage_manager
        self.logger = get_logger(self.__class__.__name__)
        self.embedder = BGEEmbedder()
        self.validator = None  # Instantiated later once dimension is known

    def process(self, document: ProcessedDocument, chunks: List[Chunk]) -> Dict[str, Any]:
        """Process chunks to generate document embedding matrix and metadata."""
        start_time = time.time()

        domain = document.domain
        source = document.source

        # 1. Check embedding cache
        cache = self.storage.load_embedding_cache(domain, source)

        cache_hits = 0
        cache_misses = 0

        # We need to assemble the full document matrix in the order of the chunks
        embeddings_list = []
        chunks_to_embed = []
        indices_to_embed = []

        for i, chunk in enumerate(chunks):
            chunk_hash = hashlib.md5(chunk.text.encode('utf-8')).hexdigest()
            if chunk_hash in cache:
                embeddings_list.append(np.array(cache[chunk_hash]))
                cache_hits += 1
            else:
                # Placeholder for missing embedding, will fill after batch processing
                embeddings_list.append(None)
                chunks_to_embed.append(chunk)
                indices_to_embed.append(i)
                cache_misses += 1

        # 2. Generate missing embeddings
        if chunks_to_embed:
            try:
                # The embedder will load the model dynamically if it hasn't been loaded
                new_embeddings = self.embedder.embed_chunks(chunks_to_embed)

                # Fill in the missing embeddings and update cache
                for idx, chunk, emb in zip(indices_to_embed, chunks_to_embed, new_embeddings):
                    embeddings_list[idx] = emb
                    chunk_hash = hashlib.md5(chunk.text.encode('utf-8')).hexdigest()
                    cache[chunk_hash] = emb.tolist()

                # Save updated cache
                self.storage.save_embedding_cache(domain, source, cache)
            except Exception as e:
                self.logger.error("Failed to generate embeddings", extra={"doc_id": document.document_id, "error": str(e)})
                raise

        if not embeddings_list:
            self.logger.warning("No chunks provided to embed", extra={"doc_id": document.document_id})
            return {"status": "skipped", "reason": "No chunks"}

        # 3. Assemble document matrix
        matrix = np.vstack(embeddings_list)

        # 4. Validate matrix
        dimension = self.embedder.dimension
        if self.validator is None:
            self.validator = EmbeddingValidator(dimension)

        try:
            self.validator.validate(matrix)
        except ValueError as e:
            self.logger.error("Embedding validation failed", extra={"doc_id": document.document_id, "error": str(e)})
            raise

        # 5. Generate embedding metadata
        chunk_metadatas = []
        for chunk in chunks:
            cmd = ChunkEmbeddingMetadata(
                document_id=document.document_id,
                chunk_id=chunk.chunk_id,
                embedding_id=chunk.chunk_id,
                embedding_model=config.embedding_model,
                model_version="v1.0",
                embedding_dimension=dimension,
                normalized=config.normalize_embeddings,
                domain=domain,
                source=source,
                language=document.language or "en"
            )
            chunk_metadatas.append(cmd)

        doc_metadata = DocumentEmbeddingMetadata(
            document_id=document.document_id,
            embedding_model=config.embedding_model,
            model_version="v1.0",
            embedding_dimension=dimension,
            normalized=config.normalize_embeddings,
            total_chunks=len(chunks),
            average_vector_norm=float(np.mean(np.linalg.norm(matrix, axis=1))),
            chunk_metadata=chunk_metadatas
        )

        # 6. Persist matrix and metadata
        self.storage.save_embedding_matrix(domain, source, document.document_id, matrix)
        self.storage.save_embedding_metadata(domain, source, document.document_id, doc_metadata.model_dump(mode='json'))

        proc_time = time.time() - start_time
        self.logger.info("Embeddings generated", extra={
            "doc_id": document.document_id,
            "total_chunks": len(chunks),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "time": proc_time
        })

        # 7. Return statistics
        return {
            "document_id": document.document_id,
            "total_chunks": len(chunks),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "dimension": dimension,
            "processing_time": proc_time
        }
