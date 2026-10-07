# src/pipeline/pipeline_manager.py
import time
from typing import List, Optional
from rag_ingestion.config.config import config, Source
from rag_ingestion.pipeline.storage_manager import StorageManager
from rag_ingestion.pipeline.download_pipeline import DownloadPipeline
from rag_ingestion.pipeline.ingestion_pipeline import IngestionPipeline
from rag_ingestion.chunking.chunk_manager import ChunkManager
from rag_ingestion.embeddings.embedding_manager import EmbeddingManager
from rag_ingestion.vectordb.index_manager import IndexManager
from rag_ingestion.utils.logger import default_logger

class PipelineManager:
    """High-level manager connecting download and ingestion pipelines."""

    def __init__(self):
        self.storage_manager = StorageManager()
        self.download_pipeline = DownloadPipeline(self.storage_manager)

        # Phase 3A: Initialize ChunkManager
        self.chunk_manager = ChunkManager(self.storage_manager)
        # Phase 3B: Initialize EmbeddingManager
        self.embedding_manager = EmbeddingManager(self.storage_manager)
        # Phase 3C: Initialize IndexManager and inject into IngestionPipeline
        self.index_manager = IndexManager(self.storage_manager)
        self.ingestion_pipeline = IngestionPipeline(self.storage_manager, self.chunk_manager, self.embedding_manager, self.index_manager)

        self.logger = default_logger

    async def run_all(self, domain: Optional[str] = None):
        """Run the full ingestion pipeline.

        Args:
            domain: If provided, only run for sources in this domain.
                    If None, run for all enabled sources.
        """
        start_time = time.time()

        sources_to_process = []
        if domain:
            try:
                sources_to_process = config.get_domain_sources(domain)
            except KeyError:
                self.logger.error(f"Domain '{domain}' not found in configuration.")
                return
        else:
            sources_to_process = config.get_enabled_sources()

        sources_to_process = [s for s in sources_to_process if s.enabled]

        if not sources_to_process:
            self.logger.warning("No enabled sources found.")
            return

        self.logger.info("Starting Unified Ingestion Pipeline", extra={"sources_count": len(sources_to_process)})

        # 1. Download Phase
        raw_documents = await self.download_pipeline.run(sources_to_process)
        downloaded_count = len(raw_documents)

        # 2. Ingestion Phase
        processed_documents = self.ingestion_pipeline.run(raw_documents)
        extraction_success_count = len(processed_documents)
        extraction_failed_count = downloaded_count - extraction_success_count

        end_time = time.time()
        processing_time = end_time - start_time

        self._print_summary(
            sources_processed=len(sources_to_process),
            downloaded=downloaded_count,
            extraction_success=extraction_success_count,
            extraction_failed=extraction_failed_count,
            processing_time=processing_time
        )

        # Print Phase 3A Chunk Summary
        print(self.chunk_manager.get_summary(processing_time))

    def _print_summary(self, sources_processed: int, downloaded: int, extraction_success: int, extraction_failed: int, processing_time: float):
        summary = f"""
====================================
INGESTION SUMMARY
====================================
Sources Processed: {sources_processed}
Documents Discovered: (Tracked in crawler.log)
Downloaded: {downloaded}
Duplicates Skipped: (Tracked in duplicates.log)
Extraction Success: {extraction_success}
Extraction Failed: {extraction_failed}
Processing Time: {processing_time:.2f} seconds
====================================
"""
        print(summary)
        self.logger.info("Ingestion Summary", extra={
            "sources": sources_processed,
            "downloaded": downloaded,
            "success": extraction_success,
            "failed": extraction_failed,
            "time": processing_time
        })
