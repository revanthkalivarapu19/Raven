# src/pipeline/download_pipeline.py
import asyncio
from typing import List, Tuple, Dict, Any, Set
from pathlib import Path

from rag_ingestion.config.config import config, Source
from rag_ingestion.crawler.website_crawler import WebsiteCrawler
from rag_ingestion.crawler.html_crawler import HTMLCrawler
from rag_ingestion.crawler.pdf_crawler import PDFCrawler
from rag_ingestion.pipeline.storage_manager import StorageManager
from rag_ingestion.utils.logger import get_crawler_logger

class DownloadPipeline:
    """Orchestrates the discovery and downloading of documents."""

    def __init__(self, storage_manager: StorageManager):
        self.storage = storage_manager
        self.logger = get_crawler_logger(self.__class__.__name__)

    async def run(self, sources: List[Source]) -> List[Tuple[Path, dict]]:
        """Run the download pipeline for the given sources.

        Returns:
            List of tuples containing (raw_file_path, source_metadata).
        """
        all_results = []

        for source in sources:
            try:
                self.logger.info("Starting download for source", extra={"source": source.name})

                # 1. Instantiate WebsiteCrawler and discover URLs
                crawler = WebsiteCrawler(
                    domain=source.name,
                    source_name=source.name,
                    config={"base_url": source.base_url, "crawl_depth": source.crawl_depth},
                    max_concurrency=config.max_concurrency,
                    timeout=config.timeout_seconds,
                    max_retries=config.retry_count
                )

                async with crawler:
                    discovered_urls = await crawler.crawl()

                self.logger.info("Discovered URLs", extra={"count": len(discovered_urls), "source": source.name})

                # 2. Setup downloaders
                html_downloader = HTMLCrawler(
                    domain=source.name, source_name=source.name, config={},
                    max_concurrency=config.max_concurrency, timeout=config.timeout_seconds, max_retries=config.retry_count
                )
                pdf_downloader = PDFCrawler(
                    domain=source.name, source_name=source.name, config={},
                    max_concurrency=config.max_concurrency, timeout=config.timeout_seconds, max_retries=config.retry_count
                )

                # 3. Download URLs
                async with html_downloader:
                    async with pdf_downloader:
                        for url in discovered_urls:
                            result = None
                            ext = url.split('.')[-1].lower() if '.' in url else 'html'

                            # Route to appropriate crawler
                            if ext == 'pdf':
                                result = await pdf_downloader.download_pdf(url)
                            else:
                                result = await html_downloader.download_page(url)
                                ext = 'html' # fallback logic

                            if not result:
                                continue

                            content, final_url, sha256 = result

                            # 4. Duplicate Detection (Persistent Hash)
                            if self.storage.exists_hash(sha256):
                                self.logger.info("Skipping persistent duplicate", extra={"url": final_url, "hash": sha256})
                                continue

                            # 5. Save raw file via StorageManager
                            raw_filename = f"{sha256}.{ext}"
                            raw_path = self.storage.save_raw(content, raw_filename)

                            metadata = {
                                "title": "", # Will be extracted later
                                "source": source.name,
                                "domain": source.base_url,
                                "url": final_url,
                                "document_type": ext,
                                "authority_score": source.authority_score,
                                "language": "en" # Default
                            }

                            all_results.append((raw_path, metadata))

            except Exception as e:
                self.logger.error("Error processing source", extra={"source": source.name, "error": str(e)})
                continue # Never stop pipeline because one source fails

        return all_results
