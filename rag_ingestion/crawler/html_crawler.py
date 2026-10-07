# src/crawler/html_crawler.py
"""HTMLCrawler downloads raw HTML documents.
It inherits from BaseCrawler for session handling, duplicate detection and logging.
Filesystem storage has been decoupled to the StorageManager.
"""

import hashlib
from typing import Optional, Set, Tuple

from rag_ingestion.crawler.base_crawler import BaseCrawler

class HTMLCrawler(BaseCrawler):
    async def crawl(self) -> Set[str]:
        raise NotImplementedError("HTMLCrawler is intended to be used via download_page().")

    async def download_page(self, url: str) -> Optional[Tuple[bytes, str, str]]:
        """Download an HTML page, deduplicate by URL, and return content.
        Returns:
            Tuple[bytes, str, str]: (content, final_url, sha256_hash) or None if skipped/failed.
        """
        if self.is_duplicate_url(url):
            self.logger.info("Duplicate HTML URL skipped", extra={"url": url})
            return None
        content, final_url = await self.fetch_with_retry(url)
        if not content:
            return None

        sha256 = self.compute_sha256(content)

        self.logger.info(
            "HTML downloaded",
            extra={"url": final_url, "sha256": sha256},
        )
        return content, final_url, sha256
