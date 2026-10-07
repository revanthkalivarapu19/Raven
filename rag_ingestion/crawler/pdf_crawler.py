# src/crawler/pdf_crawler.py
"""PDFCrawler downloads PDF documents, verifies integrity, deduplicates via SHA‑256.
It inherits from BaseCrawler for session handling, retry logic, duplicate detection and logging.
Filesystem storage has been decoupled to the StorageManager.
"""

import hashlib
from typing import Optional, Set, Tuple

from rag_ingestion.crawler.base_crawler import BaseCrawler

class PDFCrawler(BaseCrawler):
    async def crawl(self) -> Set[str]:
        raise NotImplementedError("PDFCrawler is intended to be used via download_pdf().")

    async def download_pdf(self, url: str) -> Optional[Tuple[bytes, str, str]]:
        """Download a PDF, check magic header, deduplicate by content hash.
        Returns:
            Tuple[bytes, str, str]: (content, final_url, sha256_hash) or None if skipped/failed.
        """
        if self.is_duplicate_url(url):
            self.logger.info("Duplicate PDF URL skipped", extra={"url": url})
            return None
        content, final_url = await self.fetch_with_retry(url)
        if not content:
            return None

        # Verify PDF header
        if not content.startswith(b"%PDF"):
            self.logger.warning("Invalid PDF file header", extra={"url": final_url})
            return None

        sha256 = self.compute_sha256(content)
        if sha256 in self._seen_hashes:
            self.logger.info("Duplicate PDF content skipped", extra={"sha256": sha256})
            return None
        self._seen_hashes.add(sha256)

        self.logger.info(
            "PDF downloaded",
            extra={"url": final_url, "sha256": sha256},
        )
        return content, final_url, sha256
