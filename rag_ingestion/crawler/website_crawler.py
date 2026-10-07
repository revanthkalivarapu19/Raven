# src/crawler/website_crawler.py
"""Crawler that starts from a source's base URL, respects robots.txt, and discovers
internal links up to a configurable depth. Returns a set of document URLs suitable
for further download.
"""

import asyncio
from urllib.parse import urlparse, urljoin
from typing import Set, List, Tuple

from bs4 import BeautifulSoup

from rag_ingestion.crawler.base_crawler import BaseCrawler
from rag_ingestion.crawler.robots import fetch_robots_txt, is_allowed

class WebsiteCrawler(BaseCrawler):
    async def crawl(self) -> Set[str]:
        base_url: str = self.config["base_url"]
        crawl_depth: int = int(self.config.get("crawl_depth", 2))
        allowed_domains = {urlparse(base_url).netloc}

        # Fetch and parse robots.txt
        rp = await fetch_robots_txt(self.session, base_url)
        self.logger.info("Fetched robots.txt", extra={"url": base_url})

        discovered: Set[str] = set()
        to_visit: List[Tuple[str, int]] = [(base_url, 0)]

        while to_visit:
            current_url, depth = to_visit.pop(0)
            norm_url = self.normalize_url(current_url)
            if self.is_duplicate_url(norm_url):
                self.logger.info("Duplicate URL skipped", extra={"url": norm_url})
                continue
            if not is_allowed(rp, self.user_agent, norm_url):
                self.logger.info("Disallowed by robots.txt", extra={"url": norm_url})
                continue
            self.logger.info("Fetching page", extra={"url": norm_url, "depth": depth})
            content, final_url = await self.fetch_with_retry(norm_url)
            if not content:
                continue
            soup = BeautifulSoup(content, "html.parser")
            for anchor in soup.find_all("a", href=True):
                href = anchor["href"]
                absolute = urljoin(final_url, href)
                parsed = urlparse(absolute)
                if parsed.netloc not in allowed_domains:
                    continue  # external domain – ignore
                if self._is_supported_document(absolute):
                    discovered.add(self.normalize_url(absolute))
                else:
                    # Extension not sufficient; perform lightweight HEAD request to inspect Content-Type
                    content_type = await self._head_content_type(absolute)
                    if content_type and self._is_supported_by_content_type(content_type, absolute):
                        discovered.add(self.normalize_url(absolute))
                    elif depth < crawl_depth:
                        to_visit.append((self.normalize_url(absolute), depth + 1))
        return discovered

    def _is_supported_document(self, url: str) -> bool:
        """Return True if the URL has a supported file extension.

        Supported extensions are retained for backward compatibility.
        """
        lowered = url.lower()
        return any(lowered.endswith(ext) for ext in (".pdf", ".html", ".htm", ".txt", ".md"))

    async def _head_content_type(self, url: str) -> str:
        """Perform a HEAD request to obtain the Content-Type header.

        Returns the normalized MIME type (e.g., "text/html") or an empty string on failure.
        """
        try:
            async with self._semaphore:
                async with self.session.head(url, allow_redirects=True) as resp:
                    if resp.status == 200:
                        ct = resp.headers.get("Content-Type", "")
                        # Normalize: remove parameters, lowercase, strip whitespace
                        return ct.split(";")[0].lower().strip()
        except Exception as exc:
            self.logger.warning("HEAD request failed for URL", extra={"url": url, "error": str(exc)})
        return ""

    def _is_supported_by_content_type(self, content_type: str, url: str) -> bool:
        """Determine support based on MIME type, falling back to extension check.

        Supported MIME types: text/html, text/plain, text/markdown, application/pdf.
        """
        if not content_type:
            return False
        supported_mime = {"text/html", "text/plain", "text/markdown", "application/pdf"}
        if content_type in supported_mime:
            return True
        # Fallback to extension detection for robustness (e.g., missing or incorrect MIME)
        return self._is_supported_document(url)
