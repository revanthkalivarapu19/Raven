# src/crawler/base_crawler.py
"""Abstract base class for all crawlers.
Provides aiohttp session handling, retry logic, URL normalization, duplicate
tracking, and common helper methods.
"""

import asyncio
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Set, Tuple, Optional

import aiohttp
from aiohttp import ClientTimeout, TCPConnector

from rag_ingestion.utils.logger import get_logger

class BaseCrawler(ABC):
    def __init__(
        self,
        domain: str,
        source_name: str,
        config: dict,
        *,
        max_concurrency: int = 100,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        timeout: int = 30,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    ):
        self.domain = domain
        self.source_name = source_name
        self.config = config
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self.user_agent = user_agent
        self.logger = get_logger(self.__class__.__name__)
        self._seen_urls: Set[str] = set()
        self._seen_hashes: Set[str] = set()
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        timeout = ClientTimeout(total=self.timeout)
        connector = TCPConnector(limit=self.max_concurrency, ssl=False)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"User-Agent": self.user_agent},
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------
    def normalize_url(self, url: str) -> str:
        """Return a canonical version of *url* (remove fragments, sort query).
        ``aiohttp`` provides a URL class that handles most normalisation.
        """
        parsed = aiohttp.client_reqrep.URL(url)
        return str(parsed.with_fragment(None))

    def is_duplicate_url(self, url: str) -> bool:
        norm = self.normalize_url(url)
        if norm in self._seen_urls:
            return True
        self._seen_urls.add(norm)
        return False

    def compute_sha256(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    async def fetch_with_retry(self, url: str) -> Tuple[Optional[bytes], Optional[str]]:
        """Fetch *url* with exponential back‑off.
        Returns ``(content, final_url)`` or ``(None, None)`` on failure.
        """
        attempt = 0
        wait = 1.0
        while attempt <= self.max_retries:
            try:
                async with self._semaphore:
                    async with self.session.get(url, allow_redirects=True) as resp:
                        # Diagnostic logging of request/response metadata (no behavior change)
                        user_agent = self.session.headers.get('User-Agent')
                        content_type = resp.headers.get('Content-Type')
                        location = resp.headers.get('Location')
                        x_robots = resp.headers.get('X-Robots-Tag')
                        self.logger.info(
                            "Crawler GET response",
                            extra={
                                "url": url,
                                "status": resp.status,
                                "content_type": content_type,
                                "user_agent": user_agent,
                                "location": location,
                                "x_robots_tag": x_robots,
                            },
                        )
                        if resp.status == 200:
                            content = await resp.read()
                            return content, str(resp.url)
                        self.logger.warning(
                            "Non‑200 response while fetching",
                            extra={"url": url, "status": resp.status},
                        )
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                self.logger.warning(
                    "Network error while fetching",
                    extra={"url": url, "error": str(exc)},
                )
            attempt += 1
            await asyncio.sleep(wait)
            wait *= self.backoff_factor
        self.logger.error("Exceeded maximum retries", extra={"url": url})
        return None, None

    @abstractmethod
    async def crawl(self) -> Set[str]:
        """Entry point for crawling. Returns a set of discovered document URLs."""
        raise NotImplementedError
