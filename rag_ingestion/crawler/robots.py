# src/crawler/robots.py
"""Utility functions for handling robots.txt parsing and compliance.

The functions are lightweight and use the standard library ``urllib.robotparser``
module. ``aiohttp`` is used to fetch the robots.txt file asynchronously.
"""

import aiohttp
import urllib.robotparser
from urllib.parse import urljoin, urlparse
from typing import Optional

async def fetch_robots_txt(session: aiohttp.ClientSession, base_url: str) -> Optional[urllib.robotparser.RobotFileParser]:
    """Fetch and parse ``robots.txt`` for *base_url*.

    Returns a ``RobotFileParser`` instance or ``None`` if the file could not be
    retrieved (e.g., 404). The parser will treat missing robots.txt as allowing
    all URLs.
    """
    parsed = urlparse(base_url)
    robots_url = urljoin(f"{parsed.scheme}://{parsed.netloc}", "/robots.txt")
    try:
        async with session.get(robots_url, timeout=10) as resp:
            if resp.status == 200:
                txt = await resp.text()
                rp = urllib.robotparser.RobotFileParser()
                rp.set_url(robots_url)
                rp.parse(txt.splitlines())
                return rp
    except Exception:
        # Network errors or timeouts – treat as no robots.txt
        return None
    return None

import requests

def fetch_robots_txt_sync(base_url: str, timeout: int = 10) -> Optional[urllib.robotparser.RobotFileParser]:
    parsed = urlparse(base_url)
    robots_url = urljoin(f"{parsed.scheme}://{parsed.netloc}", "/robots.txt")
    try:
        resp = requests.get(robots_url, timeout=timeout)
        if resp.status_code == 200:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robots_url)
            rp.parse(resp.text.splitlines())
            return rp
    except Exception:
        return None
    return None

def is_allowed(rp: Optional[urllib.robotparser.RobotFileParser], user_agent: str, url: str) -> bool:
    """Return ``True`` if *url* is allowed for *user_agent* according to *rp*.

    If *rp* is ``None`` (no robots.txt) the function returns ``True``.
    """
    if rp is None:
        return True
    return rp.can_fetch(user_agent, url)
