# src/utils/helpers.py
"""Utility helper functions used across the project.

All helpers are deliberately lightweight and have no external dependencies so
that they can be imported early (e.g. during configuration loading) without
triggering heavy imports.
"""

import hashlib
import os
import re
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Union

# ---------------------------------------------------------------------------
def sha256_hash(text: str) -> str:
    """Return the SHA‑256 hexadecimal digest of *text*.

    The function accepts a Unicode string, encodes it as UTF‑8 and returns the
    lower‑case 64‑character hex digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
def normalize_url(url: str) -> str:
    """Normalize a URL for consistent storage and duplicate detection.

    The normalization performed:
    * Strip surrounding whitespace
    * Ensure the scheme is lower‑cased (default to ``https`` if missing)
    * Remove default ports (80 for http, 443 for https)
    * Percent‑encode unsafe characters using ``urllib.parse.quote``
    * Remove fragment identifiers
    """
    url = url.strip()
    parsed = urllib.parse.urlparse(url, scheme="https")
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    # Remove default ports
    if (scheme == "http" and netloc.endswith(":80")) or (scheme == "https" and netloc.endswith(":443")):
        netloc = netloc.rsplit(":", 1)[0]
    path = urllib.parse.quote(parsed.path or "/", safe="/-%._~")
    query = urllib.parse.quote_plus(parsed.query, safe="=&")
    normalized = urllib.parse.urlunparse((scheme, netloc, path, "", query, ""))
    return normalized

# ---------------------------------------------------------------------------
def duplicate_hash(content: Union[str, bytes]) -> str:
    """Generate a SHA‑256 hash suitable for duplicate detection.

    * If *content* is a string it is encoded as UTF‑8.
    * If *content* is ``bytes`` it is used directly.
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()

# ---------------------------------------------------------------------------
def iso8601_timestamp(dt: datetime | None = None) -> str:
    """Return an ISO‑8601 formatted timestamp (UTC) for logging or metadata.

    If *dt* is ``None`` the current UTC time is used.
    """
    dt = dt or datetime.utcnow()
    return dt.replace(microsecond=0).isoformat() + "Z"

# ---------------------------------------------------------------------------
def sanitize_filename(name: str, max_length: int = 255) -> str:
    """Return a filesystem‑safe filename derived from *name*.

    The function:
    * Replaces any character that is not alphanumeric, dash, underscore or dot
      with an underscore.
    * Truncates the result to *max_length* characters while preserving the file
      extension if present.
    """
    # Preserve extension if there is one
    stem, ext = os.path.splitext(name)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)
    safe_ext = re.sub(r"[^A-Za-z0-9._-]", "_", ext)
    result = f"{safe_stem}{safe_ext}"
    if len(result) > max_length:
        # Keep the extension and truncate the stem
        ext_len = len(safe_ext)
        result = result[: max_length - ext_len] + safe_ext
    return result

# ---------------------------------------------------------------------------
def ensure_dir(path: Union[str, Path]) -> Path:
    """Create *path* as a directory if it does not already exist.

    Returns the absolute ``Path`` object.
    """
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

# The module's ``__all__`` makes static analysis tools aware of the public API.
__all__ = [
    "sha256_hash",
    "normalize_url",
    "duplicate_hash",
    "iso8601_timestamp",
    "sanitize_filename",
    "ensure_dir",
]
