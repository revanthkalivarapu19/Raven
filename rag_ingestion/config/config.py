# src/config/config.py
"""Configuration loader for the RAG ingestion project.

This module reads environment variables (via ``python-dotenv``) and all JSON
source definition files under ``config/``. It validates the JSON structure
against the expected schema and provides helper functions for accessing the
configuration.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, validator
from rag_ingestion.config.domains import canonical_domain

# Load .env file from project root if present
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# ---------- Pydantic models for source config ----------
class Source(BaseModel):
    name: str
    base_url: str
    type: str
    priority: str
    authority_score: float = 1.0
    allowed_formats: List[str]
    crawl_depth: int = 2
    enabled: bool = True

    @validator("authority_score")
    def authority_between_zero_one(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("authority_score must be between 0 and 1")
        return v

    @validator("crawl_depth")
    def depth_non_negative(cls, v):
        if v < 0:
            raise ValueError("crawl_depth must be non‑negative")
        return v

# ---------- Core configuration class ----------
class ConfigLoader:
    """Loads and validates configuration files and environment settings.

    After instantiation the class holds a dictionary ``domains`` where the key
    is the domain name (e.g. ``medical``) and the value is a list of ``Source``
    objects.
    """

    def __init__(self, config_dir: Path | str | None = None):
        self.config_dir = Path(config_dir) if config_dir else PROJECT_ROOT / "config"
        self.domains: Dict[str, List[Source]] = {}

        # Pipeline Configuration Defaults
        self.max_file_size_bytes = int(self.get_env("MAX_FILE_SIZE_BYTES", str(50 * 1024 * 1024))) # 50 MB
        self.timeout_seconds = int(self.get_env("TIMEOUT_SECONDS", "30"))
        self.retry_count = int(self.get_env("RETRY_COUNT", "3"))
        self.supported_document_types = ["html", "pdf", "txt", "md", "htm"]
        self.max_concurrency = int(self.get_env("MAX_CONCURRENCY", "10"))

        # Storage Paths
        self.data_dir = Path(self.get_env("DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser()
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.metadata_dir = self.data_dir / "metadata"

        # Chunking Configuration Defaults
        self.max_chunk_size = int(self.get_env("MAX_CHUNK_SIZE", "500")) # Words
        self.min_chunk_size = int(self.get_env("MIN_CHUNK_SIZE", "50"))
        self.chunk_overlap = int(self.get_env("CHUNK_OVERLAP", "100"))
        # Embedding Configuration Defaults
        self.embedding_model = self.get_env("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        self.embedding_batch_size = int(self.get_env("EMBEDDING_BATCH_SIZE", "32"))
        self.normalize_embeddings = self.get_env("NORMALIZE_EMBEDDINGS", "True").lower() == "true"

        self._load_all()

    # ------------------------------------------------------------------
    def _load_all(self) -> None:
        """Iterate over JSON files in ``self.config_dir`` and populate ``self.domains``.
        """
        if not self.config_dir.is_dir():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")
        for json_file in self.config_dir.glob("*_sources.json"):
            raw_domain_key = json_file.stem.replace("_sources", "")
            domain_key = canonical_domain(raw_domain_key)
            with json_file.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            # Expect a top‑level key like "medical_sources"
            list_key = f"{raw_domain_key}_sources"
            if list_key not in raw:
                raise ValueError(f"Expected key '{list_key}' in {json_file.name}")
            sources_raw = raw[list_key]
            sources: List[Source] = []
            for src in sources_raw:
                try:
                    source_obj = Source(**src)
                    sources.append(source_obj)
                except ValidationError as exc:
                    raise ValueError(f"Invalid source definition in {json_file.name}: {exc}")
            self.domains[domain_key] = sources

    # ------------------------------------------------------------------
    # Helper public API ------------------------------------------------
    def get_enabled_sources(self) -> List[Source]:
        """Return a flat list of all enabled sources across domains."""
        return [s for lst in self.domains.values() for s in lst if s.enabled]

    def get_domain_sources(self, domain: str) -> List[Source]:
        """Return the list of sources for a specific domain (e.g. ``"medical"``).
        Raises ``KeyError`` if the domain does not exist.
        """
        return self.domains[canonical_domain(domain)]

    def load_sources(self, domain: str) -> List[Source]:
        """Alias for ``get_domain_sources`` – kept for backward compatibility.
        """
        return self.get_domain_sources(domain)

    # ------------------------------------------------------------------
    # Environment helpers ------------------------------------------------
    @staticmethod
    def get_env(var_name: str, default: str | None = None) -> str | None:
        """Retrieve an environment variable, optionally providing a default.
        """
        return os.getenv(var_name, default)

# Instantiate a singleton for convenient import elsewhere
config = ConfigLoader()
