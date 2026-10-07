import json
from pathlib import Path
from typing import Dict, Any

class IndexMetadata:
    """Metadata for a domain‑wise FAISS index.

    Attributes
    ----------
    version: int
        Simple integer version (1, 2, ...)
    model_name: str
        Embedding model identifier (e.g., "BAAI/bge-large-en-v1.5").
    dimension: int
        Vector dimension.
    created_at: str
        ISO‑8601 timestamp when the index was created.
    """

    def __init__(self, version: int, model_name: str, dimension: int, created_at: str):
        self.version = version
        self.model_name = model_name
        self.dimension = dimension
        self.created_at = created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            version=data["version"],
            model_name=data["model_name"],
            dimension=data["dimension"],
            created_at=data["created_at"],
        )

class IndexRegistryEntry:
    """Registry entry tracking a domain index.

    Used by ``StorageManager`` to map domain → index directory.
    """

    def __init__(self, domain: str, index_dir: Path, version: int):
        self.domain = domain
        self.index_dir = index_dir
        self.version = version

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "index_dir": str(self.index_dir),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            domain=data["domain"],
            index_dir=Path(data["index_dir"]),
            version=data["version"],
        )
