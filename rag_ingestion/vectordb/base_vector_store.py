import abc
from pathlib import Path
from typing import Iterable, Tuple, Any

class BaseVectorStore(abc.ABC):
    """Abstract base class for vector stores.

    Implementations must provide methods for persisting the index and for
    adding vectors. Search functionality is out of scope for Phase 3C.
    """

    @abc.abstractmethod
    def add_vectors(self, ids: Iterable[Any], vectors: Iterable[Iterable[float]]) -> None:
        """Add vectors with associated identifiers to the store."""



    @abc.abstractmethod
    def get_index(self) -> Any:
        """Return the underlying index object (e.g., a faiss.Index)."""
