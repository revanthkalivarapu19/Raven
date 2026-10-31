import pytest

import rag_ingestion.embeddings.bge_embedder as module


class FakeModel:
    def encode(self, values, **kwargs):
        return [[0.1, 0.2] for _ in values]

    def get_sentence_embedding_dimension(self):
        return 2


def test_bge_uses_local_files_only(monkeypatch):
    module._MODEL_CACHE.clear()
    calls = []

    def load(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return FakeModel()

    monkeypatch.setattr(module, "SentenceTransformer", load)
    monkeypatch.setattr(module.config, "embedding_local_files_only", True)
    embedder = module.BGEEmbedder()
    embedder._load_model()
    assert calls == [("BAAI/bge-large-en-v1.5", {"device": "cpu", "local_files_only": True})]


def test_bge_model_identifier_is_unchanged(monkeypatch):
    module._MODEL_CACHE.clear()
    names = []
    monkeypatch.setattr(module, "SentenceTransformer", lambda name, **kwargs: names.append(name) or FakeModel())
    monkeypatch.setattr(module.config, "embedding_local_files_only", True)
    module.BGEEmbedder()._load_model()
    assert names == ["BAAI/bge-large-en-v1.5"]


def test_bge_local_only_failure_is_actionable(monkeypatch):
    module._MODEL_CACHE.clear()
    monkeypatch.setattr(module.config, "embedding_local_files_only", True)
    monkeypatch.setattr(module, "SentenceTransformer", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("cached files missing")))
    with pytest.raises(module.BGEModelLoadError, match=r"local_files_only=True"):
        module.BGEEmbedder()._load_model()
