import requests
import pytest

from rag_ingestion.external_retrieval.gdelt_source import GDELTSource


class Response:
    def __init__(self, status=200, content_type="application/json", data=None, text=""):
        self.status_code = status
        self.headers = {"Content-Type": content_type}
        self.content = text.encode()
        self._data = data
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return next(self.responses)


def _source(responses, monkeypatch):
    source = GDELTSource(max_retries=2, retry_backoff_seconds=0)
    source.session = Session(responses)
    monkeypatch.setattr("rag_ingestion.external_retrieval.gdelt_source.time.sleep", lambda _: None)
    return source


def test_valid_json_is_converted_to_evidence(monkeypatch):
    source = _source([Response(data={"articles": [{"url": "https://example.test/a", "title": "A"}]})], monkeypatch)
    result = source.search("query")
    assert len(result) == 1
    assert result[0].title == "A"


def test_429_exhaustion_isolated_as_error(monkeypatch):
    source = _source([Response(429), Response(429), Response(429)], monkeypatch)
    with pytest.raises(RuntimeError, match="429"):
        source.search("query")


def test_429_then_success_retries(monkeypatch):
    source = _source([Response(429), Response(data={"articles": []})], monkeypatch)
    assert source.search("query") == []
    assert len(source.session.calls) == 2


def test_200_html_is_never_evidence(monkeypatch):
    source = _source([Response(content_type="text/html", text="<html>error</html>")], monkeypatch)
    with pytest.raises(RuntimeError, match="non-JSON"):
        source.search("query")


def test_200_invalid_json_is_error(monkeypatch):
    source = _source([Response(data=ValueError("bad json"))], monkeypatch)
    with pytest.raises(RuntimeError, match="Invalid JSON"):
        source.search("query")


def test_timeout_is_error(monkeypatch):
    source = _source([], monkeypatch)
    source.session = Session([])
    source.session.get = lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout("slow"))
    with pytest.raises(RuntimeError, match="timed out"):
        source.search("query")


def test_retry_count_is_bounded(monkeypatch):
    source = _source([], monkeypatch)
    source.session = Session([])
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise requests.Timeout("slow")

    source.session.get = fail
    with pytest.raises(RuntimeError):
        source.search("query")
    assert len(calls) == 1
