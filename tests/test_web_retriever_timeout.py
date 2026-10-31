from types import SimpleNamespace

from rag_ingestion.external_retrieval.web_retriever import WebSourceRetriever


def test_retrieve_uses_configured_timeout_without_missing_method(monkeypatch):
    timeouts = []

    class Response:
        status_code = 200
        url = "https://example.com/article"
        headers = {"Content-Type": "text/html"}

        def raise_for_status(self):
            pass

        def close(self):
            pass

        def iter_content(self, chunk_size):
            return iter([b"<html>article</html>"])

    monkeypatch.setattr(
        "rag_ingestion.external_retrieval.web_retriever._fetch_robots_txt_pinned",
        lambda url, timeout, user_agent: timeouts.append(timeout) or object(),
    )
    monkeypatch.setattr(
        "rag_ingestion.external_retrieval.web_retriever.is_allowed",
        lambda *args: True,
    )
    monkeypatch.setattr(
        "rag_ingestion.external_retrieval.web_retriever._is_public_http_url",
        lambda url: True,
    )
    monkeypatch.setattr(
        "rag_ingestion.external_retrieval.web_retriever._validated_destination_ip",
        lambda url: "93.184.216.34",
    )
    monkeypatch.setattr(
        "rag_ingestion.external_retrieval.web_retriever._get_pinned",
        lambda url, destination_ip, **kwargs: (
            timeouts.append(kwargs["timeout"]) or Response(),
            SimpleNamespace(close=lambda: None),
        ),
    )

    retriever = WebSourceRetriever(timeout=7)
    monkeypatch.setattr(
        retriever.html_cleaner,
        "extract_from_content",
        lambda content, metadata: SimpleNamespace(text="article text"),
    )

    result = retriever.retrieve("https://example.com/article")

    assert result.error is None
    assert result.text == "article text"
    assert timeouts == [7.0, 7.0]


def test_private_redirect_target_is_rejected(monkeypatch):
    requests = []

    class Response:
        status_code = 302
        url = "https://example.com/article"
        headers = {"Location": "http://127.0.0.1/private"}

        def close(self):
            pass

    monkeypatch.setattr(
        "rag_ingestion.external_retrieval.web_retriever._is_public_http_url",
        lambda url: "127.0.0.1" not in url,
    )
    monkeypatch.setattr(
        "rag_ingestion.external_retrieval.web_retriever._fetch_robots_txt_pinned",
        lambda url, timeout, user_agent: object(),
    )
    monkeypatch.setattr(
        "rag_ingestion.external_retrieval.web_retriever.is_allowed",
        lambda *args: True,
    )
    monkeypatch.setattr(
        "rag_ingestion.external_retrieval.web_retriever._validated_destination_ip",
        lambda url: "93.184.216.34",
    )
    monkeypatch.setattr(
        "rag_ingestion.external_retrieval.web_retriever._get_pinned",
        lambda url, destination_ip, **kwargs: (
            requests.append(url) or Response(),
            SimpleNamespace(close=lambda: None),
        ),
    )

    result = WebSourceRetriever().retrieve("https://example.com/article")

    assert requests == ["https://example.com/article"]
    assert result.error == "Redirect target is not a public HTTP(S) URL"
