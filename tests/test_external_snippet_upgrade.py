from datetime import datetime
from types import SimpleNamespace

from rag_ingestion.external_retrieval.evidence_normalizer import (
    ExternalEvidenceNormalizer,
)
from rag_ingestion.external_retrieval.source_schema import ExternalEvidence
from rag_ingestion.external_retrieval.web_retriever import WebRetrievalResult
from rag_ingestion.pipeline.claim_processing import ClaimProcessingPipeline
from rag_ingestion.retrieval.evidence_fusion import EvidenceFusionManager
from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.llm_resilience import request_deadline
import time


def _snippet_evidence(url="https://example.com/article", evidence_id="news-1"):
    return Evidence(
        evidence_id=evidence_id,
        chunk_id=evidence_id,
        document_id=evidence_id,
        text="A short provider summary.",
        domain="general",
        source="Example News",
        language="en",
        retrieved_date=datetime.utcnow(),
        retrieval_method="external",
        metadata={
            "content_quality": "metadata_or_snippet",
            "provider": "news_api",
            "text_limitation": "truncated",
        },
        url=url,
        title="Reserve Bank rate decision",
    )


def _pipeline(web_retriever):
    pipeline = ClaimProcessingPipeline.__new__(ClaimProcessingPipeline)
    pipeline.fusion_manager = EvidenceFusionManager()
    pipeline.web_retriever = web_retriever
    pipeline.normalizer = ExternalEvidenceNormalizer()
    return pipeline


def test_metadata_snippet_is_upgraded_with_full_page_content():
    page_text = (
        "The Reserve Bank of India reduced the repo rate by 0.50 percentage "
        "points on Friday. " * 5
    )
    web = SimpleNamespace(
        retrieve=lambda url: SimpleNamespace(
            text=page_text, error=None, final_url=url
        )
    )
    pipeline = _pipeline(web)

    [upgraded] = pipeline._upgrade_metadata_snippets(
        "The Reserve Bank of India reduced the repo rate by 0.50% on Friday.",
        [_snippet_evidence()],
    )

    assert upgraded.text == page_text.strip()
    assert upgraded.metadata["content_quality"] == "source_text"
    assert upgraded.metadata["full_page_extracted"] is True
    assert upgraded.metadata["provider"] == "news_api"
    assert upgraded.metadata["text_limitation"] == "truncated"


def test_failed_fetch_keeps_original_snippet():
    original = _snippet_evidence()
    web = SimpleNamespace(
        retrieve=lambda url: SimpleNamespace(
            text=None, error="Request timed out", final_url=url
        )
    )

    [result] = _pipeline(web)._upgrade_metadata_snippets("rate cut", [original])

    assert result == original


def test_normalizer_marks_fetched_page_as_source_text_and_keeps_metadata():
    external = ExternalEvidence(
        evidence_id="news-2",
        text="A short provider summary.",
        source="Example News",
        title="Article title",
        url="https://example.com/article",
        domain="general",
        metadata={
            "content_quality": "metadata_or_snippet",
            "provider": "news_api",
        },
    )
    page_text = "Extracted article body. " * 5
    web_result = WebRetrievalResult(
        url=external.url,
        final_url=external.url,
        status_code=200,
        content_type="text/html",
        text=page_text,
        retrieved_date=datetime.utcnow(),
    )

    normalized = ExternalEvidenceNormalizer().normalize(external, web_result)

    assert normalized.text == page_text.strip()
    assert normalized.metadata["content_quality"] == "source_text"
    assert normalized.metadata["provider"] == "news_api"
    assert normalized.metadata["full_page_extracted"] is True


def test_unsafe_url_is_rejected_before_http_request(monkeypatch):
    from rag_ingestion.external_retrieval.web_retriever import WebSourceRetriever

    def fail_if_called(*args, **kwargs):
        raise AssertionError("unsafe URL must be rejected before network access")

    monkeypatch.setattr(
        "rag_ingestion.external_retrieval.web_retriever.fetch_robots_txt_sync",
        fail_if_called,
    )
    monkeypatch.setattr(
        "rag_ingestion.external_retrieval.web_retriever.requests.get",
        fail_if_called,
    )

    result = WebSourceRetriever().retrieve("http://127.0.0.1/private")

    assert result.error == "URL is invalid or does not resolve exclusively to public addresses"


def test_deadline_guard_skips_page_fetch():
    calls = []
    web = SimpleNamespace(
        retrieve=lambda url: calls.append(url)
        or SimpleNamespace(text="x" * 100, error=None, final_url=url)
    )
    original = _snippet_evidence()

    with request_deadline(time.monotonic() + 0.01):
        [result] = _pipeline(web)._upgrade_metadata_snippets("rate cut", [original])

    assert calls == []
    assert result == original


def test_page_fetch_timeout_is_bounded_by_request_deadline():
    from rag_ingestion.external_retrieval.web_retriever import WebSourceRetriever

    retriever = WebSourceRetriever(timeout=15)
    with request_deadline(time.monotonic() + 2):
        assert 0 < retriever.request_timeout() <= 2


def test_page_upgrade_fetches_at_most_one_candidate():
    calls = []
    web = SimpleNamespace(
        retrieve=lambda url: calls.append(url)
        or SimpleNamespace(text="Full article content. " * 10, error=None, final_url=url)
    )
    evidence = [
        _snippet_evidence(
            f"https://example.com/article-{index}", evidence_id=f"news-{index}"
        )
        for index in range(5)
    ]

    _pipeline(web)._upgrade_metadata_snippets("rate cut", evidence)

    assert len(calls) == 1
