from __future__ import annotations

import logging
from datetime import datetime

import requests

from rag_ingestion.external_retrieval.base_source import BaseExternalSource
from rag_ingestion.external_retrieval.searxng_source import SearXNGSource
from rag_ingestion.external_retrieval.source_manager import ExternalSourceManager
from rag_ingestion.external_retrieval.source_schema import ExternalEvidence
from rag_ingestion.pipeline.claim_processing import ClaimProcessingPipeline
from rag_ingestion.retrieval.evidence_fusion import EvidenceFusionManager
from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification.verification_schema import (
    OverallVerification,
    VerificationResult,
)


CLAIM = (
    "Israeli President Isaac Herzog has tasked opposition politician Gadi Eisenkot "
    "with forming a government in a major setback for Benjamin Netanyahu."
)
RELEVANT_PAGE = (
    "Israeli President Isaac Herzog tasked opposition politician Gadi Eisenkot "
    "with forming a government, in a major setback for Benjamin Netanyahu. "
    "This report describes the political negotiations and the announcement made "
    "by the president, including the opposition leader's mandate to form a government."
)


def _candidate(evidence_id="result", url="https://news.example/story", text=RELEVANT_PAGE[:100]):
    return ExternalEvidence(
        evidence_id=evidence_id,
        title="Israeli government formation update",
        text=text,
        source="news.example",
        url=url,
        domain="political",
        source_type="searxng_search",
        metadata={
            "provider": "SearXNG",
            "publisher_site": "news.example",
            "text_limitation": "Search snippet; fetch source page before fusion.",
            "discovery_only": True,
        },
    )


class Provider(BaseExternalSource):
    supported_domains = frozenset({"political"})

    def __init__(self, name, results=None, error=None, calls=None):
        self.name = name
        self.results = results or []
        self.error = error
        self.calls = calls if calls is not None else []

    def search(self, query, **_kwargs):
        self.calls.append(query)
        if self.error:
            raise self.error
        return self.results


class EmptyLocal:
    def search(self, **_kwargs):
        return []


class Verifier:
    def __init__(self):
        self.calls = []

    def verify(self, claim, evidence_list, **_kwargs):
        self.calls.append(list(evidence_list))
        results = []
        if evidence_list:
            results.append(VerificationResult(
                evidence_id=evidence_list[0].evidence_id,
                claim=claim,
                relationship="SUPPORTS",
                confidence=0.8,
                reasoning="Test verifier result.",
                verifier="test",
                claim_aligned=True,
                directly_establishes_proposition=True,
            ))
        return OverallVerification(
            claim=claim,
            results=results,
            overall_assessment="SUPPORTED" if results else "INSUFFICIENT_EVIDENCE",
            overall_confidence=0.8 if results else 0,
        )


class Pages:
    def __init__(self, text):
        self.text = text
        self.urls = []

    def retrieve(self, url):
        from rag_ingestion.external_retrieval.web_retriever import WebRetrievalResult
        self.urls.append(url)
        return WebRetrievalResult(
            url=url,
            retrieved_date=datetime.utcnow(),
            final_url=url,
            status_code=200,
            content_type="text/html",
            text=self.text,
        )


def _pipeline(manager, pages=None, verifier=None):
    return ClaimProcessingPipeline(
        retrieval_manager=EmptyLocal(),
        external_manager=manager,
        fusion_manager=EvidenceFusionManager(),
        verification_manager=verifier or Verifier(),
        web_retriever=pages or Pages(RELEVANT_PAGE),
        minimum_sufficient_external_evidence=2,
        max_web_fallback_urls=3,
    )


def test_searxng_disabled_when_base_url_is_empty(monkeypatch, caplog):
    monkeypatch.setenv("SEARXNG_BASE_URL", "")
    source = SearXNGSource()
    with caplog.at_level(logging.INFO):
        assert source.search(CLAIM, domain="political") == []
    assert source.enabled is False
    assert "searxng enabled=false outcome=disabled" in caplog.text


def test_searxng_successful_search_requests_json_and_returns_at_most_five(monkeypatch, caplog):
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://127.0.0.1:8080/")
    source = SearXNGSource()
    request = {}

    class Response:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {"results": [
                {"title": f"Story {i}", "url": f"https://publisher{i}.example/story",
                 "content": "Search snippet for discovery only.",
                 "publishedDate": "2026-11-08T12:00:00Z",
                 "parsed_url": ["https", f"publisher{i}.example", "/story"]}
                for i in range(8)
            ]}

    def fake_get(url, **kwargs):
        request.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr("rag_ingestion.external_retrieval.searxng_source.requests.get", fake_get)
    with caplog.at_level(logging.INFO):
        results = source.search(CLAIM, domain="political", top_k=10)
    assert len(results) == 5
    assert request["url"] == "http://127.0.0.1:8080/search"
    assert request["params"] == {"q": CLAIM, "format": "json"}
    assert "headers" not in request
    assert request["timeout"] <= 5
    assert results[0].source == "publisher0.example"
    assert results[0].metadata["discovery_only"] is True
    assert results[0].publication_date == datetime(2026, 11, 8, 12)
    assert "outcome=success http_status=200" in caplog.text
    assert "result_count=5" in caplog.text


def test_searxng_zero_results_are_distinguished(monkeypatch, caplog):
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://127.0.0.1:8080")
    source = SearXNGSource()

    class Response:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {"results": []}

    monkeypatch.setattr("rag_ingestion.external_retrieval.searxng_source.requests.get", lambda *_a, **_k: Response())
    with caplog.at_level(logging.INFO):
        assert source.search(CLAIM) == []
    assert "outcome=zero_results http_status=200" in caplog.text
    assert "result_count=0" in caplog.text


def test_searxng_failure_has_safe_diagnostic(monkeypatch, caplog):
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://127.0.0.1:8080")
    source = SearXNGSource()
    monkeypatch.setattr(
        "rag_ingestion.external_retrieval.searxng_source.requests.get",
        lambda *_a, **_k: (_ for _ in ()).throw(requests.exceptions.Timeout("private detail")),
    )
    manager = ExternalSourceManager()
    manager.register(source)
    with caplog.at_level(logging.INFO):
        result = manager.search(CLAIM, domain="political")
    assert result.evidence == []
    assert result.provider_statuses["searxng"] == "timed_out"
    assert "outcome=timeout" in caplog.text
    assert "private detail" not in caplog.text


def test_relevant_searxng_result_reaches_existing_page_fetch_and_verifier():
    url = "https://news.example/story"
    calls = []
    manager = ExternalSourceManager()
    manager.register(Provider("searxng", results=[_candidate(url=url)], calls=calls))
    pages = Pages(RELEVANT_PAGE)
    verifier = Verifier()
    result = _pipeline(manager, pages, verifier).process(
        claim=CLAIM, domain="political", retrieval_queries=[CLAIM]
    )
    assert calls == [CLAIM]
    assert pages.urls == [url]
    assert any(item.evidence_id == "result" for item in result.fused_evidence)
    assert any(item.evidence_id == "result" for item in verifier.calls[-1])
    evidence = next(item for item in result.external_evidence if item.evidence_id == "result")
    assert evidence.text == RELEVANT_PAGE
    assert evidence.metadata["full_page_extracted"] is True


def test_unrelated_searxng_result_is_rejected_by_fusion():
    manager = ExternalSourceManager()
    manager.register(Provider("searxng", results=[_candidate()]))
    pages = Pages(
        "A detailed report about local weather and rainfall across the region. "
        "It contains no information about the Israeli president or government formation."
    )
    verifier = Verifier()
    result = _pipeline(manager, pages, verifier).process(
        claim=CLAIM, domain="political", retrieval_queries=[CLAIM]
    )
    assert pages.urls == ["https://news.example/story"]
    assert result.fused_evidence == []
    assert verifier.calls[-1] == []


def test_local_evidence_is_verified_after_searxng_retrieval():
    searx_calls = []
    manager = ExternalSourceManager()
    manager.register(Provider("searxng", calls=searx_calls))
    local_item = Evidence(
        evidence_id="local", chunk_id="local", document_id="local",
        text=RELEVANT_PAGE, domain="political", source="Local source",
        language="en", metadata={}, similarity_score=0.95,
        retrieval_method="faiss_local",
    )

    class Local:
        def search(self, **_kwargs):
            return [local_item]

    class SupportsLocal(Verifier):
        def verify(self, claim, evidence_list, **_kwargs):
            self.calls.append(list(evidence_list))
            return OverallVerification(
                claim=claim,
                results=[VerificationResult(
                    evidence_id="local", claim=claim, relationship="SUPPORTS",
                    confidence=0.9, reasoning="Direct support.", verifier="test",
                    claim_aligned=True, directly_establishes_proposition=True,
                )],
                overall_assessment="SUPPORTED", overall_confidence=0.9,
            )

    pipeline = _pipeline(manager, verifier=SupportsLocal())
    pipeline.retrieval_manager = Local()
    result = pipeline.process(claim=CLAIM, domain="political", retrieval_queries=[CLAIM])
    assert result.fused_evidence == [local_item]
    assert searx_calls == [CLAIM]
    assert len(pipeline.verification_manager.calls) == 1


def test_five_candidates_still_fetches_at_most_three_pages():
    candidates = [
        _candidate(f"result-{i}", f"https://news{i}.example/story", RELEVANT_PAGE[:100])
        for i in range(5)
    ]
    manager = ExternalSourceManager()
    manager.register(Provider("searxng", results=candidates))
    pages = Pages(RELEVANT_PAGE)
    _pipeline(manager, pages).process(claim=CLAIM, domain="political", retrieval_queries=[CLAIM])
    assert len(pages.urls) <= 3
