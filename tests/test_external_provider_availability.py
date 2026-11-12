import logging

import requests

from rag_ingestion.pipeline.claim_processing import ClaimProcessingPipeline
from rag_ingestion.external_retrieval.source_manager import (
    ExternalSourceManager,
    ExternalSourceResult,
)
from rag_ingestion.external_retrieval.base_source import BaseExternalSource
from rag_ingestion.external_retrieval.source_schema import ExternalEvidence


def test_missing_key_providers_are_not_registered(monkeypatch):
    monkeypatch.delenv("GOOGLE_FACTCHECK_API_KEY", raising=False)
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("ENABLE_GDELT", "false")

    manager = ClaimProcessingPipeline._default_external_manager()
    names = {provider.name for provider in manager.providers}

    assert "gdelt" not in names
    assert "fact_check" not in names
    assert "news_api" not in names
    assert "searxng" not in names


def test_searxng_is_not_registered_in_the_stabilized_pipeline(monkeypatch):
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_FACTCHECK_API_KEY", raising=False)
    monkeypatch.setenv("ENABLE_GDELT", "false")

    providers = ClaimProcessingPipeline._default_external_manager().providers
    assert "searxng" not in {provider.name for provider in providers}
    monkeypatch.setenv("SEARXNG_BASE_URL", "https://search.example")
    providers = ClaimProcessingPipeline._default_external_manager().providers
    assert "searxng" not in {provider.name for provider in providers}
    monkeypatch.setenv("SEARXNG_BASE_URL", "")
    assert "searxng" not in {
        provider.name for provider in ClaimProcessingPipeline._default_external_manager().providers
    }


def test_political_provider_order_puts_general_search_after_news_sources():
    manager = ExternalSourceManager()
    calls = []

    def source(name, news=False):
        class Probe(BaseExternalSource):
            supported_domains = frozenset({"political"})
            news_capable = news
            def __init__(self):
                self.name = name
            def search(self, **_kwargs):
                calls.append(self.name)
                return []
        return Probe()

    manager.register(source("gdelt", news=True))
    manager.register(source("searxng"))
    manager.register(source("fact_check"))
    manager.register(source("news_api", news=True))
    manager.search("claim", domain="political")
    assert calls == ["news_api", "fact_check", "gdelt", "searxng"]


def test_gdelt_is_skipped_when_newsapi_already_found_candidates():
    manager = ExternalSourceManager()

    class NewsCandidate(BaseExternalSource):
        name = "news_api"
        news_capable = True
        def search(self, **_kwargs):
            return [ExternalEvidence(
                evidence_id="news-1", title="Current event", text="Article body",
                url="https://example.test/news", source="Publisher", domain="general",
                source_type="news_api",
            )]

    class GdeltProbe(BaseExternalSource):
        name = "gdelt"
        news_capable = True
        def search(self, **_kwargs):
            raise AssertionError("GDELT must remain an optional fallback")

    manager.register(NewsCandidate())
    manager.register(GdeltProbe())

    result = manager.search("current event", domain="general")

    assert len(result.evidence) == 1
    assert result.provider_statuses["gdelt"] == "skipped_newsapi_candidates"


def test_news_api_remains_registered_when_configured(monkeypatch):
    monkeypatch.setenv("NEWS_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_FACTCHECK_API_KEY", raising=False)
    monkeypatch.setenv("ENABLE_GDELT", "false")

    names = {provider.name for provider in ClaimProcessingPipeline._default_external_manager().providers}

    assert "news_api" in names
    assert "gdelt" not in names


def test_disabled_gdelt_is_not_called(monkeypatch):
    monkeypatch.setenv("ENABLE_GDELT", "false")
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_FACTCHECK_API_KEY", raising=False)

    manager = ClaimProcessingPipeline._default_external_manager()
    assert all(provider.name != "gdelt" for provider in manager.providers)


class FailingProvider(BaseExternalSource):
    name = "failing"
    supported_domains = None

    def search(self, **kwargs):
        raise RuntimeError("provider unavailable")


class WorkingProvider(BaseExternalSource):
    name = "working"
    supported_domains = None

    def search(self, **kwargs):
        return [ExternalEvidence(
            evidence_id="working-1", title="Test", text="usable evidence",
            url="https://example.test/evidence", source="working", domain="general",
        )]


def test_provider_failure_isolated_and_recorded():
    manager = ExternalSourceManager()
    manager.register(FailingProvider())
    manager.register(WorkingProvider())

    result = manager.search("query", domain="general", top_k=5)

    assert len(result.evidence) == 1
    assert result.evidence[0].source == "working"
    assert len(result.errors) == 1
    assert result.errors[0].provider_name == "failing"
    assert result.provider_statuses == {"failing": "unavailable", "working": "success"}
    assert result.external_status == "partial"


def test_successful_zero_results_is_distinct_from_provider_failure():
    class EmptyProvider(WorkingProvider):
        name = "empty"
        def search(self, **kwargs):
            return []

    manager = ExternalSourceManager()
    manager.register(EmptyProvider())
    result = manager.search("query", domain="general")

    assert result.evidence == []
    assert result.errors == []
    assert result.provider_statuses == {"empty": "zero_results"}
    assert result.external_status == "zero_results"


def test_provider_diagnostics_include_runtime_status_count_and_failure_category():
    class EmptyProvider(BaseExternalSource):
        name = "empty"
        def search(self, **_kwargs):
            return []

    class FailedProvider(BaseExternalSource):
        name = "failed"
        def search(self, **_kwargs):
            raise requests.exceptions.ConnectionError("offline")

    manager = ExternalSourceManager()
    manager.provider_configuration = {"disabled": False}
    manager.register(EmptyProvider())
    manager.register(FailedProvider())
    result = manager.search("grounded claim", domain="general")

    assert result.provider_diagnostics["disabled"] == {
        "provider": "disabled", "enabled": False, "query": "grounded claim",
        "status": "disabled", "http_status": None, "runtime_ms": 0,
        "candidate_count": 0, "failure_category": "disabled",
    }
    assert result.provider_diagnostics["empty"]["status"] == "empty_results"
    assert result.provider_diagnostics["empty"]["failure_category"] == "empty_results"
    assert result.provider_diagnostics["empty"]["candidate_count"] == 0
    assert result.provider_diagnostics["empty"]["runtime_ms"] >= 0
    assert result.provider_diagnostics["failed"]["failure_category"] == "connection_error"
    assert result.provider_diagnostics["failed"]["http_status"] is None


def test_external_manager_stops_after_caller_confirms_sufficient_fused_evidence():
    calls = []

    def provider(name, evidence_id):
        class CandidateProvider(BaseExternalSource):
            supported_domains = None

            def __init__(self):
                self.name = name

            def search(self, **_kwargs):
                calls.append(self.name)
                return [ExternalEvidence(
                    evidence_id=evidence_id, title="Evidence",
                    text="A sufficiently long evidence excerpt for a controlled test.",
                    url=f"https://{self.name}.example.test/evidence",
                    source=self.name, domain="general",
                )]

        return CandidateProvider()

    manager = ExternalSourceManager()
    manager.register(provider("first", "first-evidence"))
    manager.register(provider("second", "second-evidence"))
    manager.register(provider("third", "third-evidence"))

    result = manager.search(
        "query", domain="general", stop_when=lambda items: len(items) >= 2,
    )

    assert calls == ["first", "second"]
    assert result.provider_statuses == {
        "first": "success", "second": "success",
        "third": "skipped_sufficient_evidence",
    }


def test_timeout_and_all_provider_failures_are_explicit():
    class TimeoutProvider(WorkingProvider):
        name = "timeout"
        def search(self, **kwargs):
            raise requests.exceptions.Timeout("request timed out")

    manager = ExternalSourceManager()
    manager.register(TimeoutProvider())
    manager.register(FailingProvider())
    result = manager.search("query", domain="general")

    assert result.external_status == "provider_unavailable"
    assert result.provider_statuses == {"timeout": "timed_out", "failing": "unavailable"}


def test_provider_diagnostics_capture_wrapped_http_status(caplog):
    class RateLimitedProvider(FailingProvider):
        name = "rate_limited"

        def search(self, **kwargs):
            response = requests.Response()
            response.status_code = 429
            cause = requests.exceptions.HTTPError("HTTP 429", response=response)
            raise RuntimeError("provider request failed") from cause

    manager = ExternalSourceManager()
    manager.register(RateLimitedProvider())
    with caplog.at_level(logging.INFO):
        result = manager.search("query", domain="general", top_k=5)

    assert not result.evidence
    assert result.errors[0].category == "rate_limited"
    assert result.external_status == "provider_unavailable"
    assert "root_exception_type=HTTPError" in caplog.text
    assert "http_status=429" in caplog.text
    assert "candidate_count=0" in caplog.text


def test_provider_result_diagnostics_count_raw_and_duplicate_results():
    manager = ExternalSourceManager()

    class DuplicateProvider(WorkingProvider):
        name = "duplicate"

        def search(self, **kwargs):
            item = ExternalEvidence(
                evidence_id="shared-id", title="Test", text="usable evidence",
                url="https://example.test/evidence", source=self.name,
                domain="general",
            )
            return [item, item]

    class SharedIdProvider(WorkingProvider):
        name = "shared_id"

        def search(self, **kwargs):
            return [ExternalEvidence(
                evidence_id="shared-id", title="Test", text="usable evidence",
                url="https://example.test/evidence", source=self.name,
                domain="general",
            )]

    manager.register(DuplicateProvider())
    manager.register(SharedIdProvider())
    result = manager.search("query", domain="general")

    assert result.provider_counts == {"duplicate": 2, "shared_id": 1}
    assert result.raw_result_count == 3
    assert result.duplicate_count == 2
    assert len(result.evidence) == 1


def test_provider_capabilities_match_provider_purpose():
    providers = {
        provider.name: provider
        for provider in ClaimProcessingPipeline._default_external_manager().providers
    }

    assert providers["wikimedia"].supported_domains == frozenset({"general", "medical", "science", "finance", "political"})
    assert providers["europe_pmc"].supported_domains == frozenset({"medical", "science"})
    assert providers["crossref"].supported_domains == frozenset({"medical", "science"})
    assert providers["world_bank"].supported_domains == frozenset({"political", "finance"})
    assert providers["ipu_parline"].supported_domains == frozenset({"political"})


def test_political_search_skips_wikimedia():
    manager = ExternalSourceManager()
    calls = []

    class WikimediaProbe(WorkingProvider):
        name = "wikimedia"
        supported_domains = frozenset({"general"})

        def search(self, **kwargs):
            calls.append(self.name)
            return []

    class PoliticalProbe(WorkingProvider):
        name = "political_source"
        supported_domains = frozenset({"political"})

        def search(self, **kwargs):
            calls.append(self.name)
            return []

    manager.register(WikimediaProbe())
    manager.register(PoliticalProbe())
    manager.search("query", domain="political")

    assert calls == ["political_source"]


def test_political_search_prioritizes_news_and_survives_gdelt_429():
    manager = ExternalSourceManager()
    calls = []

    class GDELTProbe(WorkingProvider):
        name = "gdelt"
        news_capable = True
        supported_domains = frozenset({"political"})

        def search(self, **kwargs):
            calls.append(self.name)
            error = RuntimeError("HTTP 429 Too Many Requests")
            error.status_code = 429
            raise error

    class NewsAPIProbe(WorkingProvider):
        name = "news_api"
        news_capable = True
        supported_domains = frozenset({"political"})

        def search(self, **kwargs):
            calls.append(self.name)
            return [ExternalEvidence(
                evidence_id="news-1", title="Current political report",
                text="A political event was reported today.",
                url="https://news.example.test/article", source="news_api",
                domain="political",
            )]

    class OtherPoliticalProbe(WorkingProvider):
        name = "other_political"
        supported_domains = frozenset({"political"})

        def search(self, **kwargs):
            calls.append(self.name)
            return []

    manager.register(OtherPoliticalProbe())
    manager.register(NewsAPIProbe())
    manager.register(GDELTProbe())
    result = manager.search("political event", domain="political")

    assert calls == ["news_api", "gdelt", "other_political"]
    assert [item.evidence_id for item in result.evidence] == ["news-1"]
    assert any(error.provider_name == "gdelt" for error in result.errors)


def test_default_gdelt_registration_disables_internal_retries(monkeypatch):
    monkeypatch.setenv("ENABLE_GDELT", "true")
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_FACTCHECK_API_KEY", raising=False)

    providers = ClaimProcessingPipeline._default_external_manager().providers
    gdelt = next(provider for provider in providers if provider.name == "gdelt")

    assert gdelt.max_retries == 0


def test_rate_limited_provider_is_skipped_for_remaining_queries():
    manager = ExternalSourceManager()
    calls = []

    class RateLimited(WorkingProvider):
        name = "limited"
        def search(self, **kwargs):
            calls.append(self.name)
            error = RuntimeError("HTTP 429 Too Many Requests")
            error.status_code = 429
            raise error

    manager.register(RateLimited())
    manager.begin_attempt()
    first = manager.search("one", domain="general")
    second = manager.search("two", domain="general")
    assert calls == ["limited"]
    assert len(first.errors) == 1
    assert len(second.errors) == 1
    manager.begin_attempt()
    manager.search("three", domain="general")
    assert calls == ["limited", "limited"]


def test_provider_without_historical_support_is_skipped_with_diagnostic():
    manager = ExternalSourceManager()

    class RecentOnly(WorkingProvider):
        name = "recent_only"
        supports_historical = False

        def search(self, **kwargs):
            raise AssertionError("recent-only provider must not be called")

    manager.register(RecentOnly())
    result = manager.search(
        "event 2020", domain="general", temporal_context={"has_explicit_timeframe": True}
    )
    assert result.evidence == []
    assert result.errors[0].provider_name == "recent_only"
    assert "historical" in str(result.errors[0].exception)


def test_unknown_domain_reaches_external_providers_without_invalid_filter():
    from types import SimpleNamespace

    class Local:
        def search(self, **_kwargs):
            return []

    class External:
        def __init__(self):
            self.domain = "not-called"

        def search(self, **kwargs):
            self.domain = kwargs["domain"]
            return SimpleNamespace(evidence=[], errors=[], provider_statuses={})

    pipeline = ClaimProcessingPipeline.__new__(ClaimProcessingPipeline)
    pipeline.retrieval_manager = Local()
    pipeline.external_manager = External()
    pipeline.normalizer = object()
    pipeline.web_retriever = SimpleNamespace(retrieve=lambda _url: None)
    pipeline.max_web_fallback_urls = 0
    pipeline.minimum_sufficient_external_evidence = 2

    pipeline._retrieve_query("claim", "claim", "Unknown", 5, 5)

    assert pipeline.external_manager.domain is None


def test_successful_provider_candidates_reach_fusion():
    from rag_ingestion.pipeline.claim_processing import ClaimProcessingPipeline
    from rag_ingestion.retrieval.evidence_fusion import EvidenceFusionManager
    from rag_ingestion.verification.verification_schema import OverallVerification

    external = WorkingProvider()
    external.search = lambda **kwargs: [ExternalEvidence(
        evidence_id="smoking-evidence", title="Smoking and lung cancer",
        text="Smoking increases lung cancer risk in adults, according to the study.",
        url="https://evidence.example.test/smoking", source="Trusted source",
        domain="medical",
    )]
    manager = ExternalSourceManager()
    manager.register(external)

    class Local:
        def search(self, **kwargs):
            return []

    class Verifier:
        def verify(self, claim, evidence_list, **kwargs):
            return OverallVerification(
                claim=claim, results=[], overall_assessment="INSUFFICIENT_EVIDENCE",
                overall_confidence=0,
            )

    class CapturingFusion:
        def __init__(self):
            self.seen = []
            self.delegate = EvidenceFusionManager()
        def fuse(self, **kwargs):
            self.seen.extend(kwargs.get("external_evidence", []))
            return self.delegate.fuse(**kwargs)

    fusion = CapturingFusion()
    pipeline = ClaimProcessingPipeline(
        retrieval_manager=Local(), external_manager=manager,
        fusion_manager=fusion, verification_manager=Verifier(),
        max_web_fallback_urls=0,
    )
    result = pipeline.process(
        claim="Smoking increases lung cancer risk.", domain="medical",
        retrieval_queries=["Smoking increases lung cancer risk."],
    )

    assert [item.evidence_id for item in fusion.seen] == ["smoking-evidence"]
    assert [item.evidence_id for item in result.fused_evidence] == ["smoking-evidence"]
    assert result.provider_statuses == {"working": "success"}


def test_faiss_evidence_is_verified_once_after_external_retrieval():
    from rag_ingestion.retrieval.evidence_schema import Evidence
    from rag_ingestion.verification.verification_schema import OverallVerification, VerificationResult

    events = []
    local_item = Evidence(
        evidence_id="local-1", chunk_id="chunk-1", document_id="doc-1",
        text="Smoking increases lung cancer risk.", domain="medical",
        source="CDC", language="en", metadata={}, similarity_score=0.9,
    )

    class Local:
        def search(self, **_kwargs):
            events.append("faiss")
            return [local_item]

    class External:
        providers = [type("Provider", (), {"name": "external"})()]
        def search(self, **_kwargs):
            events.append("external")
            from rag_ingestion.external_retrieval.source_manager import ExternalSourceResult
            return ExternalSourceResult(
                provider_statuses={"external": "zero_results"},
                provider_diagnostics={"external": {
                    "provider": "external", "enabled": True,
                    "status": "empty_results", "candidate_count": 0,
                    "runtime_ms": 0, "failure_category": "empty_results",
                }},
            )

    class Fusion:
        def fuse(self, *, local_evidence, **_kwargs):
            events.append("fusion")
            return list(local_evidence)

    class Verifier:
        def verify(self, claim, evidence_list, **_kwargs):
            events.append("verifier")
            if not evidence_list:
                return OverallVerification(
                    claim=claim, results=[], overall_assessment="INSUFFICIENT_EVIDENCE",
                    overall_confidence=0,
                )
            return OverallVerification(
                claim=claim,
                results=[VerificationResult(
                    evidence_id="local-1", claim=claim, relationship="SUPPORTS",
                    confidence=0.9, reasoning="Direct support.", verifier="test",
                    claim_aligned=True, directly_establishes_proposition=True,
                )],
                overall_assessment="SUPPORTED", overall_confidence=0.9,
            )

    pipeline = ClaimProcessingPipeline(
        retrieval_manager=Local(), external_manager=External(),
        fusion_manager=Fusion(), verification_manager=Verifier(),
    )
    result = pipeline.process(
        claim="Smoking increases lung cancer risk.", domain="medical",
        retrieval_queries=["Smoking increases lung cancer risk."],
    )

    assert events.index("faiss") < events.index("fusion") < events.index("external") < events.index("verifier")
    assert result.fused_evidence == [local_item]
    assert result.provider_statuses == {"external": "zero_results"}


def test_faiss_insufficient_evidence_uses_external_fallback_then_verifies():
    from rag_ingestion.retrieval.evidence_fusion import EvidenceFusionManager
    from rag_ingestion.verification.verification_schema import OverallVerification

    events = []
    external = WorkingProvider()

    def search_external(**_kwargs):
        events.append("external")
        return [ExternalEvidence(
            evidence_id="external-1", title="Smoking and lung cancer",
            text=("Public health research finds that smoking increases the risk of lung cancer "
                  "among adults who smoke."),
            url="https://example.test/smoking", source="Public health source",
            domain="medical",
        )]

    external.search = search_external
    manager = ExternalSourceManager()
    manager.register(external)

    class Local:
        def search(self, **_kwargs):
            events.append("faiss")
            return []

    class Verifier:
        def verify(self, claim, evidence_list, **_kwargs):
            events.append("verifier")
            return OverallVerification(
                claim=claim, results=[], overall_assessment="INSUFFICIENT_EVIDENCE",
                overall_confidence=0,
            )

    class Fusion:
        def __init__(self):
            self.delegate = EvidenceFusionManager()
        def fuse(self, **kwargs):
            events.append("fusion")
            return self.delegate.fuse(**kwargs)

    pipeline = ClaimProcessingPipeline(
        retrieval_manager=Local(), external_manager=manager,
        fusion_manager=Fusion(), verification_manager=Verifier(),
        max_web_fallback_urls=0,
    )
    result = pipeline.process(
        claim="Smoking increases lung cancer risk.", domain="medical",
        retrieval_queries=["Smoking increases lung cancer risk."],
    )

    assert events.index("faiss") < events.index("external") < events.index("verifier")
    assert result.provider_statuses == {"working": "success"}
    assert result.external_evidence
    assert result.verification_result.overall_assessment == "INSUFFICIENT_EVIDENCE"
