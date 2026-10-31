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
    monkeypatch.setenv("ENABLE_GDELT", "false")

    manager = ClaimProcessingPipeline._default_external_manager()
    names = {provider.name for provider in manager.providers}

    assert "gdelt" not in names
    assert "fact_check" not in names
    assert "news_api" not in names


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

    assert providers["wikimedia"].supported_domains == frozenset({"general"})
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
