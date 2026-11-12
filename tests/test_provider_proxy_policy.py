from rag_ingestion.external_retrieval import proxy_policy


def test_unreachable_loopback_proxy_is_bypassed_per_request(monkeypatch):
    monkeypatch.setattr(
        proxy_policy.requests.utils,
        "get_environ_proxies",
        lambda _url: {
            "http": "http://127.0.0.1:9",
            "https": "http://127.0.0.1:9",
            "all": "http://127.0.0.1:9",
        },
    )
    monkeypatch.setattr(proxy_policy, "_tcp_reachable", lambda _host, _port: False)

    assert proxy_policy.provider_proxy_overrides("https://newsapi.org/v2/everything") == {
        "http": "", "https": "", "all": "",
    }


def test_non_local_proxy_is_preserved(monkeypatch):
    monkeypatch.setattr(
        proxy_policy.requests.utils,
        "get_environ_proxies",
        lambda _url: {"https": "http://proxy.company.example:8080"},
    )
    monkeypatch.setattr(proxy_policy, "_tcp_reachable", lambda _host, _port: False)

    assert proxy_policy.provider_proxy_overrides("https://newsapi.org/v2/everything") is None


def test_reachable_loopback_proxy_is_preserved(monkeypatch):
    monkeypatch.setattr(
        proxy_policy.requests.utils,
        "get_environ_proxies",
        lambda _url: {"https": "http://127.0.0.1:8080"},
    )
    monkeypatch.setattr(proxy_policy, "_tcp_reachable", lambda _host, _port: True)

    assert proxy_policy.provider_proxy_overrides("https://newsapi.org/v2/everything") is None


def test_no_inherited_proxy_needs_no_override(monkeypatch):
    monkeypatch.setattr(proxy_policy.requests.utils, "get_environ_proxies", lambda _url: {})

    assert proxy_policy.provider_proxy_overrides("https://newsapi.org/v2/everything") is None
