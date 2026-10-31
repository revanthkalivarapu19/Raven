import pytest

from rag_ingestion.llm_resilience import LLMInvocationError, LLMResilience, request_deadline


def test_success_is_backward_compatible():
    response = {"message": {"content": "result"}}
    assert LLMResilience().chat(lambda: response) == response


def test_timeout_is_classified():
    with pytest.raises(LLMInvocationError, match="timed out") as error:
        LLMResilience(timeout_seconds=0.01).chat(lambda: __import__("time").sleep(0.1))
    assert error.value.kind == "timeout"


def test_http_500_is_transport_failure():
    def fail():
        raise RuntimeError("HTTP 500 Internal Server Error")
    with pytest.raises(LLMInvocationError) as error:
        LLMResilience().chat(fail)
    assert error.value.kind == "transport"


def test_server_unavailable_is_classified():
    with pytest.raises(LLMInvocationError) as error:
        LLMResilience().chat(lambda: (_ for _ in ()).throw(ConnectionError("connection refused")))
    assert error.value.kind == "server_unavailable"


def test_deterministic_runtime_failure_is_not_retried():
    calls = []

    def fail():
        calls.append(1)
        raise RuntimeError("CUDA error: shared object initialization failed")

    client = LLMResilience()
    with pytest.raises(LLMInvocationError) as first:
        client.chat(fail)
    with pytest.raises(LLMInvocationError) as second:
        client.chat(fail)
    assert first.value.kind == second.value.kind == "deterministic_runtime"
    assert calls == [1]


def test_malformed_response_is_classified():
    with pytest.raises(LLMInvocationError) as error:
        LLMResilience().chat(lambda: {"message": {}})
    assert error.value.kind == "malformed_response"


class _Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_generate_accepts_ollama_generate_response():
    response = _Response(payload={"response": '{"ok": true}'})
    assert LLMResilience().generate(lambda: response)["response"] == '{"ok": true}'


def test_generate_http_500_is_classified():
    with pytest.raises(LLMInvocationError) as error:
        LLMResilience().generate(lambda: _Response(status_code=500, payload={}))
    assert error.value.kind == "http_error"


def test_generate_malformed_json_is_classified():
    with pytest.raises(LLMInvocationError) as error:
        LLMResilience().generate(lambda: _Response(payload={"unexpected": "value"}))
    assert error.value.kind == "malformed_response"


class _Message:
    def __init__(self, content):
        self.content = content


class _ChatResponse:
    def __init__(self, content):
        self.message = _Message(content)


def test_chat_accepts_ollama_chatresponse_object():
    response = _ChatResponse("structured answer")
    assert LLMResilience().chat(lambda: response) is response


def test_chat_rejects_empty_content():
    with pytest.raises(LLMInvocationError, match="malformed"):
        LLMResilience().chat(lambda: _ChatResponse("  "))


def test_chat_rejects_missing_message_and_content():
    for response in ({}, {"message": {}}, {"message": {"content": None}}):
        with pytest.raises(LLMInvocationError, match="malformed"):
            LLMResilience().chat(lambda response=response: response)


def test_chat_http_500_is_classified():
    with pytest.raises(LLMInvocationError) as error:
        LLMResilience().chat(lambda: _Response(status_code=500))
    assert error.value.kind == "http_error"


def test_llm_uses_remaining_request_budget():
    import time
    with request_deadline(time.monotonic() + 0.01):
        with pytest.raises(LLMInvocationError, match="timed out"):
            LLMResilience(timeout_seconds=10).chat(lambda: time.sleep(0.2))
