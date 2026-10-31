"""Bounded, classified invocation for local Ollama calls."""

from __future__ import annotations

import logging
import threading
import contextvars
from contextlib import contextmanager
import time
from collections.abc import Mapping
from typing import Any, Callable

logger = logging.getLogger(__name__)
_REQUEST_DEADLINE = contextvars.ContextVar("raven_request_deadline", default=None)


@contextmanager
def request_deadline(deadline):
    token = _REQUEST_DEADLINE.set(deadline)
    try:
        yield
    finally:
        _REQUEST_DEADLINE.reset(token)


def remaining_request_budget() -> float | None:
    deadline = _REQUEST_DEADLINE.get()
    return None if deadline is None else max(0.0, deadline - time.monotonic())


class LLMInvocationError(RuntimeError):
    def __init__(self, message: str, kind: str, *, retryable: bool = False):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


class LLMResilience:
    """Per-client health state; failures do not poison unrelated clients."""

    def __init__(self, timeout_seconds: float = 120.0):
        self.timeout_seconds = timeout_seconds
        self._deterministic_failure = False
        self._lock = threading.Lock()

    def chat(self, call: Callable[[], Any]) -> Any:
        return self._invoke(call, self._validate_chat)

    def generate(self, call: Callable[[], Any]) -> dict[str, Any]:
        """Invoke an Ollama ``/api/generate`` request through the same guard."""
        return self._invoke(call, self._validate_generate)

    def _invoke(self, call: Callable[[], Any], validator: Callable[[Any], Any]) -> Any:
        with self._lock:
            if self._deterministic_failure:
                raise LLMInvocationError(
                    "Ollama model runtime is unavailable after a deterministic initialization failure",
                    "deterministic_runtime",
                )
        try:
            remaining = self.timeout_seconds
            deadline = _REQUEST_DEADLINE.get()
            if deadline is not None:
                remaining = min(remaining, max(0.001, deadline - time.monotonic()))
            result: dict[str, Any] = {}
            completed = threading.Event()

            def invoke() -> None:
                try:
                    result["response"] = call()
                except BaseException as exc:  # propagate through the bounded caller
                    result["error"] = exc
                finally:
                    completed.set()

            # A daemon worker ensures an uncooperative HTTP client cannot keep the
            # process alive after the request deadline has safely returned.
            worker = threading.Thread(target=invoke, name="raven-llm-call", daemon=True)
            worker.start()
            if not completed.wait(timeout=remaining):
                raise LLMInvocationError("Ollama request timed out", "timeout", retryable=True)
            if "error" in result:
                raise result["error"]
            response = result["response"]
        except LLMInvocationError:
            raise
        except TimeoutError as exc:
            raise LLMInvocationError("Ollama request timed out", "timeout", retryable=True) from exc
        except Exception as exc:
            text = str(exc).lower()
            deterministic = any(term in text for term in (
                "cuda", "shared object", "shared library", "llama-server process terminated",
                "0xc0000409", "initialization failed",
            ))
            unavailable = any(term in text for term in (
                "connection refused", "failed to connect", "server unavailable", "connection error",
            ))
            kind = "deterministic_runtime" if deterministic else "server_unavailable" if unavailable else "transport"
            if deterministic:
                with self._lock:
                    self._deterministic_failure = True
            raise LLMInvocationError(
                f"Ollama {kind} failure: {type(exc).__name__}", kind,
                retryable=not deterministic,
            ) from exc
        if hasattr(response, "status_code") and response.status_code != 200:
            raise LLMInvocationError(
                f"Ollama returned HTTP status {response.status_code}",
                "http_error",
                retryable=response.status_code >= 500,
            )
        return validator(response)

    @staticmethod
    def _validate_chat(response: Any) -> Any:
        if isinstance(response, Mapping):
            message = response.get("message")
        else:
            message = getattr(response, "message", None)
        if isinstance(message, Mapping):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise LLMInvocationError("Ollama returned a malformed response", "malformed_response")
        return response

    @staticmethod
    def _validate_generate(response: Any) -> dict[str, Any]:
        if hasattr(response, "status_code") and response.status_code != 200:
            raise LLMInvocationError(
                f"Ollama returned HTTP status {response.status_code}",
                "http_error",
                retryable=response.status_code >= 500,
            )
        try:
            data = response.json() if hasattr(response, "json") else response
        except Exception as exc:
            raise LLMInvocationError("Ollama returned malformed JSON", "malformed_response") from exc
        if not isinstance(data, dict) or not isinstance(data.get("response"), str):
            raise LLMInvocationError("Ollama returned a malformed response", "malformed_response")
        return data

    def reset(self) -> None:
        with self._lock:
            self._deterministic_failure = False
