import asyncio
import os
from pathlib import Path
import time

import pytest
from fastapi import HTTPException
from unittest.mock import Mock

from api import main


class _FakeProcess:
    def __init__(self, target, args, *, name):
        self.target = target
        self.args = args
        self.name = name
        self.exitcode = None
        self._alive = False

    def start(self):
        self._alive = True
        self.observed_thread_environment = {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM")
        }
        directory = Path(self.args[0])
        if self.name == "raven-warm-verification-worker":
            main._atomic_pickle(directory / "worker-ready.pickle", {
                "status": "ready", "pid": 123,
                "worker_started_at": time.time(),
                "worker_ready_at": time.time() + 0.01,
                "worker_startup_seconds": 0.01,
            })

    def is_alive(self):
        return self._alive

    def terminate(self):
        self._alive = False

    def join(self, timeout=None):
        return None


class _FakeContext:
    def __init__(self):
        self.processes = []

    def Process(self, target, args, *, name):
        process = _FakeProcess(target, args, name=name)
        self.processes.append(process)
        return process


def test_warm_worker_starts_once_and_records_readiness(monkeypatch, tmp_path):
    worker = main.WarmPipelineWorker()
    worker.work_dir = tmp_path
    previous_environment = {
        key: os.environ.get(key)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM")
    }
    monkeypatch.setattr(main.multiprocessing, "get_context", lambda _kind: _FakeContext())

    asyncio.run(worker.start())
    original_process = worker.process
    asyncio.run(worker.start())

    assert worker.is_ready
    assert worker.process is original_process
    assert worker.process.observed_thread_environment == {
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
    }
    assert {
        key: os.environ.get(key)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM")
    } == previous_environment
    assert worker.ready["worker_ready_at"] >= worker.ready["worker_started_at"]
    assert worker.ready["worker_startup_seconds"] >= 0
    assert main._warm_pipeline_worker_entry.__module__ == "api.warm_worker"
    asyncio.run(worker.stop())


def test_malformed_readiness_artifact_terminates_and_releases_child(monkeypatch, tmp_path):
    worker = main.WarmPipelineWorker()
    worker.work_dir = tmp_path
    context = _FakeContext()

    class MalformedReadyProcess(_FakeProcess):
        def start(self):
            self._alive = True
            (Path(self.args[0]) / "worker-ready.pickle").write_bytes(b"not a pickle")

    class MalformedReadyContext:
        def Process(self, target, args, *, name):
            process = MalformedReadyProcess(target, args, name=name)
            context.processes.append(process)
            return process

    monkeypatch.setattr(main.multiprocessing, "get_context", lambda _kind: MalformedReadyContext())

    with pytest.raises(RuntimeError, match="invalid worker readiness artifact"):
        asyncio.run(worker.start())

    assert len(context.processes) == 1
    assert context.processes[0]._alive is False
    assert worker.process is None
    assert worker.is_ready is False


def test_warm_worker_retries_transient_readiness_permission_error(monkeypatch, tmp_path):
    worker = main.WarmPipelineWorker()
    worker.work_dir = tmp_path
    monkeypatch.setattr(main.multiprocessing, "get_context", lambda _kind: _FakeContext())
    original_open = Path.open
    denied_once = {"value": False}

    def transiently_denied_open(path, *args, **kwargs):
        if path.name == "worker-ready.pickle" and args and args[0] == "rb" and not denied_once["value"]:
            denied_once["value"] = True
            raise PermissionError("transient Windows file lock")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", transiently_denied_open)
    asyncio.run(worker.start())

    assert denied_once["value"] is True
    assert worker.is_ready
    asyncio.run(worker.stop())


def test_warm_request_diagnostics_show_zero_request_startup():
    worker = main.WarmPipelineWorker()
    worker.ready = {
        "worker_started_at": 100.0,
        "worker_ready_at": 101.25,
        "worker_startup_seconds": 1.25,
    }
    state = worker._add_lifecycle_diagnostics(
        {"request_diagnostics": {"total_duration_seconds": 8.0}},
        queue_wait=0.02, graph_seconds=7.8, total_seconds=8.1,
        restarted=False,
    )

    diagnostics = state["request_diagnostics"]
    assert diagnostics["worker_warm"] is True
    assert diagnostics["worker_startup_seconds"] == 0.0
    assert diagnostics["worker_lifetime_startup_seconds"] == 1.25
    assert diagnostics["queue_wait_seconds"] == pytest.approx(0.02)
    assert diagnostics["graph_execution_seconds"] == pytest.approx(7.8)
    assert diagnostics["total_request_seconds"] == pytest.approx(8.1)


def test_busy_worker_returns_controlled_503():
    worker = main.WarmPipelineWorker()
    worker.busy = True

    with pytest.raises(HTTPException) as error:
        asyncio.run(worker.run("claim", None))

    assert error.value.status_code == 503
    assert "busy" in error.value.detail.lower()


def test_bge_failure_disables_only_bge_backed_local_retrieval():
    from rag_ingestion.pipeline.claim_processing import ClaimProcessingPipeline

    retrieval = Mock()
    retrieval.embedder = object()
    pipeline = ClaimProcessingPipeline(
        retrieval_manager=retrieval,
        external_manager=Mock(),
    )
    pipeline.mark_semantic_embedder_unavailable(RuntimeError("paging file exhausted"))

    local, _external, error, _errors, _statuses, _diagnostics = pipeline._retrieve_query(
        "unseen factual claim", "unseen factual claim", "General", 5, 5,
        include_external=False,
    )

    assert local == []
    assert error.provider_name == "local_retrieval"
    assert "paging file exhausted" in str(error.exception)
    assert pipeline.fusion_manager.semantic_embedder is None
    retrieval.search.assert_not_called()


def test_timeout_kills_worker_and_starts_replacement(monkeypatch, tmp_path):
    worker = main.WarmPipelineWorker()
    worker.work_dir = tmp_path
    context = _FakeContext()
    monkeypatch.setattr(main.multiprocessing, "get_context", lambda _kind: context)
    monkeypatch.setattr(main, "MAX_REQUEST_BUDGET_SECONDS", 0.05)

    async def exercise():
        await worker.start()
        state = await worker.run("slow claim", None)
        await worker._restart_task
        return state

    state = asyncio.run(exercise())

    assert state["verification_failure"].error_type == "RequestDeadlineExceeded"
    assert state["request_diagnostics"]["worker_timeout_count"] == 1
    assert state["request_diagnostics"]["worker_restarts"] == 1
    assert state["request_diagnostics"]["worker_restarted"] is True
    assert worker.is_ready
    assert len(context.processes) == 2
    assert context.processes[0]._alive is False
    assert context.processes[1]._alive is True
    assert worker.process is context.processes[1]
    asyncio.run(worker.stop())
    assert all(not process._alive for process in context.processes)
