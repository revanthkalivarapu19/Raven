"""Lightweight spawn target for Raven's persistent verification worker.

This module intentionally imports only the standard library at module import
time. On Windows, ``spawn`` imports the target module before invoking the
target function, so importing the graph from this module would load Torch
before the child can record startup checkpoints or set conservative thread
limits.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_THREAD_ENV_DEFAULTS = {
    # Four intra-op threads materially speed up BGE on the local CPU while
    # keeping the spawned worker's memory and paging behavior bounded.
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "TOKENIZERS_PARALLELISM": "false",
}


def _atomic_pickle(path: Path, value: Any) -> None:
    temporary = Path(f"{path}.part")
    with temporary.open("wb") as output:
        pickle.dump(value, output, protocol=pickle.HIGHEST_PROTOCOL)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def warm_pipeline_worker_entry(work_dir: str, limit_threads: bool = True) -> None:
    """Initialize Raven in explicit stages, then handle queued requests."""
    started_wall = time.time()
    directory = Path(work_dir)
    ready_path = directory / "worker-ready.pickle"
    checkpoint_path = directory / "worker-checkpoints.pickle"
    checkpoints: list[str] = []
    checkpoint_events: list[dict[str, Any]] = []
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    def checkpoint(name: str, **details: Any) -> None:
        checkpoints.append(name)
        checkpoint_events.append({"checkpoint": name, "details": details, "at": time.time()})
        logger.info(
            "verification_worker_checkpoint pid=%s checkpoint=%s details=%s",
            os.getpid(), name, details,
        )
        try:
            _atomic_pickle(checkpoint_path, {
                "pid": os.getpid(),
                "checkpoints": list(checkpoints),
                "checkpoint_details": list(checkpoint_events),
                "last_successful_checkpoint": name,
                "details": details,
            })
        except Exception:
            logger.exception("verification_worker failed to persist checkpoint=%s", name)

    checkpoint(
        "worker_process_started",
        pid=os.getpid(),
        start_method=multiprocessing.get_start_method(allow_none=True),
    )
    checkpoint(
        "python_runtime_ready",
        version=sys.version.split()[0],
        executable=sys.executable,
        torch_preloaded="torch" in sys.modules,
        sentence_transformers_preloaded="sentence_transformers" in sys.modules,
        thread_environment={key: os.environ.get(key) for key in _THREAD_ENV_DEFAULTS},
    )
    checkpoint("imports_started")

    try:
        # This module is a clean spawn target; these defaults are applied
        # before importing any native ML runtime in the child.
        if limit_threads:
            for key, value in _THREAD_ENV_DEFAULTS.items():
                os.environ[key] = value
        checkpoint(
            "cpu_thread_limits_applied" if limit_threads else "cpu_thread_limits_unmodified",
            values={key: os.environ.get(key) for key in _THREAD_ENV_DEFAULTS},
        )
        import torch

        checkpoint(
            "torch_imported",
            version=getattr(torch, "__version__", "unknown"),
            intraop_threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),
            cuda_build=getattr(getattr(torch, "version", None), "cuda", None),
            cuda_available=torch.cuda.is_available(),
        )
        import sentence_transformers

        checkpoint(
            "sentence_transformers_imported",
            version=getattr(sentence_transformers, "__version__", "unknown"),
        )

        from agents.graph import (
            get_claim_agent,
            get_claim_processing_pipeline,
            get_domain_agent,
            get_raven_graph,
            run_pipeline,
        )
        from rag_ingestion.verification.ollama_verifier import _OLLAMA_SESSION
        from rag_ingestion.config.config import config as raven_config

        get_raven_graph()
        checkpoint("graph_initialized")
        pipeline = get_claim_processing_pipeline()
        get_claim_agent()
        get_domain_agent()
        checkpoint("retrieval_initialized")
        bge_load_started = time.perf_counter()
        checkpoint("BGE_load_started", model=raven_config.embedding_model)

        bge_ready = False
        bge_error = None
        try:
            pipeline.retrieval_manager.embedder._load_model_if_needed()
            bge_ready = True
            checkpoint(
                "BGE_load_completed",
                dimension=pipeline.retrieval_manager.embedder.dimension,
                load_seconds=time.perf_counter() - bge_load_started,
            )
            # Prime the model's CPU kernels before the first timed request.
            # This remains a single reusable BGE instance; it only moves the
            # first-call setup cost into worker startup.
            warmup_started = time.perf_counter()
            try:
                pipeline.retrieval_manager.embedder._model.encode(
                    ["General factual evidence passage for semantic ranking."],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                checkpoint(
                    "BGE_inference_warmup_completed",
                    warmup_seconds=time.perf_counter() - warmup_started,
                )
            except Exception as exc:
                checkpoint(
                    "BGE_inference_warmup_failed",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
        except Exception as exc:
            bge_error = f"{type(exc).__name__}: {exc}"
            pipeline.mark_semantic_embedder_unavailable(exc)
            logger.exception("verification_worker BGE unavailable; using generic ranking fallback")
            checkpoint("BGE_load_failed", error_type=type(exc).__name__, error=str(exc))

        checkpoint("Ollama_session_initialized", initialized=_OLLAMA_SESSION is not None)
        ready_wall = time.time()
        _atomic_pickle(ready_path, {
            "status": "ready",
            "pid": os.getpid(),
            "worker_started_at": started_wall,
            "worker_ready_at": ready_wall,
            "worker_startup_seconds": ready_wall - started_wall,
            "ollama_session_initialized": _OLLAMA_SESSION is not None,
            "bge_ready": bge_ready,
            "bge_error": bge_error,
            "bge_fallback_active": not bge_ready,
            "checkpoints": list(checkpoints),
            "checkpoint_details": list(checkpoint_events),
            "start_method": multiprocessing.get_start_method(allow_none=True),
            "thread_environment": {key: os.environ.get(key) for key in _THREAD_ENV_DEFAULTS},
            "torch_threads": {
                "intraop": torch.get_num_threads(),
                "interop": torch.get_num_interop_threads(),
            },
        })
        checkpoint("worker_ready", bge_ready=bge_ready, startup_seconds=ready_wall - started_wall)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        try:
            _atomic_pickle(ready_path, {
                "status": "error",
                "error": error,
                "worker_started_at": started_wall,
                "worker_ready_at": time.time(),
                "last_successful_checkpoint": checkpoints[-1] if checkpoints else None,
                "checkpoints": list(checkpoints),
                "checkpoint_details": list(checkpoint_events),
            })
        except Exception:
            logger.exception("verification_worker failed to write startup diagnostics")
        logger.exception(
            "verification_worker event=startup_error last_checkpoint=%s",
            checkpoints[-1] if checkpoints else "none",
        )
        return

    while True:
        jobs = sorted(directory.glob("job-*.pickle"))
        if not jobs:
            time.sleep(0.02)
            continue
        job_path = jobs[0]
        running_path = directory / f"running-{job_path.name}"
        try:
            os.replace(job_path, running_path)
        except FileNotFoundError:
            continue
        job_id = job_path.stem.removeprefix("job-")
        result_path = directory / f"result-{job_id}.pickle"
        started_path = directory / f"started-{job_id}.pickle"
        try:
            with running_path.open("rb") as job_file:
                job = pickle.load(job_file)
            claimed_wall = time.time()
            _atomic_pickle(started_path, {"worker_claimed_at": claimed_wall})
            remaining = max(0.0, float(job["wall_deadline_at"]) - claimed_wall)
            local_deadline = time.monotonic() + remaining
            state = _run_pipeline(run_pipeline, job["text"], job["image"], local_deadline)
            _atomic_pickle(result_path, ("ok", state))
        except BaseException as exc:
            _atomic_pickle(result_path, ("error", f"{type(exc).__name__}: {exc}"))
        finally:
            try:
                running_path.unlink()
            except FileNotFoundError:
                pass


def _run_pipeline(run_pipeline, text: str, image: Any, deadline_at: float):
    """Run text or image input inside the worker, preserving the existing API path."""
    if image is None:
        return run_pipeline(text=text, deadline_at=deadline_at)

    # The image validation helper remains owned by api.main. Import it lazily
    # after worker startup so it cannot affect the ML import checkpoint order.
    from api.main import _image_parts

    image_parts = _image_parts(image)
    if image_parts is None:
        return run_pipeline(text=text, deadline_at=deadline_at)
    image_bytes, suffix = image_parts
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="raven-upload-", suffix=suffix, delete=False) as image_file:
            image_file.write(image_bytes)
            temporary_path = Path(image_file.name)
        return run_pipeline(text=text or None, image_path=str(temporary_path), deadline_at=deadline_at)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
