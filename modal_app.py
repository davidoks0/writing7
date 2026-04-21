from __future__ import annotations

import json
import math
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from corpus.gutenberg_catalog import build_gutenberg_catalog_index as _build_gutenberg_catalog_index
from corpus.gutenberg_catalog import fetch_gutenberg_catalog
from corpus.fetch_gutenberg import (
    fetch_gutenberg_http_range,
    ingest_gutenberg_rsync,
    inspect_gutenberg_rsync_inventory as _inspect_gutenberg_rsync_inventory,
)
from corpus.manifests import build_corpus_manifests as _build_corpus_manifests
from corpus.manifests import freeze_corpus_splits as _freeze_corpus_splits
from corpus.manifests import load_corpus_config
from corpus.passage_pools import build_passage_pools as _build_passage_pools
from eval.aggregate_benchmark_results import aggregate_benchmark_results as _aggregate_benchmark_results
from eval.benchmark_v2 import run_benchmark as _run_benchmark
from eval.build_benchmark_manifests import build_benchmark_manifests as _build_benchmark_manifests
from eval.benchmark_io import load_json, write_json
from training.build_scorer_dataset import build_scorer_dataset as _build_scorer_dataset
from training.calibrate_style_scorer import calibrate_style_scorer as _calibrate_style_scorer
from training.train_style_scorer import train_style_scorer as _train_style_scorer


CORPUS_VOLUME_NAME = "stylebench-corpus"
ARTIFACTS_VOLUME_NAME = "stylebench-artifacts"
HF_CACHE_VOLUME_NAME = "stylebench-hf-cache"
HF_SECRET_NAME = "stylebench-huggingface"
PROVIDER_SECRET_NAME_ENV = "STYLEBENCH_PROVIDER_SECRET_NAME"

CORPUS_MOUNT = "/corpus"
ARTIFACTS_MOUNT = "/artifacts"
HF_CACHE_MOUNT = "/hf-cache"
ROOT_MOUNT = Path("/root")
RSYNC_RUNS_ROOT = Path(CORPUS_MOUNT) / "meta" / "rsync_ingest_runs"


try:  # pragma: no cover - exercised in Modal, not local smoke tests
    import modal
except ImportError:  # pragma: no cover
    modal = None


if modal is not None:  # pragma: no cover
    def _provider_secrets() -> list[Any]:
        secrets: list[Any] = []
        inline_env = {key: os.environ[key] for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if os.environ.get(key)}
        if inline_env:
            secrets.append(modal.Secret.from_dict(inline_env))
        named_secret = os.environ.get(PROVIDER_SECRET_NAME_ENV)
        if named_secret:
            secrets.append(modal.Secret.from_name(named_secret))
        return secrets

    BASE_IMAGE = (
        modal.Image.debian_slim(python_version="3.10")
        .run_commands(
            "apt-get update && apt-get install -y rsync git && rm -rf /var/lib/apt/lists/*",
        )
        .pip_install_from_requirements("requirements.lock.txt")
        .env(
            {
                "HF_HOME": HF_CACHE_MOUNT,
                "HUGGINGFACE_HUB_CACHE": HF_CACHE_MOUNT,
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
            }
        )
        .add_local_dir("configs", remote_path="/root/configs", copy=True)
        .add_local_dir("docs", remote_path="/root/docs", copy=True)
        .add_local_dir("eval/benchmark_data", remote_path="/root/eval/benchmark_data", copy=True)
        .add_local_file("requirements.lock.txt", remote_path="/root/requirements.lock.txt", copy=True)
        .add_local_python_source("corpus")
        .add_local_python_source("training")
        .add_local_python_source("eval")
        .add_local_python_source("benchmark")
    )
    try:
        HF_SECRET = modal.Secret.from_name(HF_SECRET_NAME)
    except modal.exception.NotFoundError:
        HF_SECRET = None
    BENCHMARK_PROVIDER_SECRETS = _provider_secrets()
    CORPUS_VOLUME = modal.Volume.from_name(CORPUS_VOLUME_NAME, create_if_missing=True)
    ARTIFACTS_VOLUME = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
    HF_CACHE_VOLUME = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
    app = modal.App("writing7-stylebench")

    def modal_function(*args, **kwargs):
        return app.function(*args, **kwargs)

else:

    class _DummyApp:
        def function(self, *args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        def local_entrypoint(self, *args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

    BASE_IMAGE = None
    HF_SECRET = None
    BENCHMARK_PROVIDER_SECRETS = []
    CORPUS_VOLUME = None
    ARTIFACTS_VOLUME = None
    HF_CACHE_VOLUME = None
    app = _DummyApp()

    def modal_function(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator


def _containerize_path(path_value: Optional[str]) -> Optional[str]:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path.as_posix()
    candidate = ROOT_MOUNT / path
    return candidate.as_posix() if candidate.exists() else path.as_posix()


def _print_result(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_timestamp() -> str:
    return _utc_now().isoformat().replace("+00:00", "Z")


def _default_run_id(prefix: str) -> str:
    return f"{prefix}_{_utc_now().strftime('%Y%m%dT%H%M%SZ')}"


def _rsync_run_dir(run_id: str) -> Path:
    return RSYNC_RUNS_ROOT / run_id


def _load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    return load_json(path)


def _ensure_rsync_run_manifest(run_id: str, *, max_files: Optional[int], rsync_source: str, rsync_shards: int) -> None:
    manifest_path = _rsync_run_dir(run_id) / "manifest.json"
    if manifest_path.exists():
        return
    write_json(
        manifest_path,
        {
            "run_id": run_id,
            "created_at": _utc_timestamp(),
            "max_files": max_files,
            "rsync_source": rsync_source,
            "rsync_shards": rsync_shards,
        },
    )


def _latest_rsync_run_id() -> Optional[str]:
    if not RSYNC_RUNS_ROOT.exists():
        return None
    candidates = [path for path in RSYNC_RUNS_ROOT.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).name


def _summarize_rsync_run(run_id: Optional[str]) -> dict[str, Any]:
    resolved_run_id = run_id or _latest_rsync_run_id()
    if resolved_run_id is None:
        return {"exists": False, "run_id": None}

    run_dir = _rsync_run_dir(resolved_run_id)
    manifest = _load_json_if_exists(run_dir / "manifest.json") or {}
    shard_dir = run_dir / "shards"
    shard_rows: list[dict[str, Any]] = []
    if shard_dir.exists():
        for path in sorted(shard_dir.glob("shard_*.json")):
            payload = _load_json_if_exists(path)
            if isinstance(payload, dict):
                shard_rows.append(payload)

    total_shards = int(manifest.get("rsync_shards", 0))
    status_counts = {"pending": max(0, total_shards - len(shard_rows)), "running": 0, "completed": 0, "failed": 0, "error": 0}
    totals = {"listed_count": 0, "selected_file_count": 0, "new_file_count": 0, "present_after_file_count": 0}
    for row in shard_rows:
        status = str(row.get("status", "unknown"))
        if status in status_counts:
            status_counts[status] += 1
        totals["listed_count"] += int(row.get("listed_count", 0))
        totals["selected_file_count"] += int(row.get("selected_file_count", 0))
        totals["new_file_count"] += int(row.get("new_file_count", 0))
        totals["present_after_file_count"] += int(row.get("present_after_file_count", 0))

    return {
        "exists": run_dir.exists(),
        "run_id": resolved_run_id,
        "manifest": manifest,
        "status_counts": status_counts,
        "totals": totals,
        "results": shard_rows,
    }


def _chunk_ranges(start_id: int, end_id: int, chunk_size: int) -> list[tuple[int, int]]:
    ranges = []
    current = start_id
    while current <= end_id:
        chunk_end = min(end_id, current + chunk_size - 1)
        ranges.append((current, chunk_end))
        current = chunk_end + 1
    return ranges


def _reload_volume(volume) -> None:
    if volume is not None:
        volume.reload()


def _commit_volumes(*volumes) -> None:
    for volume in volumes:
        if volume is not None:
            volume.commit()


def _runtime_corpus_config(config_path: str) -> dict[str, Any]:
    config = load_corpus_config(_containerize_path(config_path))
    config["output_root"] = CORPUS_MOUNT
    if config.get("input_books_manifest"):
        config["input_books_manifest"] = _containerize_path(config["input_books_manifest"])
    return config


def _runtime_scorer_config(config_path: str) -> dict[str, Any]:
    config = load_json(_containerize_path(config_path))
    config["corpus_root"] = CORPUS_MOUNT
    config["artifacts_root"] = ARTIFACTS_MOUNT
    if config.get("split_path"):
        config["split_path"] = str(Path(CORPUS_MOUNT) / "splits" / Path(config["split_path"]).name)
    if config.get("calibration_split_path"):
        config["calibration_split_path"] = str(Path(CORPUS_MOUNT) / "splits" / Path(config["calibration_split_path"]).name)
    return config


def _runtime_benchmark_config(config_path: str) -> dict[str, Any]:
    config = load_json(_containerize_path(config_path))
    if config.get("prompt_bank_path"):
        config["prompt_bank_path"] = _containerize_path(config["prompt_bank_path"])
    return config


@modal_function(
    image=BASE_IMAGE,
    volumes={CORPUS_MOUNT: CORPUS_VOLUME},
    timeout=60 * 60,
    cpu=2,
)
def _corpus_fetch_catalog_remote(
    *,
    url: Optional[str] = None,
    timeout: float = 90.0,
) -> dict[str, Any]:
    _reload_volume(CORPUS_VOLUME)
    result = fetch_gutenberg_catalog(output_root=CORPUS_MOUNT, url=url or "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv", timeout=timeout)
    _commit_volumes(CORPUS_VOLUME)
    return result


@modal_function(
    image=BASE_IMAGE,
    volumes={CORPUS_MOUNT: CORPUS_VOLUME},
    timeout=2 * 60 * 60,
    cpu=2,
)
def _build_catalog_index_remote(catalog_path: Optional[str] = None) -> dict[str, Any]:
    _reload_volume(CORPUS_VOLUME)
    result = _build_gutenberg_catalog_index(output_root=CORPUS_MOUNT, catalog_path=catalog_path)
    _commit_volumes(CORPUS_VOLUME)
    return result


@modal_function(
    image=BASE_IMAGE,
    volumes={CORPUS_MOUNT: CORPUS_VOLUME},
    timeout=60 * 60,
    cpu=2,
)
def _corpus_fetch_http_chunk(
    start_id: int,
    end_id: int,
    *,
    timeout: float = 30.0,
    max_workers: int = 12,
) -> dict[str, Any]:
    _reload_volume(CORPUS_VOLUME)
    results = fetch_gutenberg_http_range(
        start_id,
        end_id,
        output_root=CORPUS_MOUNT,
        index_output_path=Path(CORPUS_MOUNT) / "meta" / "http_fetch_indexes" / f"fetch_{start_id}_{end_id}.json",
        max_workers=max_workers,
        timeout=timeout,
    )
    _commit_volumes(CORPUS_VOLUME)
    return {
        "start_id": start_id,
        "end_id": end_id,
        "fetched": sum(1 for row in results if row.get("ok")),
        "failed": sum(1 for row in results if not row.get("ok")),
        "results": results,
    }


@modal_function(
    image=BASE_IMAGE,
    volumes={CORPUS_MOUNT: CORPUS_VOLUME},
    timeout=12 * 60 * 60,
    cpu=4,
)
def _corpus_ingest_rsync_remote(
    shard_index: int = 0,
    *,
    max_files: Optional[int] = None,
    rsync_source: Optional[str] = None,
    shard_count: int = 1,
    run_id: Optional[str] = None,
) -> dict[str, Any]:
    _reload_volume(CORPUS_VOLUME)
    resolved_source = rsync_source or "aleph.gutenberg.org::gutenberg"
    resolved_run_id = run_id or _default_run_id("gutenberg_rsync")
    _ensure_rsync_run_manifest(resolved_run_id, max_files=max_files, rsync_source=resolved_source, rsync_shards=shard_count)
    shard_status_path = _rsync_run_dir(resolved_run_id) / "shards" / f"shard_{shard_index:03d}.json"
    started_at = _utc_now()
    write_json(
        shard_status_path,
        {
            "run_id": resolved_run_id,
            "status": "running",
            "started_at": started_at.isoformat().replace("+00:00", "Z"),
            "shard_index": shard_index,
            "shard_count": shard_count,
            "max_files": max_files,
            "rsync_source": resolved_source,
        },
    )
    _commit_volumes(CORPUS_VOLUME)
    try:
        result = ingest_gutenberg_rsync(
            output_root=CORPUS_MOUNT,
            rsync_source=resolved_source,
            max_files=max_files,
            shard_index=shard_index,
            shard_count=shard_count,
        )
        result.update(
            {
                "run_id": resolved_run_id,
                "status": "completed" if result.get("ok") else "failed",
                "started_at": started_at.isoformat().replace("+00:00", "Z"),
                "finished_at": _utc_timestamp(),
                "duration_seconds": round((_utc_now() - started_at).total_seconds(), 2),
            }
        )
    except Exception as exc:
        result = {
            "ok": False,
            "run_id": resolved_run_id,
            "status": "error",
            "started_at": started_at.isoformat().replace("+00:00", "Z"),
            "finished_at": _utc_timestamp(),
            "duration_seconds": round((_utc_now() - started_at).total_seconds(), 2),
            "shard_index": shard_index,
            "shard_count": shard_count,
            "max_files": max_files,
            "rsync_source": resolved_source,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    write_json(shard_status_path, result)
    _commit_volumes(CORPUS_VOLUME)
    return result


@modal_function(
    image=BASE_IMAGE,
    volumes={CORPUS_MOUNT: CORPUS_VOLUME.read_only() if CORPUS_VOLUME is not None else None},
    timeout=60 * 60,
    cpu=2,
)
def _inspect_gutenberg_rsync_progress_remote(run_id: Optional[str] = None) -> dict[str, Any]:
    _reload_volume(CORPUS_VOLUME)
    return {"run": _summarize_rsync_run(run_id)}


@modal_function(
    image=BASE_IMAGE,
    volumes={CORPUS_MOUNT: CORPUS_VOLUME.read_only() if CORPUS_VOLUME is not None else None},
    timeout=60 * 60,
    cpu=2,
)
def _inspect_gutenberg_rsync_inventory_remote() -> dict[str, Any]:
    _reload_volume(CORPUS_VOLUME)
    return _inspect_gutenberg_rsync_inventory(output_root=CORPUS_MOUNT)


@modal_function(
    image=BASE_IMAGE,
    volumes={CORPUS_MOUNT: CORPUS_VOLUME},
    timeout=4 * 60 * 60,
    cpu=4,
)
def _build_corpus_manifests_remote(corpus_config: str = "configs/corpus_v1.json") -> dict[str, Any]:
    _reload_volume(CORPUS_VOLUME)
    result = _build_corpus_manifests(_runtime_corpus_config(corpus_config))
    _commit_volumes(CORPUS_VOLUME)
    return result


@modal_function(
    image=BASE_IMAGE,
    volumes={CORPUS_MOUNT: CORPUS_VOLUME},
    timeout=60 * 60,
    cpu=2,
)
def _freeze_corpus_splits_remote(corpus_config: str = "configs/corpus_v1.json") -> dict[str, Any]:
    _reload_volume(CORPUS_VOLUME)
    result = _freeze_corpus_splits(_runtime_corpus_config(corpus_config))
    _commit_volumes(CORPUS_VOLUME)
    return result


@modal_function(
    image=BASE_IMAGE,
    volumes={CORPUS_MOUNT: CORPUS_VOLUME},
    timeout=4 * 60 * 60,
    cpu=4,
)
def _build_passage_pools_remote(corpus_config: str = "configs/corpus_v1.json") -> dict[str, Any]:
    _reload_volume(CORPUS_VOLUME)
    result = _build_passage_pools(_runtime_corpus_config(corpus_config))
    _commit_volumes(CORPUS_VOLUME)
    return result


@modal_function(
    image=BASE_IMAGE,
    volumes={
        CORPUS_MOUNT: CORPUS_VOLUME.read_only() if CORPUS_VOLUME is not None else None,
        ARTIFACTS_MOUNT: ARTIFACTS_VOLUME,
    },
    timeout=8 * 60 * 60,
    cpu=4,
)
def _build_scorer_dataset_remote(config: str = "configs/scorer_train_v1.json") -> dict[str, Any]:
    _reload_volume(CORPUS_VOLUME)
    _reload_volume(ARTIFACTS_VOLUME)
    result = _build_scorer_dataset(_runtime_scorer_config(config))
    _commit_volumes(ARTIFACTS_VOLUME)
    return result


@modal_function(
    image=BASE_IMAGE,
    volumes={
        CORPUS_MOUNT: CORPUS_VOLUME.read_only() if CORPUS_VOLUME is not None else None,
        ARTIFACTS_MOUNT: ARTIFACTS_VOLUME,
        HF_CACHE_MOUNT: HF_CACHE_VOLUME,
    },
    secrets=[HF_SECRET] if HF_SECRET is not None else [],
    timeout=24 * 60 * 60,
    cpu=8,
    gpu=os.environ.get("STYLEBENCH_MODAL_TRAIN_GPU", "A10G"),
)
def _train_style_scorer_remote(config: str = "configs/scorer_train_v1.json") -> dict[str, Any]:
    _reload_volume(ARTIFACTS_VOLUME)
    _reload_volume(HF_CACHE_VOLUME)
    runtime_config = _runtime_scorer_config(config)
    result = _train_style_scorer(runtime_config)
    _commit_volumes(ARTIFACTS_VOLUME, HF_CACHE_VOLUME)
    return result


@modal_function(
    image=BASE_IMAGE,
    volumes={
        CORPUS_MOUNT: CORPUS_VOLUME.read_only() if CORPUS_VOLUME is not None else None,
        ARTIFACTS_MOUNT: ARTIFACTS_VOLUME,
        HF_CACHE_MOUNT: HF_CACHE_VOLUME,
    },
    timeout=8 * 60 * 60,
    cpu=4,
)
def _calibrate_style_scorer_remote(config: str = "configs/scorer_train_v1.json") -> dict[str, Any]:
    _reload_volume(CORPUS_VOLUME)
    _reload_volume(ARTIFACTS_VOLUME)
    result = _calibrate_style_scorer(_runtime_scorer_config(config))
    _commit_volumes(ARTIFACTS_VOLUME)
    return result


@modal_function(
    image=BASE_IMAGE,
    volumes={
        CORPUS_MOUNT: CORPUS_VOLUME.read_only() if CORPUS_VOLUME is not None else None,
        ARTIFACTS_MOUNT: ARTIFACTS_VOLUME,
        HF_CACHE_MOUNT: HF_CACHE_VOLUME,
    },
    timeout=8 * 60 * 60,
    cpu=4,
)
def _build_benchmark_manifests_remote(
    benchmark_config: str = "configs/benchmark_v1.json",
    out_root: str = f"{ARTIFACTS_MOUNT}/benchmark/manifests",
) -> dict[str, Any]:
    _reload_volume(CORPUS_VOLUME)
    _reload_volume(ARTIFACTS_VOLUME)
    config = _runtime_benchmark_config(benchmark_config)
    config_path = Path(ARTIFACTS_MOUNT) / "benchmark" / "runtime_benchmark_config.json"
    write_json(config_path, config)
    result = _build_benchmark_manifests(
        books_manifest=Path(CORPUS_MOUNT) / "meta" / "books_manifest_v1.jsonl",
        out_dir=out_root,
        benchmark_config=config_path,
        split_root=Path(CORPUS_MOUNT) / "splits",
        scorer_dir=Path(ARTIFACTS_MOUNT) / "scorer" / "final",
    )
    _commit_volumes(ARTIFACTS_VOLUME)
    return result


@modal_function(
    image=BASE_IMAGE,
    volumes={
        ARTIFACTS_MOUNT: ARTIFACTS_VOLUME,
        HF_CACHE_MOUNT: HF_CACHE_VOLUME,
    },
    secrets=BENCHMARK_PROVIDER_SECRETS,
    timeout=24 * 60 * 60,
    cpu=4,
)
def _run_benchmark_remote(
    track: str,
    split: str,
    model: str,
    *,
    benchmark_config: str = "configs/benchmark_v1.json",
) -> str:
    _reload_volume(ARTIFACTS_VOLUME)
    config = _runtime_benchmark_config(benchmark_config)
    runtime_config_path = Path(ARTIFACTS_MOUNT) / "benchmark" / "runtime_benchmark_config.json"
    write_json(runtime_config_path, config)
    cases = Path(ARTIFACTS_MOUNT) / "benchmark" / "manifests" / f"cases_{track}_{split}_v1.jsonl"
    prompts = Path(ARTIFACTS_MOUNT) / "benchmark" / "manifests" / "prompts_v1.json"
    output = Path(ARTIFACTS_MOUNT) / "results" / "benchmark_runs" / f"{track}_{split}_{model.replace(':', '_')}.jsonl"
    result = _run_benchmark(
        track=track,
        split=split,
        cases_path=cases,
        prompts_path=prompts,
        model=model,
        model_dir=Path(ARTIFACTS_MOUNT) / "scorer" / "final",
        out_path=output,
        config_path=runtime_config_path,
        passages_path=Path(ARTIFACTS_MOUNT) / "benchmark" / "manifests" / f"passages_{track}_v1.jsonl",
    )
    _commit_volumes(ARTIFACTS_VOLUME)
    return result


@modal_function(
    image=BASE_IMAGE,
    volumes={ARTIFACTS_MOUNT: ARTIFACTS_VOLUME},
    timeout=2 * 60 * 60,
    cpu=2,
)
def _aggregate_benchmark_remote(input_path: str, out_path: Optional[str] = None) -> dict[str, Any]:
    _reload_volume(ARTIFACTS_VOLUME)
    source = Path(input_path if input_path.startswith("/") else f"{ARTIFACTS_MOUNT}/{input_path}")
    destination = Path(out_path if out_path and out_path.startswith("/") else f"{ARTIFACTS_MOUNT}/{out_path}") if out_path else source.with_suffix(".summary.json")
    result = _aggregate_benchmark_results(source, destination)
    _commit_volumes(ARTIFACTS_VOLUME)
    return result


@app.local_entrypoint()
def setup_modal_objects() -> None:  # pragma: no cover
    if modal is None:
        raise RuntimeError("Modal is not installed in the local environment.")
    modal.Volume.objects.create(CORPUS_VOLUME_NAME, allow_existing=True)
    modal.Volume.objects.create(ARTIFACTS_VOLUME_NAME, allow_existing=True)
    modal.Volume.objects.create(HF_CACHE_VOLUME_NAME, allow_existing=True)
    _print_result(
        {
            "corpus_volume": CORPUS_VOLUME_NAME,
            "artifacts_volume": ARTIFACTS_VOLUME_NAME,
            "hf_cache_volume": HF_CACHE_VOLUME_NAME,
            "hf_secret": HF_SECRET_NAME,
            "provider_secret_env": PROVIDER_SECRET_NAME_ENV,
            "provider_secret_keys": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        }
    )


@app.local_entrypoint()
def corpus_fetch_gutenberg_catalog(timeout: float = 90.0, url: str = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv") -> None:  # pragma: no cover
    _print_result(_corpus_fetch_catalog_remote.remote(url=url, timeout=timeout))


@app.local_entrypoint()
def build_gutenberg_catalog_index(catalog_path: Optional[str] = None) -> None:  # pragma: no cover
    _print_result(_build_catalog_index_remote.remote(catalog_path))


@app.local_entrypoint()
def corpus_fetch_gutenberg_http(start_id: int = 1, end_id: int = 1, timeout: float = 30.0) -> None:  # pragma: no cover
    _print_result(_corpus_fetch_http_chunk.remote(start_id, end_id, timeout=timeout))


@app.local_entrypoint()
def corpus_fetch_all_gutenberg_http(
    start_id: int = 1,
    end_id: int = 80000,
    chunk_size: int = 500,
    containers: int = 64,
    per_container_concurrency: int = 12,
    timeout: float = 30.0,
) -> None:  # pragma: no cover
    ranges = _chunk_ranges(start_id, end_id, chunk_size)
    aggregate = {"fetched": 0, "failed": 0, "chunks": len(ranges)}
    for batch_start in range(0, len(ranges), max(1, containers)):
        batch = ranges[batch_start : batch_start + max(1, containers)]
        for result in _corpus_fetch_http_chunk.starmap(
            [(range_start, range_end) for range_start, range_end in batch],
            kwargs={"timeout": timeout, "max_workers": per_container_concurrency},
        ):
            aggregate["fetched"] += result["fetched"]
            aggregate["failed"] += result["failed"]
    _print_result(aggregate)


@app.local_entrypoint()
def corpus_ingest_gutenberg_rsync(
    max_files: int = 0,
    rsync_source: str = "aleph.gutenberg.org::gutenberg",
    rsync_shards: int = 1,
    run_id: str = "",
) -> None:  # pragma: no cover
    requested_limit = None if max_files <= 0 else max_files
    resolved_run_id = run_id or _default_run_id("gutenberg_rsync")
    if rsync_shards <= 1:
        _print_result(
            _corpus_ingest_rsync_remote.remote(
                max_files=requested_limit,
                rsync_source=rsync_source,
                run_id=resolved_run_id,
            )
        )
        return
    aggregate = {
        "ok": True,
        "run_id": resolved_run_id,
        "shards": rsync_shards,
        "listed_count": 0,
        "available_count": 0,
        "selected_file_count": 0,
        "new_file_count": 0,
        "results": [],
    }
    for result in _corpus_ingest_rsync_remote.starmap(
        [(shard_index,) for shard_index in range(rsync_shards)],
        kwargs={
            "max_files": requested_limit,
            "rsync_source": rsync_source,
            "shard_count": rsync_shards,
            "run_id": resolved_run_id,
        },
        return_exceptions=True,
        wrap_returned_exceptions=False,
    ):
        if isinstance(result, Exception):
            result = {
                "ok": False,
                "run_id": resolved_run_id,
                "status": "error",
                "error_type": type(result).__name__,
                "error": str(result),
            }
        aggregate["ok"] = aggregate["ok"] and bool(result.get("ok"))
        aggregate["listed_count"] += int(result.get("listed_count", 0))
        aggregate["available_count"] = max(aggregate["available_count"], int(result.get("available_count", 0)))
        aggregate["selected_file_count"] += int(result.get("selected_file_count", 0))
        aggregate["new_file_count"] += int(result.get("new_file_count", 0))
        aggregate["results"].append(result)
    _print_result(aggregate)


@app.local_entrypoint()
def inspect_gutenberg_rsync_progress(run_id: str = "") -> None:  # pragma: no cover
    _print_result(_inspect_gutenberg_rsync_progress_remote.remote(run_id or None))


@app.local_entrypoint()
def inspect_gutenberg_rsync_inventory() -> None:  # pragma: no cover
    _print_result(_inspect_gutenberg_rsync_inventory_remote.remote())


@app.local_entrypoint()
def build_corpus_manifests(corpus_config: str = "configs/corpus_v1.json") -> None:  # pragma: no cover
    _print_result(_build_corpus_manifests_remote.remote(corpus_config))


@app.local_entrypoint()
def freeze_corpus_splits(corpus_config: str = "configs/corpus_v1.json") -> None:  # pragma: no cover
    _print_result(_freeze_corpus_splits_remote.remote(corpus_config))


@app.local_entrypoint()
def build_passage_pools(corpus_config: str = "configs/corpus_v1.json") -> None:  # pragma: no cover
    _print_result(_build_passage_pools_remote.remote(corpus_config))


@app.local_entrypoint()
def build_scorer_dataset(config: str = "configs/scorer_train_v1.json") -> None:  # pragma: no cover
    _print_result(_build_scorer_dataset_remote.remote(config))


@app.local_entrypoint()
def train_style_scorer(config: str = "configs/scorer_train_v1.json") -> None:  # pragma: no cover
    _print_result(_train_style_scorer_remote.remote(config))


@app.local_entrypoint()
def calibrate_style_scorer(config: str = "configs/scorer_train_v1.json") -> None:  # pragma: no cover
    _print_result(_calibrate_style_scorer_remote.remote(config))


@app.local_entrypoint()
def build_benchmark_manifests(benchmark_config: str = "configs/benchmark_v1.json") -> None:  # pragma: no cover
    _print_result(_build_benchmark_manifests_remote.remote(benchmark_config))


@app.local_entrypoint()
def run_benchmark(track: str = "author", split: str = "test", model: str = "stub:fixed_prose") -> None:  # pragma: no cover
    print(_run_benchmark_remote.remote(track, split, model))


@app.local_entrypoint()
def aggregate_benchmark(input: str, out: Optional[str] = None) -> None:  # pragma: no cover
    _print_result(_aggregate_benchmark_remote.remote(input, out))
