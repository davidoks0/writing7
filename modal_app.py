"""
Modal setup for preparing data and training the classifier(s).

Provides:
- prepare: generates datasets and stores them in a Modal Volume
- train / train_gpu: baseline classifier training (CPU or GPU)
- train_contrastive / train_contrastive_gpu: contrastive model training
- pipeline: prepare then baseline train on CPU

Quick examples:
- modal run modal_app.py::prepare -- --training-dir training
- modal run modal_app.py::train -- --model roberta-base --epochs 3
- modal run modal_app.py::train_gpu -- --model roberta-base --epochs 3
- modal run modal_app.py::train_contrastive -- --model roberta-base
- modal run modal_app.py::train_contrastive_gpu -- --model roberta-base
"""
import os
from pathlib import Path

import modal


APP_NAME = "writing7"
ARTIFACT_VOL_NAME = "writing7-artifacts"
TRAINING_VOL_NAME = "writing7-training"

app = modal.App(APP_NAME)
artifacts_vol = modal.Volume.from_name(ARTIFACT_VOL_NAME, create_if_missing=True)
training_vol = modal.Volume.from_name(TRAINING_VOL_NAME, create_if_missing=True)


# Images
image_cpu = (
    modal.Image.debian_slim()
    .apt_install("curl", "ca-certificates")
    .add_local_file("requirements.txt", "/workspace/requirements.txt", copy=True)
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .run_commands("ln -sf /root/.local/bin/uv /usr/local/bin/uv")
    .run_commands("uv pip install --system -r /workspace/requirements.txt")
    .run_commands("python -m spacy download en_core_web_sm || true")
    .add_local_dir("eval", "/workspace/eval")
    .add_local_dir("scripts", "/workspace/scripts")
    .add_local_file("standardize_training.py", "/workspace/standardize_training.py")
    .add_local_file("train.py", "/workspace/train.py")
    .add_local_file("train_contrastive.py", "/workspace/train_contrastive.py")
    .add_local_file("calibrate_contrastive.py", "/workspace/calibrate_contrastive.py")
    .add_local_file("evaluate_contrastive.py", "/workspace/evaluate_contrastive.py")
    .add_local_file("prepare_data.py", "/workspace/prepare_data.py")
    .add_local_file("inference.py", "/workspace/inference.py")
    .add_local_file("inference_contrastive.py", "/workspace/inference_contrastive.py")
    .add_local_file("inference_two_stage.py", "/workspace/inference_two_stage.py")
    .add_local_file("calibrate_style_similarity.py", "/workspace/calibrate_style_similarity.py")
    .add_local_file("style_map.py", "/workspace/style_map.py")
)

image_gpu = (
    modal.Image.debian_slim()
    .apt_install("curl", "ca-certificates")
    .add_local_file("requirements.txt", "/workspace/requirements.txt", copy=True)
    # Install uv and CUDA-enabled torch first (avoids downloading CPU torch)
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .run_commands("ln -sf /root/.local/bin/uv /usr/local/bin/uv")
    .run_commands(
        "uv pip install --system --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121"
    )
    # Then the rest of requirements (keeps existing torch)
    .run_commands("uv pip install --system -r /workspace/requirements.txt")
    .run_commands("python -m spacy download en_core_web_sm || true")
    .add_local_dir("eval", "/workspace/eval")
    .add_local_file("standardize_training.py", "/workspace/standardize_training.py")
    .add_local_file("train.py", "/workspace/train.py")
    .add_local_file("train_contrastive.py", "/workspace/train_contrastive.py")
    .add_local_file("calibrate_contrastive.py", "/workspace/calibrate_contrastive.py")
    .add_local_file("evaluate_contrastive.py", "/workspace/evaluate_contrastive.py")
    .add_local_file("prepare_data.py", "/workspace/prepare_data.py")
    .add_local_file("inference.py", "/workspace/inference.py")
    .add_local_file("inference_contrastive.py", "/workspace/inference_contrastive.py")
    .add_local_file("inference_two_stage.py", "/workspace/inference_two_stage.py")
    .add_local_file("calibrate_style_similarity.py", "/workspace/calibrate_style_similarity.py")
    .add_local_file("style_map.py", "/workspace/style_map.py")
)

# Lightweight data image for I/O and mirroring tools
image_data = (
    modal.Image.debian_slim()
    .pip_install("httpx")
    .apt_install("rsync", "curl")
    .add_local_file("standardize_training.py", "/workspace/standardize_training.py")
)


# ---- GPU specs (string form to avoid deprecation warnings) ----
def _gpu_str(kind: str, count: int = 1) -> str:
    """Return Modal GPU spec string, e.g. "H100:4" or "A100-40GB:8".

    Using string form avoids deprecated enum usage (`gpu=H100(...)`).
    """
    # Modal expects "A100-40GB" for A100 variants in string form
    name = "A100-40GB" if kind.upper() == "A100" else kind.upper()
    return f"{name}:{int(count)}"

# Precompute GPU specs used in decorators (strings)
_GPU_H200_1 = _gpu_str("H200", 1)
_GPU_H200_2 = _gpu_str("H200", 2)
_GPU_H200_4 = _gpu_str("H200", 4)
_GPU_H200_8 = _gpu_str("H200", 8)
_GPU_H100_4 = _gpu_str("H100", 4)
_GPU_H100_8 = _gpu_str("H100", 8)
_GPU_A100_4 = _gpu_str("A100", 4)
_GPU_A100_8 = _gpu_str("A100", 8)


def _ensure_dirs():
    os.makedirs("/vol/data/processed", exist_ok=True)
    os.makedirs("/vol/models/book_matcher", exist_ok=True)
    os.makedirs("/vol/models/book_matcher_contrastive", exist_ok=True)
    os.makedirs("/vol/hf", exist_ok=True)
    os.makedirs("/vol/tmp/prepare_shards", exist_ok=True)


COMMON_ENV = {
    "HF_HOME": "/vol/hf",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    # Reduce CUDA fragmentation per PyTorch docs when VRAM is tight (PyTorch>=2.4)
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
}


@app.function(
    image=image_cpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    timeout=60 * 60 * 12,
    cpu=8,
    memory=16384,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def prepare_remote(
    training_dir: str = "/input/training",
    chunk_size: int = 14,
    overlap: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_chunks_per_book: int = 800,
    use_hard_negatives: bool = True,
    use_embedding_hard_negatives: bool = True,
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    num_chunks_for_embed: int = 80,
    num_hard_negative_books: int = 50,
    n_positive_per_book: int = 20,
    n_negative_per_book: int = 40,
    random_neg_frac: float = 0.10,
    # Model-mined negatives (optional; CPU-heavy)
    use_model_mined_negatives: bool = True,
    miner_model: str = 'contrastive',
    miner_model_dir: str | None = None,
    n_mined_trials: int = 200,
    n_mined_keep: int = 20,
    # ANN chunk-level hard negatives (EXPERIMENTAL)
    use_ann_chunk_negatives: bool = False,
    ann_miner_model_dir: str | None = None,
    ann_k_neighbors: int = 20,
    ann_sim_threshold: float = 0.55,
    ann_prob_max: float = 0.20,
    ann_anchors_per_book: int = 120,
    ann_pool_samples_per_book: int = 200,
    ann_batch_size: int = 32,
    ann_max_negatives_per_book: int = 100,
    ann_max_total_negatives: int | None = None,
):
    import random
    import numpy as np
    import os
    from pathlib import Path as _Path
    from prepare_data import prepare_datasets

    _ensure_dirs()

    random.seed(42)
    np.random.seed(42)

    # Helpful diagnostics for empty directories
    try:
        print(f"Listing /input: {os.listdir('/input')}")
    except Exception:
        pass
    try:
        td = str(training_dir)
        print(f"Listing {td}: {os.listdir(td) if os.path.exists(td) else 'MISSING'}")
        num_txt = len(list(_Path(training_dir).rglob('*.txt')))
        print(f"Found {num_txt} *.txt files under {training_dir} (recursive)")
    except Exception:
        pass

    # Standardize raw texts into a cleaned mirror directory
    try:
        from standardize_training import clean_text as _clean_text
        src_dir = _Path(training_dir)
        dst_dir = _Path("/input/training_clean")
        dst_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(src_dir.rglob('*.txt'))
        cleaned = 0
        for src in files:
            rel = src.relative_to(src_dir)
            dst = dst_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                raw = src.read_text(encoding='utf-8', errors='ignore')
                txt = _clean_text(raw)
                dst.write_text(txt, encoding='utf-8')
                cleaned += 1
            except Exception:
                # On error, copy original
                import shutil as _sh
                dst.parent.mkdir(parents=True, exist_ok=True)
                _sh.copy2(src, dst)
        print(f"Standardized {cleaned}/{len(files)} files -> {dst_dir}")
        training_dir = str(dst_dir)
    except Exception as e:
        print(f"Standardization step skipped due to error: {e}")

    datasets = prepare_datasets(
        training_dir=Path(training_dir),
        chunk_size=chunk_size,
        overlap=overlap,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        max_chunks_per_book=max_chunks_per_book,
        use_hard_negatives=use_hard_negatives,
        use_embedding_hard_negatives=use_embedding_hard_negatives,
        embedding_model=embedding_model,
        num_chunks_for_embed=num_chunks_for_embed,
        num_hard_negative_books=num_hard_negative_books,
        n_positive_per_book=n_positive_per_book,
        n_negative_per_book=n_negative_per_book,
        random_neg_frac=random_neg_frac,
        use_model_mined_negatives=use_model_mined_negatives,
        miner_model=miner_model,
        miner_model_dir=miner_model_dir,
        n_mined_trials=n_mined_trials,
        n_mined_keep=n_mined_keep,
        use_ann_chunk_negatives=use_ann_chunk_negatives,
        ann_miner_model_dir=ann_miner_model_dir,
        ann_k_neighbors=ann_k_neighbors,
        ann_sim_threshold=ann_sim_threshold,
        ann_prob_max=ann_prob_max,
        ann_anchors_per_book=ann_anchors_per_book,
        ann_pool_samples_per_book=ann_pool_samples_per_book,
        ann_batch_size=ann_batch_size,
        ann_max_negatives_per_book=ann_max_negatives_per_book,
        ann_max_total_negatives=ann_max_total_negatives,
    )

    out_dir = "/vol/data/processed"
    datasets.save_to_disk(out_dir)
    print(f"Saved datasets to {out_dir}")


# GPU-accelerated variant of prepare to speed up embedding-based negatives
@app.function(
    image=image_gpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    gpu=_GPU_H200_1,
    cpu=8,
    timeout=60 * 60 * 12,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def prepare_remote_gpu(
    training_dir: str = "/input/training",
    chunk_size: int = 14,
    overlap: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_chunks_per_book: int = 800,
    use_hard_negatives: bool = True,
    use_embedding_hard_negatives: bool = True,
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    num_chunks_for_embed: int = 80,
    num_hard_negative_books: int = 50,
    n_positive_per_book: int = 20,
    n_negative_per_book: int = 40,
    random_neg_frac: float = 0.10,
    # Model-mined negatives (optional; CPU/GPU heavy)
    use_model_mined_negatives: bool = True,
    miner_model: str = 'contrastive',
    miner_model_dir: str | None = None,
    n_mined_trials: int = 200,
    n_mined_keep: int = 20,
    # ANN chunk-level hard negatives (EXPERIMENTAL)
    use_ann_chunk_negatives: bool = False,
    ann_miner_model_dir: str | None = None,
    ann_k_neighbors: int = 20,
    ann_sim_threshold: float = 0.55,
    ann_prob_max: float = 0.20,
    ann_anchors_per_book: int = 120,
    ann_pool_samples_per_book: int = 200,
    ann_batch_size: int = 32,
    ann_max_negatives_per_book: int = 100,
    ann_max_total_negatives: int | None = None,
):
    import random
    import numpy as np
    import os
    from pathlib import Path as _Path
    from prepare_data import prepare_datasets

    _ensure_dirs()

    random.seed(42)
    np.random.seed(42)

    # Helpful diagnostics for empty directories
    try:
        print(f"Listing /input: {os.listdir('/input')}")
    except Exception:
        pass
    try:
        td = str(training_dir)
        print(f"Listing {td}: {os.listdir(td) if os.path.exists(td) else 'MISSING'}")
        num_txt = len(list(_Path(training_dir).rglob('*.txt')))
        print(f"Found {num_txt} *.txt files under {training_dir} (recursive)")
    except Exception:
        pass

    # Standardize raw texts into a cleaned mirror directory
    try:
        from standardize_training import clean_text as _clean_text
        src_dir = _Path(training_dir)
        dst_dir = _Path("/input/training_clean")
        dst_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(src_dir.rglob('*.txt'))
        cleaned = 0
        for src in files:
            rel = src.relative_to(src_dir)
            dst = dst_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                raw = src.read_text(encoding='utf-8', errors='ignore')
                txt = _clean_text(raw)
                dst.write_text(txt, encoding='utf-8')
                cleaned += 1
            except Exception:
                # On error, copy original
                import shutil as _sh
                dst.parent.mkdir(parents=True, exist_ok=True)
                _sh.copy2(src, dst)
        print(f"Standardized {cleaned}/{len(files)} files -> {dst_dir}")
        training_dir = str(dst_dir)
    except Exception as e:
        print(f"Standardization step skipped due to error: {e}")

    datasets = prepare_datasets(
        training_dir=_Path(training_dir),
        chunk_size=chunk_size,
        overlap=overlap,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        max_chunks_per_book=max_chunks_per_book,
        use_hard_negatives=use_hard_negatives,
        use_embedding_hard_negatives=use_embedding_hard_negatives,
        embedding_model=embedding_model,
        num_chunks_for_embed=num_chunks_for_embed,
        num_hard_negative_books=num_hard_negative_books,
        n_positive_per_book=n_positive_per_book,
        n_negative_per_book=n_negative_per_book,
        random_neg_frac=random_neg_frac,
        use_model_mined_negatives=use_model_mined_negatives,
        miner_model=miner_model,
        miner_model_dir=miner_model_dir,
        n_mined_trials=n_mined_trials,
        n_mined_keep=n_mined_keep,
        use_ann_chunk_negatives=use_ann_chunk_negatives,
        ann_miner_model_dir=ann_miner_model_dir,
        ann_k_neighbors=ann_k_neighbors,
        ann_sim_threshold=ann_sim_threshold,
        ann_prob_max=ann_prob_max,
        ann_anchors_per_book=ann_anchors_per_book,
        ann_pool_samples_per_book=ann_pool_samples_per_book,
        ann_batch_size=ann_batch_size,
        ann_max_negatives_per_book=ann_max_negatives_per_book,
        ann_max_total_negatives=ann_max_total_negatives,
    )

    out_dir = "/vol/data/processed"
    datasets.save_to_disk(out_dir)
    print(f"Saved datasets to {out_dir}")


# --------------------- Sharded prepare across containers ---------------------
@app.function(
    image=image_cpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    timeout=60 * 60 * 12,
    cpu=8,
    memory=16384,
    max_containers=100,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def prepare_chunk_shard_remote(
    base_dir: str = "/input/training",
    rel_paths: list[str] | None = None,
    shard_index: int = 0,
    out_dir: str = "/vol/tmp/prepare_shards",
    chunk_size: int = 14,
    overlap: int = 4,
    max_chunks_per_book: int = 800,
    use_hard_negatives: bool = True,
    workers: int | None = 8,
):
    """Process a subset of books and write a shard JSONL to the volume."""
    from pathlib import Path as _Path
    from prepare_data import chunk_books_to_jsonl

    _ensure_dirs()
    bdir = _Path(base_dir)
    files = rel_paths or []
    out_path = _Path(out_dir) / f"shard_{int(shard_index):04d}.jsonl"
    print({
        "shard": int(shard_index),
        "base_dir": str(bdir),
        "files": len(files),
        "out": str(out_path),
    })
    stats = chunk_books_to_jsonl(
        base_dir=bdir,
        rel_paths=[str(p) for p in files],
        out_jsonl=out_path,
        chunk_size=int(chunk_size),
        overlap=int(overlap),
        max_chunks_per_book=int(max_chunks_per_book),
        use_hard_negatives=bool(use_hard_negatives),
        workers=(None if workers is None else int(workers)),
    )
    print({"shard": int(shard_index), **stats})
    return {"ok": True, **stats, "out": str(out_path)}


@app.function(
    image=image_gpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    gpu=_GPU_H200_1,
    timeout=60 * 60 * 12,
    cpu=8,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def prepare_merge_shards_remote_gpu(
    shards_dir: str = "/vol/tmp/prepare_shards",
    expected_shards: int | None = None,
    wait_timeout_s: int = 60 * 60 * 2,
    poll_interval_s: int = 10,
    manifest_path: str | None = None,
    # Build params (subset of prepare_remote_gpu)
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    use_hard_negatives: bool = True,
    use_embedding_hard_negatives: bool = True,
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    num_chunks_for_embed: int = 80,
    num_hard_negative_books: int = 50,
    n_positive_per_book: int = 20,
    n_negative_per_book: int = 40,
    # Optional advanced miners off by default for merge step
    use_model_mined_negatives: bool = False,
):
    """Wait for shard JSONLs, merge them, and finish dataset build on GPU."""
    import time as _time
    from pathlib import Path as _Path
    from prepare_data import load_shards_jsonl, prepare_datasets_from_prechunked

    _ensure_dirs()
    sdir = _Path(shards_dir)
    sdir.mkdir(parents=True, exist_ok=True)

    # Wait until all expected shards exist (if provided)
    start = _time.time()
    shards = []
    manifest_files = []
    if manifest_path:
        mp = _Path(manifest_path)
        if mp.exists():
            try:
                import json as _json
                data = _json.loads(mp.read_text(encoding='utf-8'))
                shard_list = [ _Path(p) for p in data.get('shards', []) ]
                manifest_files = [str(p) for p in shard_list]
                # Poll until all shards in manifest exist or timeout
                while True:
                    ready = [p for p in shard_list if p.exists()]
                    if len(ready) >= len(shard_list):
                        shards = ready
                        break
                    if (_time.time() - start) > wait_timeout_s:
                        print({
                            "warning": "wait_timeout exceeded while waiting for manifest shards",
                            "expected": len(shard_list),
                            "found": len(ready),
                        })
                        shards = ready
                        break
                    _time.sleep(max(1, int(poll_interval_s)))
            except Exception as e:
                print({"warning": f"manifest read failed: {e}", "manifest_path": str(mp)})

    if not shards:
        if expected_shards is not None and expected_shards > 0:
            while True:
                shards = sorted(p for p in sdir.glob("shard_*.jsonl") if p.is_file())
                if len(shards) >= int(expected_shards):
                    break
                if (_time.time() - start) > wait_timeout_s:
                    print({
                        "warning": "wait_timeout exceeded while waiting for shards",
                        "expected": int(expected_shards),
                        "found": len(shards),
                    })
                    break
                _time.sleep(max(1, int(poll_interval_s)))
        shards = shards or sorted(p for p in sdir.glob("shard_*.jsonl") if p.is_file())

    print({"shards_found": len(shards), "from_manifest": bool(manifest_files)})
    if not shards:
        raise RuntimeError("No shard files found; cannot merge.")

    book_chunks, book_metadata = load_shards_jsonl(shards)
    print({"books": len(book_chunks)})
    datasets = prepare_datasets_from_prechunked(
        book_chunks=book_chunks,
        book_metadata=book_metadata,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        use_hard_negatives=use_hard_negatives,
        use_embedding_hard_negatives=use_embedding_hard_negatives,
        embedding_model=embedding_model,
        num_chunks_for_embed=num_chunks_for_embed,
        num_hard_negative_books=num_hard_negative_books,
        n_positive_per_book=n_positive_per_book,
        n_negative_per_book=n_negative_per_book,
        use_model_mined_negatives=use_model_mined_negatives,
    )
    out_dir = "/vol/data/processed"
    datasets.save_to_disk(out_dir)
    print({"merged_saved_to": out_dir})
    return {"ok": True, "books": len(book_chunks), "shards": len(shards), "out": out_dir, "manifest": manifest_files}


@app.function(
    image=image_cpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    timeout=60 * 60 * 12,
    cpu=8,
    memory=16384,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def prepare_sharded_remote(
    training_dir: str = "/input/training",
    # Sharding controls
    containers: int = 100,
    per_container_workers: int = 8,
    # Pre-standardize all texts (slow). Default off; shard step cleans per file anyway.
    standardize: bool = False,
    # Chunking params
    chunk_size: int = 14,
    overlap: int = 4,
    max_chunks_per_book: int = 800,
    use_hard_negatives: bool = True,
    # Merge/build params
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    use_embedding_hard_negatives: bool = True,
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    num_chunks_for_embed: int = 80,
    num_hard_negative_books: int = 50,
    n_positive_per_book: int = 20,
    n_negative_per_book: int = 40,
    # Wait controls
    shard_wait_timeout_s: int = 60 * 60 * 2,
):
    """Sharded prepare: CPU chunking across containers + GPU merge/build.

    Writes shard JSONLs to /vol/tmp/prepare_shards and then completes on GPU.
    """
    from pathlib import Path as _Path
    import math as _math
    import os as _os

    _ensure_dirs()

    base_dir = str(training_dir)
    if standardize:
        # Optional: Pre-standardize into /input/training_clean. Note this is slow and typically unnecessary
        # because shard workers load and clean text before chunking.
        try:
            from standardize_training import clean_text as _clean_text
            src_dir = _Path(training_dir)
            dst_dir = _Path("/input/training_clean")
            dst_dir.mkdir(parents=True, exist_ok=True)
            files = sorted(src_dir.rglob('*.txt'))
            cleaned = 0
            for src in files:
                rel = src.relative_to(src_dir)
                dst = dst_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    raw = src.read_text(encoding='utf-8', errors='ignore')
                    txt = _clean_text(raw)
                    dst.write_text(txt, encoding='utf-8')
                    cleaned += 1
                except Exception:
                    import shutil as _sh
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    _sh.copy2(src, dst)
            print({"standardized": cleaned, "total": len(files)})
            base_dir = str(dst_dir)
        except Exception as e:
            print({"standardize_error": str(e)})

    # Enumerate files and build shards
    bdir = _Path(base_dir)
    all_files = sorted([p for p in bdir.rglob('*.txt') if p.is_file()])
    total = len(all_files)
    if total == 0:
        raise RuntimeError(f"No .txt files found under {base_dir}")
    # Relativize paths for portability
    rels = [str(p.relative_to(bdir)) for p in all_files]
    # Bound containers to [1, 100]
    k = int(max(1, min(int(containers), 100)))
    if k > total:
        k = total
    # Clean shard dir
    sdir = _Path("/vol/tmp/prepare_shards")
    sdir.mkdir(parents=True, exist_ok=True)
    for old in sdir.glob("shard_*.jsonl"):
        try:
            old.unlink()
        except Exception:
            pass
    per = int(_math.ceil(total / float(k)))
    print({"total_files": total, "containers": k, "per_container": per})

    # Dispatch shard jobs concurrently
    calls = []
    for i in range(k):
        start = i * per
        end = min(total, start + per)
        if start >= end:
            break
        sub = rels[start:end]
        c = prepare_chunk_shard_remote.spawn(
            base_dir=base_dir,
            rel_paths=sub,
            shard_index=i,
            out_dir=str(sdir),
            chunk_size=chunk_size,
            overlap=overlap,
            max_chunks_per_book=max_chunks_per_book,
            use_hard_negatives=use_hard_negatives,
            workers=per_container_workers,
        )
        calls.append(c)

    # Block on shard tasks to guarantee files are written before merge
    completed, failed = 0, 0
    results = []
    for idx, call in enumerate(calls):
        try:
            res = call.get(timeout=60 * 60 * 3)  # up to 3h per shard batch
            # Minimal logging to aid debugging
            try:
                print({"shard_done": idx, "books": int(res.get("books", 0)), "chunks": int(res.get("chunks", 0))})
            except Exception:
                pass
            completed += 1
            results.append(res)
        except Exception as e:
            print({"shard_error": idx, "error": str(e)})
            failed += 1

    # Compute how many shard files actually exist (best-effort; volumes may show updates on new mount)
    actual_shards = len(list(sdir.glob("shard_*.jsonl")))
    print({
        "shard_calls": len(calls),
        "completed": completed,
        "failed": failed,
        "files_found": actual_shards,
        "note": "files_found may be 0 in this container; GPU merge reads via fresh mount/manifest",
    })

    # Prefer using number of completed shard tasks as the expected count; the GPU merge will also wait/poll
    expected = completed if completed > 0 else None

    # Write a manifest listing shard file paths for the GPU merge to follow
    manifest_path = sdir / "shards_manifest.json"
    try:
        import json as _json
        shard_paths = [r.get("out") for r in results if r and r.get("out")]
        data = {"expected": completed, "shards": shard_paths}
        manifest_path.write_text(_json.dumps(data), encoding='utf-8')
        print({"manifest_written": str(manifest_path), "shards": len(shard_paths)})
    except Exception as e:
        print({"manifest_write_error": str(e)})

    # Run GPU merge/build; it will wait for expected shards (via manifest or glob) to appear on a fresh mount
    return prepare_merge_shards_remote_gpu.remote(
        shards_dir=str(sdir),
        expected_shards=expected,
        manifest_path=str(manifest_path),
        wait_timeout_s=int(shard_wait_timeout_s),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        use_hard_negatives=use_hard_negatives,
        use_embedding_hard_negatives=use_embedding_hard_negatives,
        embedding_model=embedding_model,
        num_chunks_for_embed=num_chunks_for_embed,
        num_hard_negative_books=num_hard_negative_books,
        n_positive_per_book=n_positive_per_book,
        n_negative_per_book=n_negative_per_book,
        use_model_mined_negatives=False,
    )


# Local entrypoint to run sharded prepare from CLI
@app.local_entrypoint()
def prepare_sharded(
    training_dir: str = "/input/training",
    containers: int = 100,
    per_container_workers: int = 8,
    chunk_size: int = 14,
    overlap: int = 4,
    max_chunks_per_book: int = 800,
    use_hard_negatives: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    use_embedding_hard_negatives: bool = True,
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    num_chunks_for_embed: int = 80,
    num_hard_negative_books: int = 50,
    n_positive_per_book: int = 20,
    n_negative_per_book: int = 40,
):
    return prepare_sharded_remote.remote(
        training_dir=training_dir,
        containers=containers,
        per_container_workers=per_container_workers,
        chunk_size=chunk_size,
        overlap=overlap,
        max_chunks_per_book=max_chunks_per_book,
        use_hard_negatives=use_hard_negatives,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        use_embedding_hard_negatives=use_embedding_hard_negatives,
        embedding_model=embedding_model,
        num_chunks_for_embed=num_chunks_for_embed,
        num_hard_negative_books=num_hard_negative_books,
        n_positive_per_book=n_positive_per_book,
        n_negative_per_book=n_negative_per_book,
    )


@app.function(
    image=image_cpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    timeout=60 * 60 * 12,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def train_remote(
    model_name: str = "roberta-base",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 500,
    data_subdir: str = "/vol/data/processed",
    output_subdir: str = "/vol/models/book_matcher",
):
    from train import train as train_fn

    _ensure_dirs()

    print("Starting training on Modal (CPU)...")
    trainer, test_results = train_fn(
        model_name=model_name,
        output_dir=output_subdir,
        data_dir=data_subdir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
    )
    print("Training complete.")
    print({"test_results": test_results})


@app.function(
    image=image_gpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    gpu=_GPU_H200_1,
    cpu=8,
    timeout=60 * 60 * 12,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def train_remote_gpu(
    model_name: str = "roberta-base",
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_steps: int = 500,
    data_subdir: str = "/vol/data/processed",
    output_subdir: str = "/vol/models/book_matcher",
    grad_checkpointing: bool = True,
):
    from train import train as train_fn

    _ensure_dirs()

    print("Starting training on Modal (GPU)...")
    trainer, test_results = train_fn(
        model_name=model_name,
        output_dir=output_subdir,
        data_dir=data_subdir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
    )
    print("GPU training complete.")
    print({"test_results": test_results})


@app.function(
    image=image_cpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    timeout=60 * 60 * 12,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def train_contrastive_remote(
    model_name: str = "roberta-large",
    num_epochs: int = 6,
    batch_size: int = 12,
    learning_rate: float = 1e-5,
    warmup_steps: int = 1000,
    use_style_features: bool = True,
    use_symmetric_features: bool = True,
    data_subdir: str = "/vol/data/processed",
    output_subdir: str = "/vol/models/book_matcher_contrastive",
    contrastive_weight: float = 0.3,
    # Distillation & calibration
    distill_from_cross: bool = True,
    teacher_model_dir: str = "/vol/models/book_matcher/final",
    distill_weight: float = 0.5,
    distill_temperature: float = 3.0,
    calibrate_for: str = "accuracy",
    target_acc: float | None = None,
    target_recall: float | None = 0.85,
    pooling: str = "attn",
    use_projection: bool = True,
    label_smoothing: float = 0.03,
    grad_accum_steps: int = 2,
    select_metric: str = "auc",
    classifier: str = "arcface",
    arcface_margin: float = 0.25,
    arcface_scale: float = 30.0,
    contrastive_mode: str = "supcon",
    max_length: int = 512,
    grad_checkpointing: bool = True,
    teacher_on_gpu: bool = False,
    # Tokenization workers
    tokenize_workers: int = 4,
    # Prep integration: run data prepare by default so users need no flags
    prepare_before_train: bool = True,
    prepare_training_dir: str = "/input/training",
    # Defaults aligned with pipeline_contrastive
    prep_chunk_size: int = 14,
    prep_overlap: int = 4,
    prep_train_ratio: float = 0.7,
    prep_val_ratio: float = 0.15,
    prep_max_chunks_per_book: int = 800,
    prep_use_hard_negatives: bool = True,
    prep_use_embedding_hard_negatives: bool = True,
    prep_embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    prep_num_chunks_for_embed: int = 80,
    prep_num_hard_negative_books: int = 50,
    prep_n_positive_per_book: int = 20,
    prep_n_negative_per_book: int = 40,
):
    from train_contrastive import train_contrastive as train_fn

    _ensure_dirs()

    # Optionally run prepare step if dataset is missing; enable experimental ANN miner if prior checkpoint exists
    if prepare_before_train:
        import os as _os
        # Skip if datasets already exist
        _ds_dir = data_subdir
        _ds_ready = _os.path.exists(_os.path.join(_ds_dir, 'train')) and _os.path.exists(_os.path.join(_ds_dir, 'validation'))
        if not _ds_ready:
            ann_dir = "/vol/models/book_matcher_contrastive/final"
            ann_ok = _os.path.exists(f"{ann_dir}/pytorch_model.bin") or _os.path.exists(f"{ann_dir}/model.safetensors")
            try:
                prepare_remote_gpu.remote(
                    training_dir=prepare_training_dir,
                    chunk_size=prep_chunk_size,
                    overlap=prep_overlap,
                    train_ratio=prep_train_ratio,
                    val_ratio=prep_val_ratio,
                    max_chunks_per_book=prep_max_chunks_per_book,
                    use_hard_negatives=prep_use_hard_negatives,
                    use_embedding_hard_negatives=prep_use_embedding_hard_negatives,
                    embedding_model=prep_embedding_model,
                    num_chunks_for_embed=prep_num_chunks_for_embed,
                    num_hard_negative_books=prep_num_hard_negative_books,
                    n_positive_per_book=prep_n_positive_per_book,
                    n_negative_per_book=prep_n_negative_per_book,
                    # Experimental ANN miner auto-enabled if a prior contrastive checkpoint exists
                    use_model_mined_negatives=ann_ok,
                    use_ann_chunk_negatives=ann_ok,
                    ann_miner_model_dir=(ann_dir if ann_ok else None),
                    miner_model_dir=(ann_dir if ann_ok else None),
                )
            except Exception as e:
                print({"warning": f"prepare_remote failed; retrying without miners: {e}"})
                try:
                    prepare_remote_gpu.remote(
                        training_dir=prepare_training_dir,
                        chunk_size=prep_chunk_size,
                        overlap=prep_overlap,
                        train_ratio=prep_train_ratio,
                        val_ratio=prep_val_ratio,
                        max_chunks_per_book=prep_max_chunks_per_book,
                        use_hard_negatives=prep_use_hard_negatives,
                        use_embedding_hard_negatives=prep_use_embedding_hard_negatives,
                        embedding_model=prep_embedding_model,
                        num_chunks_for_embed=prep_num_chunks_for_embed,
                        num_hard_negative_books=prep_num_hard_negative_books,
                        n_positive_per_book=prep_n_positive_per_book,
                        n_negative_per_book=prep_n_negative_per_book,
                        use_model_mined_negatives=False,
                        use_ann_chunk_negatives=False,
                        ann_miner_model_dir=None,
                        miner_model_dir=None,
                    )
                except Exception as e2:
                    print({"warning": f"prepare_remote second attempt failed: {e2}"})

    # Ensure teacher exists if distilling
    if distill_from_cross:
        import os as _os
        teacher_ok = _os.path.exists(f"{teacher_model_dir}/pytorch_model.bin") or _os.path.exists(f"{teacher_model_dir}/model.safetensors")
        if not teacher_ok:
            print("Teacher (cross-encoder) not found; training teacher first on GPU...")
            train_remote_gpu.remote(
                model_name="roberta-large",
                num_epochs=3,
                batch_size=8,
                learning_rate=1e-5,
                warmup_steps=500,
                data_subdir=data_subdir,
                output_subdir="/vol/models/book_matcher",
            )

    print("Starting contrastive training on Modal (CPU)...")
    trainer, test_results = train_fn(
        model_name=model_name,
        output_dir=output_subdir,
        data_dir=data_subdir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        use_style_features=use_style_features,
        use_symmetric_features=use_symmetric_features,
        contrastive_weight=contrastive_weight,
        teacher_model_dir=(teacher_model_dir if distill_from_cross else None),
        distill_weight=distill_weight,
        distill_temperature=distill_temperature,
        calibrate_for=calibrate_for,
        target_acc=target_acc,
        target_recall=target_recall,
        pooling=pooling,
        use_projection=use_projection,
        label_smoothing=label_smoothing,
        grad_accum_steps=grad_accum_steps,
        select_metric=select_metric,
        classifier=classifier,
        arcface_margin=arcface_margin,
        arcface_scale=arcface_scale,
        contrastive_mode=contrastive_mode,
        supcon_temperature=0.1,
        max_length=max_length,
        grad_checkpointing=grad_checkpointing,
        teacher_on_gpu=teacher_on_gpu,
        # Enable enhancements with conservative defaults
        multi_head_adversary=True,
        use_independence_penalty=True,
        independence_weight=0.1,
        adv_lambda=0.3,
        tokenize_workers=tokenize_workers,
    )
    print("Contrastive CPU training complete.")
    print({"test_results": test_results})


@app.function(
    image=image_gpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    gpu=_GPU_H200_1,
    timeout=60 * 60 * 12,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def train_contrastive_remote_gpu(
    model_name: str = "roberta-large",
    num_epochs: int = 6,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    warmup_steps: int = 1000,
    use_style_features: bool = True,
    use_symmetric_features: bool = True,
    data_subdir: str = "/vol/data/processed",
    output_subdir: str = "/vol/models/book_matcher_contrastive",
    contrastive_weight: float = 0.3,
    # Distillation & calibration
    distill_from_cross: bool = False,
    teacher_model_dir: str = "/vol/models/book_matcher/final",
    distill_weight: float = 0.5,
    distill_temperature: float = 3.0,
    calibrate_for: str = "accuracy",
    target_acc: float | None = None,
    target_recall: float | None = 0.85,
    pooling: str = "attn",
    use_projection: bool = True,
    label_smoothing: float = 0.03,
    grad_accum_steps: int = 2,
    select_metric: str = "auc",
    classifier: str = "arcface",
    arcface_margin: float = 0.25,
    arcface_scale: float = 30.0,
    contrastive_mode: str = "supcon",
    max_length: int = 384,
    grad_checkpointing: bool = True,
    teacher_on_gpu: bool = True,
    # Tokenization workers (None or <=0 uses all available cores)
    tokenize_workers: int | None = None,
    # Training/eval controls
    eval_strategy: str = 'epoch',
    eval_steps: int = 500,
    save_strategy: str = 'epoch',
    save_steps: int = 500,
    logging_steps: int = 100,
    eval_subset_size: int | None = None,
    disable_distillation: bool = True,
    # Prep integration: run data prepare by default so users need no flags
    prepare_before_train: bool = True,
    prepare_training_dir: str = "/input/training",
    # Defaults aligned with pipeline_contrastive
    prep_chunk_size: int = 14,
    prep_overlap: int = 4,
    prep_train_ratio: float = 0.7,
    prep_val_ratio: float = 0.15,
    prep_max_chunks_per_book: int = 800,
    prep_use_hard_negatives: bool = True,
    prep_use_embedding_hard_negatives: bool = True,
    prep_embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    prep_num_chunks_for_embed: int = 80,
    prep_num_hard_negative_books: int = 50,
    prep_n_positive_per_book: int = 20,
    prep_n_negative_per_book: int = 40,
):
    from train_contrastive import train_contrastive as train_fn

    _ensure_dirs()

    # Optionally run prepare step if dataset is missing; enable experimental ANN miner if prior checkpoint exists
    if prepare_before_train:
        import os as _os
        # Skip if datasets already exist
        _ds_dir = data_subdir
        _ds_ready = _os.path.exists(_os.path.join(_ds_dir, 'train')) and _os.path.exists(_os.path.join(_ds_dir, 'validation'))
        if not _ds_ready:
            ann_dir = "/vol/models/book_matcher_contrastive/final"
            ann_ok = _os.path.exists(f"{ann_dir}/pytorch_model.bin") or _os.path.exists(f"{ann_dir}/model.safetensors")
            try:
                prepare_remote_gpu.remote(
                    training_dir=prepare_training_dir,
                    chunk_size=prep_chunk_size,
                    overlap=prep_overlap,
                    train_ratio=prep_train_ratio,
                    val_ratio=prep_val_ratio,
                    max_chunks_per_book=prep_max_chunks_per_book,
                    use_hard_negatives=prep_use_hard_negatives,
                    use_embedding_hard_negatives=prep_use_embedding_hard_negatives,
                    embedding_model=prep_embedding_model,
                    num_chunks_for_embed=prep_num_chunks_for_embed,
                    num_hard_negative_books=prep_num_hard_negative_books,
                    n_positive_per_book=prep_n_positive_per_book,
                    n_negative_per_book=prep_n_negative_per_book,
                    # Experimental ANN miner auto-enabled if a prior contrastive checkpoint exists
                    use_model_mined_negatives=ann_ok,
                    use_ann_chunk_negatives=ann_ok,
                    ann_miner_model_dir=(ann_dir if ann_ok else None),
                    miner_model_dir=(ann_dir if ann_ok else None),
                )
            except Exception as e:
                print({"warning": f"prepare_remote failed; retrying without miners: {e}"})
                try:
                    prepare_remote_gpu.remote(
                        training_dir=prepare_training_dir,
                        chunk_size=prep_chunk_size,
                        overlap=prep_overlap,
                        train_ratio=prep_train_ratio,
                        val_ratio=prep_val_ratio,
                        max_chunks_per_book=prep_max_chunks_per_book,
                        use_hard_negatives=prep_use_hard_negatives,
                        use_embedding_hard_negatives=prep_use_embedding_hard_negatives,
                        embedding_model=prep_embedding_model,
                        num_chunks_for_embed=prep_num_chunks_for_embed,
                        num_hard_negative_books=prep_num_hard_negative_books,
                        n_positive_per_book=prep_n_positive_per_book,
                        n_negative_per_book=prep_n_negative_per_book,
                        use_model_mined_negatives=False,
                        use_ann_chunk_negatives=False,
                        ann_miner_model_dir=None,
                        miner_model_dir=None,
                    )
                except Exception as e2:
                    print({"warning": f"prepare_remote second attempt failed: {e2}"})

    # Ensure teacher exists if distilling
    if distill_from_cross:
        import os as _os
        teacher_ok = _os.path.exists(f"{teacher_model_dir}/pytorch_model.bin") or _os.path.exists(f"{teacher_model_dir}/model.safetensors")
        if not teacher_ok:
            print("Teacher (cross-encoder) not found; training teacher first on GPU...")
            train_remote_gpu.remote(
                model_name="roberta-large",
                num_epochs=3,
                batch_size=4,
                learning_rate=1e-5,
                warmup_steps=500,
                data_subdir=data_subdir,
                output_subdir="/vol/models/book_matcher",
            )

    print("Starting contrastive training on Modal (GPU)...")
    # Resolve tokenization worker count (use all vCPUs by default)
    try:
        import os as _os
        _auto_workers = max(1, int(_os.cpu_count() or 1))
    except Exception:
        _auto_workers = 1
    _tok_workers = _auto_workers if (tokenize_workers is None or int(tokenize_workers) <= 0) else int(tokenize_workers)
    trainer, test_results = train_fn(
        model_name=model_name,
        output_dir=output_subdir,
        data_dir=data_subdir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        use_style_features=use_style_features,
        use_symmetric_features=use_symmetric_features,
        contrastive_weight=contrastive_weight,
        teacher_model_dir=(teacher_model_dir if distill_from_cross else None),
        distill_weight=distill_weight,
        distill_temperature=distill_temperature,
        calibrate_for=calibrate_for,
        target_acc=target_acc,
        target_recall=target_recall,
        pooling=pooling,
        use_projection=use_projection,
        label_smoothing=label_smoothing,
        grad_accum_steps=grad_accum_steps,
        select_metric=select_metric,
        classifier=classifier,
        arcface_margin=arcface_margin,
        arcface_scale=arcface_scale,
        contrastive_mode=contrastive_mode,
        supcon_temperature=0.1,
        max_length=max_length,
        grad_checkpointing=grad_checkpointing,
        teacher_on_gpu=teacher_on_gpu,
        # Enable enhancements with conservative defaults
        multi_head_adversary=True,
        use_independence_penalty=True,
        independence_weight=0.1,
        adv_lambda=0.3,
        tokenize_workers=_tok_workers,
        compile_model=True,
        eval_strategy=eval_strategy,
        eval_steps=int(eval_steps),
        save_strategy=save_strategy,
        save_steps=int(save_steps),
        logging_steps=int(logging_steps),
        eval_subset_size=eval_subset_size,
        disable_distillation=bool(disable_distillation),
    )
    print("Contrastive GPU training complete.")
    print({"test_results": test_results})


# --------------------- Multi-GPU training via torchrun ----------------------
@app.function(
    image=image_gpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    gpu=_GPU_H200_2,
    timeout=60 * 60 * 12,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def train_contrastive_remote_multi_gpu(
    # Hardware: fixed 2x H200 allocation via decorator
    nproc_per_node: int = 2,
    # Core training params
    model_name: str = "roberta-large",
    num_epochs: int = 6,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    warmup_steps: int = 1000,
    use_style_features: bool = True,
    use_symmetric_features: bool = True,
    data_subdir: str = "/vol/data/processed",
    output_subdir: str = "/vol/models/book_matcher_contrastive",
    contrastive_weight: float = 0.3,
    # Distillation & calibration
    distill_from_cross: bool = False,
    teacher_model_dir: str = "/vol/models/book_matcher/final",
    distill_weight: float = 0.5,
    distill_temperature: float = 3.0,
    calibrate_for: str = "accuracy",
    target_acc: float | None = None,
    target_recall: float | None = 0.85,
    pooling: str = "attn",
    use_projection: bool = True,
    label_smoothing: float = 0.03,
    grad_accum_steps: int = 2,
    select_metric: str = "auc",
    classifier: str = "arcface",
    arcface_margin: float = 0.25,
    arcface_scale: float = 30.0,
    contrastive_mode: str = "supcon",
    max_length: int = 384,
    grad_checkpointing: bool = True,
    # Tokenization (None -> auto)
    tokenize_workers: int | None = None,
    # Optional: run prepare automatically if dataset missing
    prepare_before_train: bool = True,
    prepare_training_dir: str = "/input/training",
    # Prepare params (subset)
    prep_chunk_size: int = 14,
    prep_overlap: int = 4,
    prep_train_ratio: float = 0.7,
    prep_val_ratio: float = 0.15,
    prep_max_chunks_per_book: int = 800,
    prep_use_hard_negatives: bool = True,
    prep_use_embedding_hard_negatives: bool = True,
    prep_embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    prep_num_chunks_for_embed: int = 80,
    prep_num_hard_negative_books: int = 50,
    prep_n_positive_per_book: int = 20,
    prep_n_negative_per_book: int = 40,
):
    """Run contrastive training on 2 GPUs using torchrun with epoch-level eval/save.

    Notes:
    - Uses train_contrastive.py __main__ entry with defaults updated to epoch-level eval/save.
    - Tokenization workers default to all available CPU cores.
    - DDP performance tweaks are applied in TrainingArguments.
    """
    import os as _os
    import subprocess as _sp

    _ensure_dirs()

    # Optionally run prepare step if dataset is missing
    if prepare_before_train:
        _ds_dir = data_subdir
        _ds_ready = _os.path.exists(_os.path.join(_ds_dir, 'train')) and _os.path.exists(_os.path.join(_ds_dir, 'validation'))
        if not _ds_ready:
            ann_dir = "/vol/models/book_matcher_contrastive/final"
            ann_ok = _os.path.exists(f"{ann_dir}/pytorch_model.bin") or _os.path.exists(f"{ann_dir}/model.safetensors")
            try:
                prepare_remote_gpu.remote(
                    training_dir=prepare_training_dir,
                    chunk_size=prep_chunk_size,
                    overlap=prep_overlap,
                    train_ratio=prep_train_ratio,
                    val_ratio=prep_val_ratio,
                    max_chunks_per_book=prep_max_chunks_per_book,
                    use_hard_negatives=prep_use_hard_negatives,
                    use_embedding_hard_negatives=prep_use_embedding_hard_negatives,
                    embedding_model=prep_embedding_model,
                    num_chunks_for_embed=prep_num_chunks_for_embed,
                    num_hard_negative_books=prep_num_hard_negative_books,
                    n_positive_per_book=prep_n_positive_per_book,
                    n_negative_per_book=prep_n_negative_per_book,
                    use_model_mined_negatives=ann_ok,
                    use_ann_chunk_negatives=ann_ok,
                    ann_miner_model_dir=(ann_dir if ann_ok else None),
                    miner_model_dir=(ann_dir if ann_ok else None),
                )
            except Exception as e:
                print({"warning": f"prepare_remote failed; retrying without miners: {e}"})
                try:
                    prepare_remote_gpu.remote(
                        training_dir=prepare_training_dir,
                        chunk_size=prep_chunk_size,
                        overlap=prep_overlap,
                        train_ratio=prep_train_ratio,
                        val_ratio=prep_val_ratio,
                        max_chunks_per_book=prep_max_chunks_per_book,
                        use_hard_negatives=prep_use_hard_negatives,
                        use_embedding_hard_negatives=prep_use_embedding_hard_negatives,
                        embedding_model=prep_embedding_model,
                        num_chunks_for_embed=prep_num_chunks_for_embed,
                        num_hard_negative_books=prep_num_hard_negative_books,
                        n_positive_per_book=prep_n_positive_per_book,
                        n_negative_per_book=prep_n_negative_per_book,
                        use_model_mined_negatives=False,
                        use_ann_chunk_negatives=False,
                        ann_miner_model_dir=None,
                        miner_model_dir=None,
                    )
                except Exception as e2:
                    print({"warning": f"prepare_remote second attempt failed: {e2}"})

    # Ensure teacher exists if distilling; otherwise skip distillation
    teacher_arg = []
    if distill_from_cross:
        _teacher_ok = _os.path.exists(f"{teacher_model_dir}/pytorch_model.bin") or _os.path.exists(f"{teacher_model_dir}/model.safetensors")
        if not _teacher_ok:
            print("Teacher (cross-encoder) not found; training teacher first on GPU...")
            train_remote_gpu.remote(
                model_name="roberta-large",
                num_epochs=3,
                batch_size=4,
                learning_rate=1e-5,
                warmup_steps=500,
                data_subdir=data_subdir,
                output_subdir="/vol/models/book_matcher",
            )
            _teacher_ok = _os.path.exists(f"{teacher_model_dir}/pytorch_model.bin") or _os.path.exists(f"{teacher_model_dir}/model.safetensors")
        if _teacher_ok:
            teacher_arg = ["--teacher", str(teacher_model_dir)]

    # Determine tokenization workers
    try:
        _auto_workers = max(1, int(_os.cpu_count() or 1))
    except Exception:
        _auto_workers = 1
    _tok_workers = _auto_workers if (tokenize_workers is None or int(tokenize_workers) <= 0) else int(tokenize_workers)

    # Build torchrun command using the train_contrastive module's CLI
    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={int(nproc_per_node)}", "-m", "train_contrastive",
        "--model", str(model_name),
        "--output", str(output_subdir),
        "--data", str(data_subdir),
        "--epochs", str(int(num_epochs)),
        "--batch-size", str(int(batch_size)),
        "--lr", str(float(learning_rate)),
        "--grad-accum", str(int(grad_accum_steps)),
        "--label-smoothing", str(float(label_smoothing)),
        "--pooling", str(pooling),
        "--select-metric", str(select_metric),
        "--arcface-margin", str(float(arcface_margin)),
        "--arcface-scale", str(float(arcface_scale)),
        "--contrastive-mode", str(contrastive_mode),
        "--supcon-temperature", str(0.1),
        "--compile",
    ]
    # Optional flags
    if not use_style_features:
        cmd.append("--no-style-features")
    if not use_symmetric_features:
        cmd.append("--no-symmetric-head")
    if not use_projection:
        cmd.append("--no-projection")
    # grad_checkpointing is not exposed via CLI; rely on default in train_contrastive()
    # Tokenization workers via environment since CLI may not expose it
    env = {**_os.environ}
    env["TOKENIZE_WORKERS"] = str(int(_tok_workers))

    # Include teacher if available
    cmd.extend(teacher_arg)
    print({"torchrun": cmd, "TOKENIZE_WORKERS": env.get("TOKENIZE_WORKERS")})
    proc = _sp.Popen(cmd, env=env)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"torchrun exited with code {rc}")
    return {"status": "ok", "nproc": int(nproc_per_node), "output": str(output_subdir)}


# --------------------- 4x GPU (H100) via torchrun ---------------------------
@app.function(
    image=image_gpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    gpu=_GPU_H100_4,
    timeout=60 * 60 * 24,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def train_contrastive_remote_four_gpu_h100(
    # Core training params
    model_name: str = "roberta-large",
    num_epochs: int = 6,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    warmup_steps: int = 1000,
    use_style_features: bool = True,
    use_symmetric_features: bool = True,
    data_subdir: str = "/vol/data/processed",
    output_subdir: str = "/vol/models/book_matcher_contrastive",
    contrastive_weight: float = 0.3,
    # Distillation & calibration
    distill_from_cross: bool = False,
    teacher_model_dir: str = "/vol/models/book_matcher/final",
    # Tokenization (None -> auto)
    tokenize_workers: int | None = None,
    # Optional prepare
    prepare_before_train: bool = True,
    prepare_training_dir: str = "/input/training",
):
    import os as _os
    import subprocess as _sp
    _ensure_dirs()

    # Run prepare if needed
    if prepare_before_train:
        _ds_dir = data_subdir
        _ds_ready = _os.path.exists(_os.path.join(_ds_dir, 'train')) and _os.path.exists(_os.path.join(_ds_dir, 'validation'))
        if not _ds_ready:
            try:
                prepare_remote_gpu.remote(training_dir=prepare_training_dir)
            except Exception as e:
                print({"warning": f"prepare_remote failed: {e}"})

    # Tokenize workers
    try:
        _auto_workers = max(1, int(_os.cpu_count() or 1))
    except Exception:
        _auto_workers = 1
    _tok_workers = _auto_workers if (tokenize_workers is None or int(tokenize_workers) <= 0) else int(tokenize_workers)

    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=4", "-m", "train_contrastive",
        "--model", str(model_name),
        "--output", str(output_subdir),
        "--data", str(data_subdir),
        "--epochs", str(int(num_epochs)),
        "--batch-size", str(int(batch_size)),
        "--lr", str(float(learning_rate)),
        "--grad-accum", "2",
        "--label-smoothing", "0.03",
        "--pooling", "attn",
        "--select-metric", "auc",
        "--arcface-margin", "0.25",
        "--arcface-scale", "30.0",
        "--contrastive-mode", "supcon",
        "--supcon-temperature", "0.1",
        "--compile",
    ]
    env = {**_os.environ}
    env["TOKENIZE_WORKERS"] = str(int(_tok_workers))
    print({"torchrun": cmd, "TOKENIZE_WORKERS": env.get("TOKENIZE_WORKERS")})
    rc = _sp.call(cmd, env=env)
    if rc != 0:
        raise RuntimeError(f"torchrun (4x H100) exited with code {rc}")
    return {"status": "ok", "nproc": 4, "output": str(output_subdir)}


# --------------------- 4x GPU (H200) via torchrun ---------------------------
@app.function(
    image=image_gpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    gpu=_GPU_H200_4,
    timeout=60 * 60 * 24,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def train_contrastive_remote_four_gpu_h200(
    # Core training params
    model_name: str = "roberta-large",
    num_epochs: int = 6,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    warmup_steps: int = 1000,
    use_style_features: bool = True,
    use_symmetric_features: bool = True,
    data_subdir: str = "/vol/data/processed",
    output_subdir: str = "/vol/models/book_matcher_contrastive",
    contrastive_weight: float = 0.3,
    # Distillation & calibration
    distill_from_cross: bool = False,
    teacher_model_dir: str = "/vol/models/book_matcher/final",
    # Tokenization (None -> auto)
    tokenize_workers: int | None = None,
    # Optional prepare
    prepare_before_train: bool = True,
    prepare_training_dir: str = "/input/training",
):
    import os as _os
    import subprocess as _sp
    _ensure_dirs()

    if prepare_before_train:
        _ds_dir = data_subdir
        _ds_ready = _os.path.exists(_os.path.join(_ds_dir, 'train')) and _os.path.exists(_os.path.join(_ds_dir, 'validation'))
        if not _ds_ready:
            try:
                prepare_remote_gpu.remote(training_dir=prepare_training_dir)
            except Exception as e:
                print({"warning": f"prepare_remote failed: {e}"})

    try:
        _auto_workers = max(1, int(_os.cpu_count() or 1))
    except Exception:
        _auto_workers = 1
    _tok_workers = _auto_workers if (tokenize_workers is None or int(tokenize_workers) <= 0) else int(tokenize_workers)

    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=4", "-m", "train_contrastive",
        "--model", str(model_name),
        "--output", str(output_subdir),
        "--data", str(data_subdir),
        "--epochs", str(int(num_epochs)),
        "--batch-size", str(int(batch_size)),
        "--lr", str(float(learning_rate)),
        "--grad-accum", "2",
        "--label-smoothing", "0.03",
        "--pooling", "attn",
        "--select-metric", "auc",
        "--arcface-margin", "0.25",
        "--arcface-scale", "30.0",
        "--contrastive-mode", "supcon",
        "--supcon-temperature", "0.1",
        "--compile",
    ]
    env = {**_os.environ}
    env["TOKENIZE_WORKERS"] = str(int(_tok_workers))
    print({"torchrun": cmd, "TOKENIZE_WORKERS": env.get("TOKENIZE_WORKERS")})
    rc = _sp.call(cmd, env=env)
    if rc != 0:
        raise RuntimeError(f"torchrun (4x H200) exited with code {rc}")
    return {"status": "ok", "nproc": 4, "output": str(output_subdir)}


# --------------------- 4x GPU (A100) via torchrun ---------------------------
@app.function(
    image=image_gpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    gpu=_GPU_A100_4,
    timeout=60 * 60 * 24,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def train_contrastive_remote_four_gpu_a100(
    # Core training params
    model_name: str = "roberta-large",
    num_epochs: int = 6,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    warmup_steps: int = 1000,
    use_style_features: bool = True,
    use_symmetric_features: bool = True,
    data_subdir: str = "/vol/data/processed",
    output_subdir: str = "/vol/models/book_matcher_contrastive",
    contrastive_weight: float = 0.3,
    # Distillation & calibration
    distill_from_cross: bool = False,
    teacher_model_dir: str = "/vol/models/book_matcher/final",
    # Tokenization (None -> auto)
    tokenize_workers: int | None = None,
    # Optional prepare
    prepare_before_train: bool = True,
    prepare_training_dir: str = "/input/training",
):
    import os as _os
    import subprocess as _sp
    _ensure_dirs()

    if prepare_before_train:
        _ds_dir = data_subdir
        _ds_ready = _os.path.exists(_os.path.join(_ds_dir, 'train')) and _os.path.exists(_os.path.join(_ds_dir, 'validation'))
        if not _ds_ready:
            try:
                prepare_remote_gpu.remote(training_dir=prepare_training_dir)
            except Exception as e:
                print({"warning": f"prepare_remote failed: {e}"})

    try:
        _auto_workers = max(1, int(_os.cpu_count() or 1))
    except Exception:
        _auto_workers = 1
    _tok_workers = _auto_workers if (tokenize_workers is None or int(tokenize_workers) <= 0) else int(tokenize_workers)

    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=4", "-m", "train_contrastive",
        "--model", str(model_name),
        "--output", str(output_subdir),
        "--data", str(data_subdir),
        "--epochs", str(int(num_epochs)),
        "--batch-size", str(int(batch_size)),
        "--lr", str(float(learning_rate)),
        "--grad-accum", "2",
        "--label-smoothing", "0.03",
        "--pooling", "attn",
        "--select-metric", "auc",
        "--arcface-margin", "0.25",
        "--arcface-scale", "30.0",
        "--contrastive-mode", "supcon",
        "--supcon-temperature", "0.1",
        "--compile",
    ]
    env = {**_os.environ}
    env["TOKENIZE_WORKERS"] = str(int(_tok_workers))
    print({"torchrun": cmd, "TOKENIZE_WORKERS": env.get("TOKENIZE_WORKERS")})
    rc = _sp.call(cmd, env=env)
    if rc != 0:
        raise RuntimeError(f"torchrun (4x A100) exited with code {rc}")
    return {"status": "ok", "nproc": 4, "output": str(output_subdir)}


# --------------------- 8x GPU (H100) via torchrun ---------------------------
@app.function(
    image=image_gpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    gpu=_GPU_H100_8,
    timeout=60 * 60 * 24,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def train_contrastive_remote_eight_gpu_h100(
    # Core training params
    model_name: str = "roberta-large",
    num_epochs: int = 6,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    warmup_steps: int = 1000,
    use_style_features: bool = True,
    use_symmetric_features: bool = True,
    data_subdir: str = "/vol/data/processed",
    output_subdir: str = "/vol/models/book_matcher_contrastive",
    contrastive_weight: float = 0.3,
    # Distillation & calibration
    distill_from_cross: bool = False,
    teacher_model_dir: str = "/vol/models/book_matcher/final",
    # Tokenization (None -> auto)
    tokenize_workers: int | None = None,
    # Optional prepare
    prepare_before_train: bool = True,
    prepare_training_dir: str = "/input/training",
):
    import os as _os
    import subprocess as _sp
    _ensure_dirs()

    if prepare_before_train:
        _ds_dir = data_subdir
        _ds_ready = _os.path.exists(_os.path.join(_ds_dir, 'train')) and _os.path.exists(_os.path.join(_ds_dir, 'validation'))
        if not _ds_ready:
            try:
                prepare_remote_gpu.remote(training_dir=prepare_training_dir)
            except Exception as e:
                print({"warning": f"prepare_remote failed: {e}"})

    try:
        _auto_workers = max(1, int(_os.cpu_count() or 1))
    except Exception:
        _auto_workers = 1
    _tok_workers = _auto_workers if (tokenize_workers is None or int(tokenize_workers) <= 0) else int(tokenize_workers)

    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=8", "-m", "train_contrastive",
        "--model", str(model_name),
        "--output", str(output_subdir),
        "--data", str(data_subdir),
        "--epochs", str(int(num_epochs)),
        "--batch-size", str(int(batch_size)),
        "--lr", str(float(learning_rate)),
        "--grad-accum", "2",
        "--label-smoothing", "0.03",
        "--pooling", "attn",
        "--select-metric", "auc",
        "--arcface-margin", "0.25",
        "--arcface-scale", "30.0",
        "--contrastive-mode", "supcon",
        "--supcon-temperature", "0.1",
    ]
    env = {**_os.environ}
    env["TOKENIZE_WORKERS"] = str(int(_tok_workers))
    print({"torchrun": cmd, "TOKENIZE_WORKERS": env.get("TOKENIZE_WORKERS")})
    rc = _sp.call(cmd, env=env)
    if rc != 0:
        raise RuntimeError(f"torchrun (8x H100) exited with code {rc}")
    return {"status": "ok", "nproc": 8, "output": str(output_subdir)}


# --------------------- 8x GPU (H200) via torchrun ---------------------------
@app.function(
    image=image_gpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    gpu=_GPU_H200_8,
    timeout=60 * 60 * 24,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def train_contrastive_remote_eight_gpu_h200(
    # Core training params
    model_name: str = "roberta-large",
    num_epochs: int = 6,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    warmup_steps: int = 1000,
    use_style_features: bool = True,
    use_symmetric_features: bool = True,
    data_subdir: str = "/vol/data/processed",
    output_subdir: str = "/vol/models/book_matcher_contrastive",
    contrastive_weight: float = 0.3,
    # Distillation & calibration
    distill_from_cross: bool = False,
    teacher_model_dir: str = "/vol/models/book_matcher/final",
    # Tokenization (None -> auto)
    tokenize_workers: int | None = None,
    # Optimizer/throughput tuning
    grad_accum_steps: int = 2,
    # Optional prepare
    prepare_before_train: bool = True,
    prepare_training_dir: str = "/input/training",
):
    import os as _os
    import subprocess as _sp
    _ensure_dirs()

    if prepare_before_train:
        _ds_dir = data_subdir
        _ds_ready = _os.path.exists(_os.path.join(_ds_dir, 'train')) and _os.path.exists(_os.path.join(_ds_dir, 'validation'))
        if not _ds_ready:
            try:
                prepare_remote_gpu.remote(training_dir=prepare_training_dir)
            except Exception as e:
                print({"warning": f"prepare_remote failed: {e}"})

    try:
        _auto_workers = max(1, int(_os.cpu_count() or 1))
    except Exception:
        _auto_workers = 1
    _tok_workers = _auto_workers if (tokenize_workers is None or int(tokenize_workers) <= 0) else int(tokenize_workers)

    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=8", "-m", "train_contrastive",
        "--model", str(model_name),
        "--output", str(output_subdir),
        "--data", str(data_subdir),
        "--epochs", str(int(num_epochs)),
        "--batch-size", str(int(batch_size)),
        "--lr", str(float(learning_rate)),
        "--grad-accum", str(int(grad_accum_steps)),
        "--label-smoothing", "0.03",
        "--pooling", "attn",
        "--select-metric", "auc",
        "--arcface-margin", "0.25",
        "--arcface-scale", "30.0",
        "--contrastive-mode", "supcon",
        "--supcon-temperature", "0.1",
        "--compile",
    ]
    env = {**_os.environ}
    # Conservative setting that often improves DDP kernel scheduling on Hopper
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    env["TOKENIZE_WORKERS"] = str(int(_tok_workers))
    print({"torchrun": cmd, "TOKENIZE_WORKERS": env.get("TOKENIZE_WORKERS")})
    rc = _sp.call(cmd, env=env)
    if rc != 0:
        raise RuntimeError(f"torchrun (8x H200) exited with code {rc}")
    return {"status": "ok", "nproc": 8, "output": str(output_subdir)}


# --------------------- 8x GPU (A100) via torchrun ---------------------------
@app.function(
    image=image_gpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    gpu=_GPU_A100_8,
    timeout=60 * 60 * 24,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def train_contrastive_remote_eight_gpu_a100(
    # Core training params
    model_name: str = "roberta-large",
    num_epochs: int = 6,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    warmup_steps: int = 1000,
    use_style_features: bool = True,
    use_symmetric_features: bool = True,
    data_subdir: str = "/vol/data/processed",
    output_subdir: str = "/vol/models/book_matcher_contrastive",
    contrastive_weight: float = 0.3,
    # Distillation & calibration
    distill_from_cross: bool = False,
    teacher_model_dir: str = "/vol/models/book_matcher/final",
    # Tokenization (None -> auto)
    tokenize_workers: int | None = None,
    # Optimizer/throughput tuning
    grad_accum_steps: int = 2,
    # Optional prepare
    prepare_before_train: bool = True,
    prepare_training_dir: str = "/input/training",
):
    import os as _os
    import subprocess as _sp
    _ensure_dirs()

    if prepare_before_train:
        _ds_dir = data_subdir
        _ds_ready = _os.path.exists(_os.path.join(_ds_dir, 'train')) and _os.path.exists(_os.path.join(_ds_dir, 'validation'))
        if not _ds_ready:
            try:
                prepare_remote_gpu.remote(training_dir=prepare_training_dir)
            except Exception as e:
                print({"warning": f"prepare_remote failed: {e}"})

    try:
        _auto_workers = max(1, int(_os.cpu_count() or 1))
    except Exception:
        _auto_workers = 1
    _tok_workers = _auto_workers if (tokenize_workers is None or int(tokenize_workers) <= 0) else int(tokenize_workers)

    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=8", "-m", "train_contrastive",
        "--model", str(model_name),
        "--output", str(output_subdir),
        "--data", str(data_subdir),
        "--epochs", str(int(num_epochs)),
        "--batch-size", str(int(batch_size)),
        "--lr", str(float(learning_rate)),
        "--grad-accum", str(int(grad_accum_steps)),
        "--label-smoothing", "0.03",
        "--pooling", "attn",
        "--select-metric", "auc",
        "--arcface-margin", "0.25",
        "--arcface-scale", "30.0",
        "--contrastive-mode", "supcon",
        "--supcon-temperature", "0.1",
        "--compile",
    ]
    env = {**_os.environ}
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    env["TOKENIZE_WORKERS"] = str(int(_tok_workers))
    print({"torchrun": cmd, "TOKENIZE_WORKERS": env.get("TOKENIZE_WORKERS")})
    rc = _sp.call(cmd, env=env)
    if rc != 0:
        raise RuntimeError(f"torchrun (8x A100) exited with code {rc}")
    return {"status": "ok", "nproc": 8, "output": str(output_subdir)}


# -------------------------- UMAP & Benchmarking -----------------------------
@app.function(
    image=image_cpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    timeout=60 * 60 * 6,
    cpu=8,
    memory=16384,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def umap_benchmark_remote(
    # If provided, use a precomputed embedding matrix (N x D, float32) at this path
    embeddings_npy: str | None = None,
    # Otherwise, embed books on the fly using the contrastive model
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    books_dir: str = "/input/training",
    max_books: int = 500,
    num_chunks: str = "auto",  # or an int e.g. "6"
    chunk_size: int = 14,
    overlap: int = 4,
    embed_batch_size: int = 256,
    use_projection: bool = False,
    # UMAP params
    n_neighbors: int = 50,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_seed: int = 42,
    # Outputs
    out_dir: str = "/vol/umap",
    save_plot: bool = True,
):
    """Run UMAP on embeddings and report timing and throughput.

    If embeddings_npy is not provided, this will embed up to `max_books` books from
    `books_dir` using the trained model in `model_dir` and then run UMAP.
    """
    import os as _os
    import time as _time
    import numpy as _np
    from pathlib import Path as _Path

    _ensure_dirs()
    _Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load or compute embeddings
    labels = None
    X = None
    if embeddings_npy and _os.path.exists(embeddings_npy):
        print({"load": embeddings_npy})
        X = _np.load(embeddings_npy)
        N, D = int(X.shape[0]), int(X.shape[1])
        labels = [f"item_{i}" for i in range(N)]
    else:
        print({"embed": {"model": model_dir, "books_dir": books_dir}})
        from pathlib import Path as _P
        from style_map import _prepare_book_chunks, embed_books_batched, reduce_to_nd, plot_map
        from inference_contrastive import ContrastiveBookMatcherInference

        model = ContrastiveBookMatcherInference(model_dir)
        files = sorted([p for p in _P(books_dir).glob("**/*.txt") if p.is_file()])
        if max_books and len(files) > max_books:
            files = files[: int(max_books)]
        if not files:
            raise RuntimeError(f"No .txt files found under {books_dir}")
        # Lightweight text reader
        def _read_text(path: _P, max_chars: int | None = None) -> str:
            try:
                txt = path.read_text(encoding="utf-8", errors="ignore")
                return txt if not max_chars else txt[:max_chars]
            except Exception:
                return ""
        # Prepare chunks per book
        books = []
        for p in files:
            txt = _read_text(p, None)
            chs = _prepare_book_chunks(txt, num_chunks=num_chunks, chunk_size=chunk_size, overlap=overlap)
            if chs:
                books.append((p.stem, chs))
        labels, X = embed_books_batched(
            model,
            books,
            max_length=512,
            embed_batch_size=int(embed_batch_size),
            use_projection=bool(use_projection),
            book_pool="trimmed_mean",
            trim_frac=0.2,
            book_topk=5,
        )
        if X.size == 0:
            raise RuntimeError("No embeddings computed; check inputs")
        N, D = int(X.shape[0]), int(X.shape[1])

    # UMAP reduce (CPU via umap-learn in this image)
    from style_map import reduce_to_nd, plot_map
    t0 = _time.perf_counter()
    Y = reduce_to_nd(
        X,
        method="umap",
        n_components=2,
        pca_dim=50,
        perplexity=30.0,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=str(metric),
        densmap=False,
        random_state=int(random_seed),
    )
    t1 = _time.perf_counter()
    sec = t1 - t0
    thr = float(N) / sec if sec > 0 else 0.0
    out_npy = str(_Path(out_dir) / "umap_2d.npy")
    _np.save(out_npy, Y)
    print({
        "umap_done": True,
        "N": int(N),
        "D": int(D),
        "seconds": round(sec, 3),
        "points_per_sec": round(thr, 1),
        "out_npy": out_npy,
    })

    # Optional quick plot
    if save_plot and labels is not None and len(labels) == Y.shape[0]:
        try:
            out_png = str(_Path(out_dir) / "umap.png")
            plot_map(Y, labels, _Path(out_png), clusters=None)
            print({"plot": out_png})
        except Exception as e:
            print({"plot_error": str(e)})

    return {"ok": True, "N": int(N), "D": int(D), "sec": sec, "pps": thr, "out": out_npy}


@app.local_entrypoint()
def umap_benchmark(
    embeddings_npy: str | None = None,
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    books_dir: str = "/input/training",
    max_books: int = 500,
    num_chunks: str = "auto",
    chunk_size: int = 14,
    overlap: int = 4,
    embed_batch_size: int = 256,
    use_projection: bool = False,
    n_neighbors: int = 50,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_seed: int = 42,
    out_dir: str = "/vol/umap",
    save_plot: bool = True,
):
    return umap_benchmark_remote.remote(
        embeddings_npy=embeddings_npy,
        model_dir=model_dir,
        books_dir=books_dir,
        max_books=max_books,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        overlap=overlap,
        embed_batch_size=embed_batch_size,
        use_projection=use_projection,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_seed=random_seed,
        out_dir=out_dir,
        save_plot=save_plot,
    )


@app.function(
    image=image_gpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    gpu=_GPU_H200_1,
    timeout=60 * 60 * 6,
    cpu=8,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def compute_book_embeddings_remote_gpu(
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    books_dir: str = "/input/training",
    out_path: str = "/vol/umap/book_embeddings.npy",
    max_books: int = 10000,
    num_chunks: str = "auto",
    chunk_size: int = 14,
    overlap: int = 4,
    embed_batch_size: int = 512,
    use_projection: bool = False,
    max_chars: int | None = None,
):
    """Embed books to an N x D matrix on GPU and save to .npy.

    Pools chunk embeddings per book (trimmed mean). Intended to feed UMAP.
    """
    import os as _os
    import numpy as _np
    from pathlib import Path as _P
    from style_map import _prepare_book_chunks, embed_books_batched
    from inference_contrastive import ContrastiveBookMatcherInference

    _ensure_dirs()
    _P(_os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

    model = ContrastiveBookMatcherInference(model_dir)
    files = sorted([p for p in _P(books_dir).glob("**/*.txt") if p.is_file()])
    if max_books and len(files) > max_books:
        files = files[: int(max_books)]
    if not files:
        raise RuntimeError(f"No .txt files under {books_dir}")

    def _read_text(path: _P, max_chars: int | None = None) -> str:
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
            return txt if not max_chars else txt[:max_chars]
        except Exception:
            return ""

    books = []
    for p in files:
        txt = _read_text(p, max_chars=max_chars)
        chs = _prepare_book_chunks(txt, num_chunks=num_chunks, chunk_size=chunk_size, overlap=overlap)
        if chs:
            books.append((p.stem, chs))
    labels, X = embed_books_batched(
        model,
        books,
        max_length=512,
        embed_batch_size=int(embed_batch_size),
        use_projection=bool(use_projection),
        book_pool="trimmed_mean",
        trim_frac=0.2,
        book_topk=5,
    )
    if X.size == 0:
        raise RuntimeError("No embeddings computed")
    _np.save(out_path, X.astype(_np.float32, copy=False))
    print({"saved": out_path, "N": int(X.shape[0]), "D": int(X.shape[1])})
    # Also write labels CSV alongside
    try:
        import csv as _csv
        lab_path = _P(out_path).with_suffix('.labels.csv')
        with lab_path.open('w', newline='', encoding='utf-8') as f:
            w = _csv.writer(f)
            w.writerow(["label"]) 
            for lab in labels:
                w.writerow([lab])
        print({"labels": str(lab_path)})
    except Exception as e:
        print({"labels_error": str(e)})
    return {"ok": True, "out": out_path, "N": int(X.shape[0]), "D": int(X.shape[1])}


@app.function(
    image=image_cpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    timeout=60 * 60 * 12,
    cpu=8,
    memory=16384,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def calibrate_contrastive_remote(
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    data_subdir: str = "/vol/data/processed",
    calibrate_for: str = "accuracy",
    target_acc: float | None = None,
    target_recall: float | None = 0.85,
    save_to: str | None = None,
    batch_size: int = 128,
    num_proc: int = 4,
    num_workers: int = 2,
    max_length: int = 512,
):
    from calibrate_contrastive import calibrate_contrastive as _cal
    _ensure_dirs()
    _cal(
        model_dir=model_dir,
        data_dir=data_subdir,
        calibrate_for=calibrate_for,
        target_acc=target_acc,
        target_recall=target_recall,
        save_to=save_to,
        batch_size=batch_size,
        num_proc=num_proc,
        num_workers=num_workers,
        max_length=max_length,
    )
    return {"status": "calibration complete", "model_dir": model_dir}


@app.function(
    image=image_gpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    gpu=_GPU_H200_1,
    cpu=8,
    timeout=60 * 60 * 12,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def calibrate_contrastive_remote_gpu(
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    data_subdir: str = "/vol/data/processed",
    calibrate_for: str = "accuracy",
    target_acc: float | None = None,
    target_recall: float | None = 0.85,
    save_to: str | None = None,
    batch_size: int = 512,
    num_proc: int = 4,
    num_workers: int = 4,
    max_length: int = 512,
):
    from calibrate_contrastive import calibrate_contrastive as _cal
    _ensure_dirs()
    _cal(
        model_dir=model_dir,
        data_dir=data_subdir,
        calibrate_for=calibrate_for,
        target_acc=target_acc,
        target_recall=target_recall,
        save_to=save_to,
        batch_size=batch_size,
        num_proc=num_proc,
        num_workers=num_workers,
    )
    return {"status": "calibration complete (GPU)", "model_dir": model_dir}


@app.function(
    image=image_cpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    timeout=60 * 60 * 12,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def evaluate_contrastive_remote(
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    data_subdir: str = "/vol/data/processed",
    calibration_path: str | None = None,
    max_length: int = 512,
):
    from evaluate_contrastive import evaluate_contrastive as _eval
    _ensure_dirs()
    return _eval(model_dir=model_dir, data_dir=data_subdir, calibration_path=calibration_path, max_length=max_length)


@app.function(
    image=image_cpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    timeout=60 * 60 * 12,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def style_similarity_remote(
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    text1: str = "",
    text2: str = "",
    num_chunks = 'auto',
    chunk_size: int = 14,
    overlap: int = 4,
    aggregate: str = "mean",
    topk: int = 5,
    max_length: int = 512,
):
    """Compute cosine-based style similarity remotely (CPU).

    Returns a dict with cosine in [-1,1], naive [0,1] mapping, aggregation info and pair count.
    """
    from inference_contrastive import ContrastiveBookMatcherInference
    import math
    _ensure_dirs()
    matcher = ContrastiveBookMatcherInference(model_dir)
    res = matcher.style_similarity(
        text1,
        text2,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        overlap=overlap,
        aggregate=aggregate,
        topk=topk,
        max_length=max_length,
    )
    cos = float(res.get("cosine", float("nan")))
    mapped = (cos + 1.0) / 2.0 if math.isfinite(cos) else float("nan")
    out = {
        "cosine": cos,
        "score_0_1": mapped,
        "score_calibrated": (float(res.get("calibrated")) if res.get("calibrated") is not None else None),
        "aggregate": res.get("aggregate"),
        "pairs": int(res.get("pairs", 0)),
    }
    print(out)
    return out


@app.function(
    image=image_gpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    gpu=_GPU_H200_1,
    timeout=60 * 60 * 12,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def style_similarity_remote_gpu(
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    text1: str = "",
    text2: str = "",
    num_chunks = 'auto',
    chunk_size: int = 14,
    overlap: int = 4,
    aggregate: str = "mean",
    topk: int = 5,
    max_length: int = 512,
):
    """Compute cosine-based style similarity remotely (GPU)."""
    from inference_contrastive import ContrastiveBookMatcherInference
    import math
    _ensure_dirs()
    matcher = ContrastiveBookMatcherInference(model_dir)
    res = matcher.style_similarity(
        text1,
        text2,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        overlap=overlap,
        aggregate=aggregate,
        topk=topk,
        max_length=max_length,
    )
    cos = float(res.get("cosine", float("nan")))
    mapped = (cos + 1.0) / 2.0 if math.isfinite(cos) else float("nan")
    out = {
        "cosine": cos,
        "score_0_1": mapped,
        "score_calibrated": (float(res.get("calibrated")) if res.get("calibrated") is not None else None),
        "aggregate": res.get("aggregate"),
        "pairs": int(res.get("pairs", 0)),
    }
    print(out)
    return out


@app.function(
    image=image_cpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    timeout=60 * 60,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def count_training_texts_remote(training_dir: str = "/input/training"):
    """Count .txt files under the training_dir inside the Modal volume (recursive)."""
    from pathlib import Path as _Path
    import os as _os
    td = _Path(training_dir)
    try:
        n = len(list(td.rglob("*.txt")))
    except Exception as e:
        print({"error": str(e), "training_dir": str(td)})
        n = 0
    try:
        listing = _os.listdir(str(td)) if td.exists() else []
    except Exception:
        listing = []
    print({"training_dir": str(td), "exists": td.exists(), "top_level": listing[:10], "count": n})
    return {"training_dir": str(td), "exists": td.exists(), "count": int(n)}


@app.function(
    image=image_cpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    timeout=60 * 60 * 12,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def style_map_remote(
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    books_dir: str = "/input/training",
    out_prefix: str = "/vol/style_maps/style_map",
    method: str = "umap",  # or 'pca' or 'tsne'
    perplexity: float = 30.0,
    pca_dim: int = 50,
    seed: int = 42,
    max_books: int = 300,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    densmap: bool = False,
    remove_top_pcs: int = 0,
    n_components: int = 2,
    # Robust pooling & projection toggle
    book_pool: str = "trimmed_mean",
    trim_frac: float = 0.2,
    book_topk: int = 5,
    use_projection: bool = False,
    # Optional cluster coloring and pruning
    cluster_method: str = "kmeans",
    n_clusters: int = 12,
    hdbscan_min_cluster_size: int = 10,
    hdbscan_min_samples: int | None = None,
    drop_outliers: bool = False,
    prune_top_pct: float = 0.02,
    prune_knn_k: int = 10,
    num_chunks = 'auto',
    chunk_size: int = 14,
    overlap: int = 4,
    max_chars: int | None = None,
    embed_batch_size: int = 64,
    auto_tune_embed_batch: bool = False,
    max_length: int = 384,
    interactive_html: bool = True,
):
    """Generate style maps (CSV/PNG + HTML) for books under books_dir (CPU).

    Produces both 2D and 3D interactive HTML by default and colors by clusters.
    """
    from pathlib import Path as _Path
    from inference_contrastive import ContrastiveBookMatcherInference
    from style_map import _prepare_book_chunks, embed_books_batched, reduce_to_nd, plot_map, cluster_and_prune
    import csv as _csv
    import os as _os

    _ensure_dirs()
    _os.makedirs(_Path(out_prefix).parent, exist_ok=True)

    matcher = ContrastiveBookMatcherInference(model_dir)
    bdir = _Path(books_dir)
    files = sorted([p for p in bdir.glob("**/*.txt") if p.is_file()])
    if max_books and len(files) > max_books:
        files = files[:max_books]

    books = []
    for p in files:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if max_chars and len(txt) > max_chars:
                txt = txt[:max_chars]
        except Exception:
            continue
        chunks = _prepare_book_chunks(txt, num_chunks=num_chunks, chunk_size=chunk_size, overlap=overlap)
        books.append((p.stem, chunks))

    labels, X = embed_books_batched(
        matcher,
        books,
        max_length=int(max_length),
        embed_batch_size=int(embed_batch_size),
        auto_tune_embed_batch=bool(auto_tune_embed_batch),
        use_projection=use_projection,
        book_pool=book_pool,
        trim_frac=trim_frac,
        book_topk=book_topk,
    )
    if X.size == 0:
        return {"status": "no-embeddings", "books": 0}
    # Optional dominant component removal
    if remove_top_pcs and remove_top_pcs > 0:
        from style_map import _remove_top_pcs as __rtp
        X = __rtp(X, int(remove_top_pcs))

    # Optional clustering/pruning before projection
    Xf, clusters, keep_idx = cluster_and_prune(
        X,
        cluster_method=cluster_method,
        drop_outliers=bool(drop_outliers),
        hdbscan_min_cluster_size=int(hdbscan_min_cluster_size),
        hdbscan_min_samples=(None if hdbscan_min_samples is None else int(hdbscan_min_samples)),
        prune_top_pct=float(prune_top_pct),
        prune_knn_k=int(prune_knn_k),
        seed=int(seed),
    )
    if Xf.shape[0] != X.shape[0]:
        labels = [labels[i] for i in keep_idx.tolist()]
        X = Xf

    coords = reduce_to_nd(
        X,
        method=method,
        n_components=n_components,
        pca_dim=pca_dim,
        perplexity=perplexity,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        densmap=densmap,
        random_state=seed,
    )

    # Save CSV/PNG/HTML for primary (n_components)
    out_csv = _Path(out_prefix).with_suffix(".csv")
    # Optional clusters in embedding space
    if clusters is None and cluster_method == "kmeans":
        try:
            from sklearn.cluster import KMeans
            k = max(2, int(n_clusters))
            clusters = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        except Exception as e:
            print(f"Clustering failed: {e}")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        if n_components == 3:
            cols = ["label", "x", "y", "z"]
            if clusters is not None:
                cols.append("cluster")
            w.writerow(cols)
            for i, lab in enumerate(labels):
                x, y, z = coords[i, 0], coords[i, 1], coords[i, 2]
                row = [lab, float(x), float(y), float(z)]
                if clusters is not None:
                    row.append(int(clusters[i]))
                w.writerow(row)
        else:
            cols = ["label", "x", "y"]
            if clusters is not None:
                cols.append("cluster")
            w.writerow(cols)
            for i, lab in enumerate(labels):
                x, y = coords[i, 0], coords[i, 1]
                row = [lab, float(x), float(y)]
                if clusters is not None:
                    row.append(int(clusters[i]))
                w.writerow(row)

    out_png = _Path(out_prefix).with_suffix(".png")
    if n_components == 2:
        plot_map(coords, labels, out_png, clusters=clusters)
    # Optional interactive HTML (2D or 3D)
    if interactive_html:
        try:
            import plotly.express as px
            out_html = _Path(out_prefix).with_suffix(".html")
            if n_components == 3:
                fig = px.scatter_3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], hover_name=labels, color=(clusters if clusters is not None else None), height=720, width=960)
            else:
                fig = px.scatter(x=coords[:,0], y=coords[:,1], hover_name=labels, color=(clusters if clusters is not None else None), height=720, width=960)
            fig.update_traces(marker=dict(size=3))
            fig.write_html(str(out_html), include_plotlyjs='cdn', full_html=True)
        except Exception as e:
            print(f"Interactive HTML export failed: {e}")

    # Also produce the alternate dimensionality (both 2D and 3D by default)
    try:
        other_nc = 3 if int(n_components) == 2 else 2
        coords_alt = reduce_to_nd(
            X,
            method=method,
            n_components=other_nc,
            pca_dim=pca_dim,
            perplexity=perplexity,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            densmap=densmap,
            random_state=seed,
        )
        alt_suffix = "_3d" if other_nc == 3 else "_2d"
        alt_prefix = _Path(str(_Path(out_prefix)) + alt_suffix)
        # CSV for alternate
        out_csv_alt = alt_prefix.with_suffix(".csv")
        with out_csv_alt.open("w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            if other_nc == 3:
                cols = ["label", "x", "y", "z"]
            else:
                cols = ["label", "x", "y"]
            if clusters is not None:
                cols.append("cluster")
            w.writerow(cols)
            for i, lab in enumerate(labels):
                if other_nc == 3:
                    x, y, z = coords_alt[i,0], coords_alt[i,1], coords_alt[i,2]
                    row = [lab, float(x), float(y), float(z)]
                else:
                    x, y = coords_alt[i,0], coords_alt[i,1]
                    row = [lab, float(x), float(y)]
                if clusters is not None:
                    row.append(int(clusters[i]))
                w.writerow(row)
        # PNG only for 2D alt
        if other_nc == 2:
            out_png_alt = alt_prefix.with_suffix(".png")
            plot_map(coords_alt, labels, out_png_alt, clusters=clusters)
        # HTML for alternate
        if interactive_html:
            try:
                import plotly.express as px
                out_html_alt = alt_prefix.with_suffix(".html")
                if other_nc == 3:
                    fig = px.scatter_3d(x=coords_alt[:,0], y=coords_alt[:,1], z=coords_alt[:,2], hover_name=labels, color=(clusters if clusters is not None else None), height=720, width=960)
                else:
                    fig = px.scatter(x=coords_alt[:,0], y=coords_alt[:,1], hover_name=labels, color=(clusters if clusters is not None else None), height=720, width=960)
                fig.update_traces(marker=dict(size=3))
                fig.write_html(str(out_html_alt), include_plotlyjs='cdn', full_html=True)
            except Exception as e:
                print(f"Interactive HTML export (alt) failed: {e}")
    except Exception as e:
        print(f"Alternate dimensionality export failed: {e}")

    return {"status": "ok", "csv": str(out_csv), "png": str(out_png), "books": len(labels)}


@app.function(
    image=image_gpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    gpu=_GPU_H200_1,
    # 10k+ books can take much longer; allow up to 12 hours
    timeout=60 * 60 * 12,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def umap_sweep_remote_gpu(
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    books_dir: str = "/input/training",
    out_csv: str = "/vol/style_maps/umap_sweep/results.csv",
    save_best_prefix: str | None = "/vol/style_maps/umap_sweep/best",
    plotly_html: str | None = None,
    # Data / embedding
    max_books: int | None = 400,
    num_chunks = 'auto',
    chunk_size: int = 14,
    overlap: int = 4,
    max_chars: int | None = None,
    embed_batch_size: int = 256,
    auto_tune_embed_batch: bool = True,
    max_length: int = 384,
    use_projection: bool = False,
    remove_top_pcs: int = 0,
    n_components: int = 2,
    seed: int = 42,
    # UMAP sweep grid
    grid_n_neighbors: str = "15,50,100",
    grid_min_dist: str = "0.01,0.05,0.1",
    densmap: bool = False,
    metric: str = "cosine",
    pca_dim: int = 50,
    # Scoring
    eval_k: str = "5,15",
    score_by: str = "k15",
):
    """Grid search UMAP params on GPU, score layouts, write CSV, and optionally save best layout.

    Returns: dict with results path and best config.
    """
    from pathlib import Path as _Path
    from typing import List as _List, Tuple as _Tuple
    import csv as _csv
    import numpy as _np
    from sklearn.manifold import trustworthiness as _trust
    from sklearn.neighbors import NearestNeighbors as _NN
    from inference_contrastive import ContrastiveBookMatcherInference
    from style_map import (
        _prepare_book_chunks,
        embed_books_batched,
        _remove_top_pcs as __rtp,
        reduce_to_nd,
        plot_map,
    )

    def _parse_list(s: str, cast):
        return [cast(x.strip()) for x in str(s).split(',') if str(x).strip()]

    def _read_text(path: _Path, max_chars: int | None = None) -> str:
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
            if max_chars and len(txt) > max_chars:
                return txt[:max_chars]
            return txt
        except Exception:
            return ""

    def _nhit_rate(X: _np.ndarray, Y: _np.ndarray, k: int = 15) -> float:
        k = max(1, min(k, X.shape[0] - 1))
        nnX = _NN(n_neighbors=k + 1, metric="cosine").fit(X)
        nnY = _NN(n_neighbors=k + 1, metric="euclidean").fit(Y)
        _, idxX = nnX.kneighbors(X)
        _, idxY = nnY.kneighbors(Y)
        idxX = idxX[:, 1:]
        idxY = idxY[:, 1:]
        hits = []
        for i in range(X.shape[0]):
            sX = set(idxX[i])
            sY = set(idxY[i])
            hits.append(len(sX & sY) / float(k))
        return float(_np.mean(hits))

    # Parse lists
    try:
        nc = int(num_chunks)  # type: ignore[arg-type]
    except Exception:
        nc = 'auto'
    grid_n = _parse_list(grid_n_neighbors, int)
    grid_d = _parse_list(grid_min_dist, float)
    k_eval = _parse_list(eval_k, int)

    # Model
    matcher = ContrastiveBookMatcherInference(model_dir)

    # Files
    bdir = _Path(books_dir)
    files = sorted([p for p in bdir.glob("**/*.txt") if p.is_file()])
    if max_books and len(files) > int(max_books):
        files = files[: int(max_books)]
    if not files:
        return {"status": "no-files", "dir": str(bdir)}

    # Chunks
    books = []
    for p in files:
        txt = _read_text(p, max_chars=max_chars)
        chs = _prepare_book_chunks(txt, num_chunks=nc, chunk_size=chunk_size, overlap=overlap)
        books.append((p.stem, chs))

    labels, X = embed_books_batched(
        matcher,
        books,
        max_length=int(max_length),
        embed_batch_size=int(embed_batch_size),
        auto_tune_embed_batch=bool(auto_tune_embed_batch),
        use_projection=use_projection,
        book_pool="trimmed_mean",
        trim_frac=0.2,
        book_topk=5,
    )
    if X.size == 0:
        return {"status": "no-embeddings", "books": 0}
    if remove_top_pcs and int(remove_top_pcs) > 0:
        X = __rtp(X, int(remove_top_pcs))

    # Sweep
    rows: list[dict] = []
    best_score = -1.0
    coords_best = None
    cfg_best = None
    for nn in grid_n:
        for md in grid_d:
            coords = reduce_to_nd(
                X,
                method="umap",
                n_components=int(n_components),
                pca_dim=int(pca_dim),
                perplexity=30.0,
                n_neighbors=int(nn),
                min_dist=float(md),
                metric=metric,
                densmap=bool(densmap),
                random_state=int(seed),
            )
            metrics = {}
            for k in k_eval:
                metrics[f"trust_k{k}"] = float(_trust(X, coords, n_neighbors=int(k)))
                metrics[f"nhit_k{k}"] = float(_nhit_rate(X, coords, k=int(k)))
            if score_by == "avg":
                score = float(_np.mean([metrics[f"trust_k{k}"] for k in k_eval]))
            elif score_by == "k5":
                score = float(metrics.get("trust_k5", 0.0))
            else:
                score = float(metrics.get("trust_k15", 0.0))
            row = {"n_neighbors": int(nn), "min_dist": float(md), "densmap": int(bool(densmap)), **metrics}
            rows.append(row)
            if score > best_score:
                best_score = score
                coords_best = coords
                cfg_best = row

    # Write results
    outp = _Path(out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r.keys()}, key=lambda x: (x not in {"n_neighbors","min_dist","densmap"}, x))
    with outp.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Save best layout
    best_files = {}
    if save_best_prefix and coords_best is not None:
        pref = _Path(save_best_prefix)
        pref.parent.mkdir(parents=True, exist_ok=True)
        best_csv = pref.with_suffix(".csv")
        import csv as _csv2
        with best_csv.open("w", newline="", encoding="utf-8") as f:
            w = _csv2.writer(f)
            if int(n_components) == 3:
                w.writerow(["label","x","y","z"])
                for i, lab in enumerate(labels):
                    x, y, z = coords_best[i,0], coords_best[i,1], coords_best[i,2]
                    w.writerow([lab, float(x), float(y), float(z)])
            else:
                w.writerow(["label","x","y"])
                for i, lab in enumerate(labels):
                    x, y = coords_best[i,0], coords_best[i,1]
                    w.writerow([lab, float(x), float(y)])
        best_files["csv"] = str(best_csv)
        if int(n_components) == 2:
            png = pref.with_suffix(".png")
            plot_map(coords_best, labels, png, clusters=None)
            best_files["png"] = str(png)
        if plotly_html:
            try:
                import plotly.express as px  # type: ignore
                if int(n_components) == 3:
                    fig = px.scatter_3d(x=coords_best[:,0], y=coords_best[:,1], z=coords_best[:,2], hover_name=labels, height=720, width=960)
                else:
                    fig = px.scatter(x=coords_best[:,0], y=coords_best[:,1], hover_name=labels, height=720, width=960)
                fig.update_traces(marker=dict(size=3))
                fig.write_html(plotly_html, include_plotlyjs='cdn', full_html=True)
                best_files["html"] = str(plotly_html)
            except Exception as e:
                print(f"Plotly not available or failed to save HTML: {e}")

    return {"status": "ok", "results": str(outp), "best": cfg_best, **best_files}

@app.function(
    image=image_gpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    gpu=_GPU_H200_1,
    # Allow longer for large corpora
    timeout=60 * 60 * 12,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def style_map_remote_gpu(
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    books_dir: str = "/input/training",
    out_prefix: str = "/vol/style_maps/w7_umap",
    method: str = "umap",
    perplexity: float = 30.0,
    pca_dim: int = 50,
    seed: int = 42,
    max_books: int | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    densmap: bool = False,
    remove_top_pcs: int = 0,
    n_components: int = 2,
    # Robust pooling & projection toggle
    book_pool: str = "trimmed_mean",
    trim_frac: float = 0.2,
    book_topk: int = 5,
    use_projection: bool = False,
    # Optional interactive export and cluster coloring
    interactive_html: bool = True,
    cluster_method: str = "kmeans",
    n_clusters: int = 12,
    hdbscan_min_cluster_size: int = 10,
    hdbscan_min_samples: int | None = None,
    drop_outliers: bool = False,
    prune_top_pct: float = 0.02,
    prune_knn_k: int = 10,
    num_chunks = 'auto',
    chunk_size: int = 14,
    overlap: int = 4,
    max_chars: int | None = None,
    embed_batch_size: int = 256,
    auto_tune_embed_batch: bool = True,
    max_length: int = 384,
):
    """Generate a 2D style map (CSV + PNG) for books under books_dir using the GPU container.

    This variant performs embedding inside the GPU function to leverage CUDA, rather than
    delegating to the CPU function.
    """
    from pathlib import Path as _Path
    from inference_contrastive import ContrastiveBookMatcherInference
    from style_map import _prepare_book_chunks, embed_books_batched, reduce_to_nd, plot_map, cluster_and_prune
    import csv as _csv
    import os as _os

    _ensure_dirs()
    _os.makedirs(_Path(out_prefix).parent, exist_ok=True)

    # Instantiate model inside the GPU container; it will auto-select CUDA if available
    matcher = ContrastiveBookMatcherInference(model_dir)

    # Collect input files
    bdir = _Path(books_dir)
    files = sorted([p for p in bdir.glob("**/*.txt") if p.is_file()])
    if max_books and len(files) > max_books:
        files = files[:max_books]

    # Prepare chunked books
    books = []
    for p in files:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if max_chars and len(txt) > max_chars:
                txt = txt[:max_chars]
        except Exception:
            continue
        chunks = _prepare_book_chunks(txt, num_chunks=num_chunks, chunk_size=chunk_size, overlap=overlap)
        books.append((p.stem, chunks))

    # Embed on GPU and reduce to 2D
    labels, X = embed_books_batched(
        matcher,
        books,
        max_length=int(max_length),
        embed_batch_size=int(embed_batch_size),
        auto_tune_embed_batch=bool(auto_tune_embed_batch),
        use_projection=use_projection,
        book_pool=book_pool,
        trim_frac=trim_frac,
        book_topk=book_topk,
    )
    if X.size == 0:
        return {"status": "no-embeddings", "books": 0}
    # Optional dominant component removal
    if remove_top_pcs and remove_top_pcs > 0:
        from style_map import _remove_top_pcs as __rtp
        X = __rtp(X, int(remove_top_pcs))

    # Optional clustering/pruning before projection
    Xf, clusters, keep_idx = cluster_and_prune(
        X,
        cluster_method=cluster_method,
        drop_outliers=bool(drop_outliers),
        hdbscan_min_cluster_size=int(hdbscan_min_cluster_size),
        hdbscan_min_samples=(None if hdbscan_min_samples is None else int(hdbscan_min_samples)),
        prune_top_pct=float(prune_top_pct),
        prune_knn_k=int(prune_knn_k),
        seed=int(seed),
    )
    if Xf.shape[0] != X.shape[0]:
        labels = [labels[i] for i in keep_idx.tolist()]
        X = Xf

    coords = reduce_to_nd(
        X,
        method=method,
        n_components=n_components,
        pca_dim=pca_dim,
        perplexity=perplexity,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        densmap=densmap,
        random_state=seed,
    )

    # Save CSV/PNG/HTML for primary (n_components)
    out_csv = _Path(out_prefix).with_suffix(".csv")
    # Optional clusters in embedding space
    if clusters is None and cluster_method == "kmeans":
        try:
            from sklearn.cluster import KMeans
            k = max(2, int(n_clusters))
            clusters = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        except Exception as e:
            print(f"Clustering failed: {e}")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        if n_components == 3:
            cols = ["label", "x", "y", "z"]
            if clusters is not None:
                cols.append("cluster")
            w.writerow(cols)
            for i, lab in enumerate(labels):
                x, y, z = coords[i, 0], coords[i, 1], coords[i, 2]
                row = [lab, float(x), float(y), float(z)]
                if clusters is not None:
                    row.append(int(clusters[i]))
                w.writerow(row)
        else:
            cols = ["label", "x", "y"]
            if clusters is not None:
                cols.append("cluster")
            w.writerow(cols)
            for i, lab in enumerate(labels):
                x, y = coords[i, 0], coords[i, 1]
                row = [lab, float(x), float(y)]
                if clusters is not None:
                    row.append(int(clusters[i]))
                w.writerow(row)

    # Save plot
    out_png = _Path(out_prefix).with_suffix(".png")
    if n_components == 2:
        plot_map(coords, labels, out_png, clusters=clusters)
    # Optional interactive HTML (2D or 3D)
    if interactive_html:
        try:
            import plotly.express as px
            out_html = _Path(out_prefix).with_suffix(".html")
            if n_components == 3:
                fig = px.scatter_3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], hover_name=labels, color=(clusters if clusters is not None else None), height=720, width=960)
            else:
                fig = px.scatter(x=coords[:,0], y=coords[:,1], hover_name=labels, color=(clusters if clusters is not None else None), height=720, width=960)
            fig.update_traces(marker=dict(size=3))
            fig.write_html(str(out_html), include_plotlyjs='cdn', full_html=True)
        except Exception as e:
            print(f"Interactive HTML export failed: {e}")

    # Also produce the alternate dimensionality (both 2D and 3D by default)
    try:
        other_nc = 3 if int(n_components) == 2 else 2
        coords_alt = reduce_to_nd(
            X,
            method=method,
            n_components=other_nc,
            pca_dim=pca_dim,
            perplexity=perplexity,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            densmap=densmap,
            random_state=seed,
        )
        alt_suffix = "_3d" if other_nc == 3 else "_2d"
        alt_prefix = _Path(str(_Path(out_prefix)) + alt_suffix)
        # CSV for alternate
        out_csv_alt = alt_prefix.with_suffix(".csv")
        with out_csv_alt.open("w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            if other_nc == 3:
                cols = ["label", "x", "y", "z"]
            else:
                cols = ["label", "x", "y"]
            if clusters is not None:
                cols.append("cluster")
            w.writerow(cols)
            for i, lab in enumerate(labels):
                if other_nc == 3:
                    x, y, z = coords_alt[i,0], coords_alt[i,1], coords_alt[i,2]
                    row = [lab, float(x), float(y), float(z)]
                else:
                    x, y = coords_alt[i,0], coords_alt[i,1]
                    row = [lab, float(x), float(y)]
                if clusters is not None:
                    row.append(int(clusters[i]))
                w.writerow(row)
        # PNG only for 2D alt
        out_png_alt = None
        if other_nc == 2:
            out_png_alt = alt_prefix.with_suffix(".png")
            plot_map(coords_alt, labels, out_png_alt, clusters=clusters)
        # HTML for alternate
        if interactive_html:
            try:
                import plotly.express as px
                out_html_alt = alt_prefix.with_suffix(".html")
                if other_nc == 3:
                    fig = px.scatter_3d(x=coords_alt[:,0], y=coords_alt[:,1], z=coords_alt[:,2], hover_name=labels, color=(clusters if clusters is not None else None), height=720, width=960)
                else:
                    fig = px.scatter(x=coords_alt[:,0], y=coords_alt[:,1], hover_name=labels, color=(clusters if clusters is not None else None), height=720, width=960)
                fig.update_traces(marker=dict(size=3))
                fig.write_html(str(out_html_alt), include_plotlyjs='cdn', full_html=True)
            except Exception as e:
                print(f"Interactive HTML export (alt) failed: {e}")
    except Exception as e:
        print(f"Alternate dimensionality export failed: {e}")

    out = {"status": "ok", "csv": str(out_csv), "png": str(out_png), "books": len(labels)}
    print(out)
    return out


@app.local_entrypoint()
def style_map_gpu(
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    books_dir: str = "/input/training",
    out_prefix: str = "/vol/style_maps/w7_umap",
    method: str = "umap",
    perplexity: float = 30.0,
    pca_dim: int = 50,
    seed: int = 42,
    max_books: int | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    densmap: bool = False,
    remove_top_pcs: int = 0,
    n_components: int = 2,
    book_pool: str = "trimmed_mean",
    trim_frac: float = 0.2,
    book_topk: int = 5,
    use_projection: bool = False,
    interactive_html: bool = True,
    cluster_method: str = "kmeans",
    n_clusters: int = 12,
    hdbscan_min_cluster_size: int = 10,
    hdbscan_min_samples: int | None = None,
    drop_outliers: bool = False,
    prune_top_pct: float = 0.02,
    prune_knn_k: int = 10,
    num_chunks = 'auto',
    chunk_size: int = 14,
    overlap: int = 4,
    max_chars: int | None = None,
    embed_batch_size: int = 256,
    auto_tune_embed_batch: bool = True,
    max_length: int = 384,
):
    return style_map_remote_gpu.remote(
        model_dir=model_dir,
        books_dir=books_dir,
        out_prefix=out_prefix,
        method=method,
        perplexity=perplexity,
        pca_dim=pca_dim,
        seed=seed,
        max_books=max_books,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        densmap=densmap,
        remove_top_pcs=remove_top_pcs,
        n_components=n_components,
        book_pool=book_pool,
        trim_frac=trim_frac,
        book_topk=book_topk,
        use_projection=use_projection,
        interactive_html=interactive_html,
        cluster_method=cluster_method,
        n_clusters=n_clusters,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
        drop_outliers=drop_outliers,
        prune_top_pct=prune_top_pct,
        prune_knn_k=prune_knn_k,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        overlap=overlap,
        max_chars=max_chars,
        embed_batch_size=embed_batch_size,
        auto_tune_embed_batch=auto_tune_embed_batch,
        max_length=max_length,
    )

@app.function(
    image=image_cpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    timeout=60 * 60 * 12,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def calibrate_style_similarity_remote(
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    # Default to the auto-generated path so no flag is needed
    pairs_csv: str = "/vol/data/style_pairs_autogen.csv",
    # Back-compat alias for CLI invocations using --pairs
    pairs: str | None = None,
    method: str = "auto",
    metric: str = "brier",
    n_splits: int = 5,
    group_col: str | None = None,
    save_to: str | None = None,
    num_chunks = 'auto',
    chunk_size: int = 14,
    overlap: int = 4,
    aggregate: str = "mean",
    topk: int = 5,
    max_length: int = 512,
    dataset_dir: str = "/vol/data/processed",
    # Target number of positives and negatives (balanced)
    pairs_per_class: int = 5000,
    # Prefer GPU by default for speed; will dispatch to the GPU variant
    use_gpu: bool = True,
    # Force re-generate pairs CSV even if it exists
    force_generate: bool = False,
):
    """Fit a calibration mapping from cosine -> [0,1] using labeled pairs.

    If use_gpu is True (default), dispatches to calibrate_style_similarity_remote_gpu.
    Supports --method auto to select between logistic/isotonic by CV Brier/ECE.
    """
    # Dispatch to GPU variant by default
    if use_gpu:
        return calibrate_style_similarity_remote_gpu.remote(
            model_dir=model_dir,
            pairs_csv=(pairs or pairs_csv),
            method=method,
            save_to=save_to,
            num_chunks=num_chunks,
            chunk_size=chunk_size,
            overlap=overlap,
            aggregate=aggregate,
            topk=topk,
            max_length=max_length,
            dataset_dir=dataset_dir,
            pairs_per_class=pairs_per_class,
            force_generate=force_generate,
        )

    from calibrate_style_similarity import calibrate_style_similarity as _cal
    import os as _os
    import csv as _csv
    _ensure_dirs()
    # Respect alias if provided
    if pairs:
        pairs_csv = pairs
    # If the pairs CSV is missing, attempt to auto-generate from the HF dataset
    if force_generate or not _os.path.exists(pairs_csv):
        try:
            from datasets import load_from_disk as _load_from_disk
            ds = _load_from_disk(dataset_dir)
            split = 'validation' if 'validation' in ds else 'train'
            table = ds[split]
            # Expect columns: text1, text2, label, and optionally book1, book2, same_topic, neg_type, topic1, topic2
            pos_rows = []  # list of dicts
            neg_same = []  # list of dicts
            neg_diff = []  # list of dicts
            # Helper: compute topic if not present
            def _label_topic(_txt: str) -> int:
                tl = (_txt or '').lower()
                keys = {
                    'religious': ['god','church','prayer','scripture','lord','faith','holy','sacred','divine'],
                    'historical': ['prince','king','castle','knight','lord','duke','empire','queen','court'],
                    'adventure': ['captain','ship','sea','voyage','island','expedition','jungle','desert','treasure'],
                    'romance': ['love','heart','soul','passion','beloved','darling','kiss','romance','marriage'],
                }
                scores = {k: 0 for k in keys}
                for k, ws in keys.items():
                    scores[k] = sum(tl.count(w) for w in ws)
                if scores and max(scores.values()) > 0:
                    best = max(scores.items(), key=lambda kv: kv[1])[0]
                else:
                    best = 'general'
                vocab = ['religious','historical','adventure','romance','general']
                return vocab.index(best)

            for ex in table:
                t1, t2 = ex.get('text1'), ex.get('text2')
                y = int(ex.get('label', ex.get('labels', 0)))
                if not t1 or not t2:
                    continue
                b1 = ex.get('book1') or ''
                b2 = ex.get('book2') or ''
                if y == 1:
                    pos_rows.append({
                        'text1': t1, 'text2': t2, 'label': 1,
                        'book1': b1, 'book2': b2,
                        'same_topic': ex.get('same_topic'),
                        'neg_type': ex.get('neg_type'),
                        'topic1': ex.get('topic1'),
                        'topic2': ex.get('topic2'),
                    })
                else:
                    st = ex.get('same_topic')
                    if st is None:
                        try:
                            st = (_label_topic(t1) == _label_topic(t2))
                        except Exception:
                            st = False
                    neg_entry = {
                        'text1': t1, 'text2': t2, 'label': 0,
                        'book1': b1, 'book2': b2,
                        'same_topic': st,
                        'neg_type': (ex.get('neg_type') or 'unknown'),
                        'topic1': ex.get('topic1'),
                        'topic2': ex.get('topic2'),
                    }
                    if bool(st):
                        neg_same.append(neg_entry)
                    else:
                        neg_diff.append(neg_entry)

            # Balance positives and negatives, and within negatives same/diff topic
            limit_pos = min(int(pairs_per_class), len(pos_rows))
            # Negatives target = limit_pos (balanced)
            target_neg = limit_pos
            half = target_neg // 2
            hard_pref = {"model_mined", "ann_mined", "embed_neighbor", "metadata_similar"}

            def _take(group, k):
                # prefer hard types first
                hard = [r for r in group if (r.get('neg_type') or '') in hard_pref]
                easy = [r for r in group if (r.get('neg_type') or '') not in hard_pref]
                import random as _rnd
                _rnd.shuffle(hard)
                _rnd.shuffle(easy)
                out = hard[:k]
                if len(out) < k:
                    out += easy[: (k - len(out))]
                return out

            chosen_neg = []
            n_same = min(half, len(neg_same))
            n_diff = min(target_neg - n_same, len(neg_diff))
            chosen_neg += _take(neg_same, n_same)
            chosen_neg += _take(neg_diff, n_diff)
            # If still need more, top up from the larger pool
            if len(chosen_neg) < target_neg:
                remaining = target_neg - len(chosen_neg)
                pool = neg_same if len(neg_same) > len(neg_diff) else neg_diff
                chosen_neg += _take(pool, remaining)

            # Final sample sizes
            pos_rows = pos_rows[:limit_pos]
            rows = pos_rows + chosen_neg
            if rows:
                # Shuffle rows to avoid any ordering bias
                import random as _rnd
                _rnd.shuffle(rows)
                auto_path = "/vol/data/style_pairs_autogen.csv"
                _os.makedirs(_os.path.dirname(auto_path), exist_ok=True)
                with open(auto_path, 'w', newline='', encoding='utf-8') as f:
                    w = _csv.writer(f)
                    w.writerow(["text1","text2","label","book1","book2","same_topic","neg_type","topic1","topic2"])            
                    for r in rows:
                        w.writerow([
                            r.get('text1',''), r.get('text2',''), r.get('label',0),
                            r.get('book1',''), r.get('book2',''), r.get('same_topic',''), r.get('neg_type',''),
                            r.get('topic1',''), r.get('topic2','')
                        ])
                print({"info": f"Auto-generated {len(rows)} balanced pairs (pos={limit_pos}, neg={len(rows)-limit_pos}, neg_same={len(chosen_neg) and n_same}, neg_diff={len(chosen_neg) and n_diff}) from {dataset_dir} -> {auto_path}"})
                pairs_csv = auto_path
            else:
                print({"warning": f"Dataset at {dataset_dir} did not contain usable rows; please provide pairs_csv."})
        except Exception as e:
            print({"warning": f"Could not auto-generate pairs from {dataset_dir}: {e}"})
    try:
        nc = int(num_chunks)  # type: ignore[arg-type]
    except Exception:
        nc = 'auto'
    # Final check: ensure pairs_csv exists
    if not _os.path.exists(pairs_csv):
        raise FileNotFoundError(
            f"pairs_csv not found: {pairs_csv}. Provide --pairs or ensure dataset at {dataset_dir} to auto-generate."
        )
    out_path = _cal(
        model_dir=model_dir,
        pairs_csv=pairs_csv,
        method=method,
        metric=metric,
        n_splits=int(n_splits),
        group_col=group_col,
        save_to=save_to,
        num_chunks=nc,
        chunk_size=chunk_size,
        overlap=overlap,
        aggregate=aggregate,
        topk=topk,
        max_length=max_length,
    )
    import os as _os
    _dir = _os.path.dirname(out_path)
    _report = _os.path.join(_dir, 'calibration_report.json')
    _plot = _os.path.join(_dir, 'style_calibration_reliability.png')
    return {"status": "ok", "saved_to": out_path, "report": _report, "plot": _plot, "method": method, "metric": metric, "n_splits": int(n_splits)}

@app.function(
    image=image_gpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    gpu=_GPU_H200_1,
    cpu=8,
    timeout=60 * 60 * 12,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def calibrate_style_similarity_remote_gpu(
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    pairs_csv: str = "/vol/data/style_pairs_autogen.csv",
    # Back-compat alias for CLI invocations using --pairs
    pairs: str | None = None,
    method: str = "auto",
    metric: str = "brier",
    n_splits: int = 5,
    group_col: str | None = None,
    save_to: str | None = None,
    num_chunks = 'auto',
    chunk_size: int = 14,
    overlap: int = 4,
    aggregate: str = "mean",
    topk: int = 5,
    max_length: int = 512,
    dataset_dir: str = "/vol/data/processed",
    # Target number of positives and negatives (balanced)
    pairs_per_class: int = 5000,
    # Force re-generate pairs CSV even if it exists
    force_generate: bool = False,
):
    """GPU variant of style similarity calibration."""
    from calibrate_style_similarity import calibrate_style_similarity as _cal
    import os as _os
    import csv as _csv
    _ensure_dirs()
    # Respect alias if provided
    if pairs:
        pairs_csv = pairs
    # If the pairs CSV is missing, attempt to auto-generate from the HF dataset
    if force_generate or not _os.path.exists(pairs_csv):
        try:
            from datasets import load_from_disk as _load_from_disk
            ds = _load_from_disk(dataset_dir)
            split = 'validation' if 'validation' in ds else 'train'
            table = ds[split]
            # Expect columns: text1, text2, label, and optionally book1, book2, same_topic, neg_type, topic1, topic2
            pos_rows = []  # list of dicts
            neg_same = []  # list of dicts
            neg_diff = []  # list of dicts
            def _label_topic(_txt: str) -> int:
                tl = (_txt or '').lower()
                keys = {
                    'religious': ['god','church','prayer','scripture','lord','faith','holy','sacred','divine'],
                    'historical': ['prince','king','castle','knight','lord','duke','empire','queen','court'],
                    'adventure': ['captain','ship','sea','voyage','island','expedition','jungle','desert','treasure'],
                    'romance': ['love','heart','soul','passion','beloved','darling','kiss','romance','marriage'],
                }
                scores = {k: 0 for k in keys}
                for k, ws in keys.items():
                    scores[k] = sum(tl.count(w) for w in ws)
                if scores and max(scores.values()) > 0:
                    best = max(scores.items(), key=lambda kv: kv[1])[0]
                else:
                    best = 'general'
                vocab = ['religious','historical','adventure','romance','general']
                return vocab.index(best)

            for ex in table:
                t1, t2 = ex.get('text1'), ex.get('text2')
                y = int(ex.get('label', ex.get('labels', 0)))
                if not t1 or not t2:
                    continue
                b1 = ex.get('book1') or ''
                b2 = ex.get('book2') or ''
                if y == 1:
                    pos_rows.append({
                        'text1': t1, 'text2': t2, 'label': 1,
                        'book1': b1, 'book2': b2,
                        'same_topic': ex.get('same_topic'),
                        'neg_type': ex.get('neg_type'),
                        'topic1': ex.get('topic1'),
                        'topic2': ex.get('topic2'),
                    })
                else:
                    st = ex.get('same_topic')
                    if st is None:
                        try:
                            st = (_label_topic(t1) == _label_topic(t2))
                        except Exception:
                            st = False
                    neg_entry = {
                        'text1': t1, 'text2': t2, 'label': 0,
                        'book1': b1, 'book2': b2,
                        'same_topic': st,
                        'neg_type': (ex.get('neg_type') or 'unknown'),
                        'topic1': ex.get('topic1'),
                        'topic2': ex.get('topic2'),
                    }
                    if bool(st):
                        neg_same.append(neg_entry)
                    else:
                        neg_diff.append(neg_entry)

            limit_pos = min(int(pairs_per_class), len(pos_rows))
            target_neg = limit_pos
            half = target_neg // 2
            hard_pref = {"model_mined", "ann_mined", "embed_neighbor", "metadata_similar"}

            def _take(group, k):
                hard = [r for r in group if (r.get('neg_type') or '') in hard_pref]
                easy = [r for r in group if (r.get('neg_type') or '') not in hard_pref]
                import random as _rnd
                _rnd.shuffle(hard)
                _rnd.shuffle(easy)
                out = hard[:k]
                if len(out) < k:
                    out += easy[: (k - len(out))]
                return out

            chosen_neg = []
            n_same = min(half, len(neg_same))
            n_diff = min(target_neg - n_same, len(neg_diff))
            chosen_neg += _take(neg_same, n_same)
            chosen_neg += _take(neg_diff, n_diff)
            if len(chosen_neg) < target_neg:
                remaining = target_neg - len(chosen_neg)
                pool = neg_same if len(neg_same) > len(neg_diff) else neg_diff
                chosen_neg += _take(pool, remaining)

            pos_rows = pos_rows[:limit_pos]
            rows = pos_rows + chosen_neg
            if rows:
                import random as _rnd
                _rnd.shuffle(rows)
                auto_path = "/vol/data/style_pairs_autogen.csv"
                _os.makedirs(_os.path.dirname(auto_path), exist_ok=True)
                with open(auto_path, 'w', newline='', encoding='utf-8') as f:
                    w = _csv.writer(f)
                    w.writerow(["text1","text2","label","book1","book2","same_topic","neg_type","topic1","topic2"])            
                    for r in rows:
                        w.writerow([
                            r.get('text1',''), r.get('text2',''), r.get('label',0),
                            r.get('book1',''), r.get('book2',''), r.get('same_topic',''), r.get('neg_type',''),
                            r.get('topic1',''), r.get('topic2','')
                        ])
                print({"info": f"Auto-generated {len(rows)} balanced pairs (pos={limit_pos}, neg={len(rows)-limit_pos}, neg_same={len(chosen_neg) and n_same}, neg_diff={len(chosen_neg) and n_diff}) from {dataset_dir} -> {auto_path}"})
                pairs_csv = auto_path
            else:
                print({"warning": f"Dataset at {dataset_dir} did not contain usable rows; please provide pairs_csv."})
        except Exception as e:
            print({"warning": f"Could not auto-generate pairs from {dataset_dir}: {e}"})
    try:
        nc = int(num_chunks)  # type: ignore[arg-type]
    except Exception:
        nc = 'auto'
    # Final check: ensure pairs_csv exists
    if not _os.path.exists(pairs_csv):
        raise FileNotFoundError(
            f"pairs_csv not found: {pairs_csv}. Provide --pairs or ensure dataset at {dataset_dir} to auto-generate."
        )
    out_path = _cal(
        model_dir=model_dir,
        pairs_csv=pairs_csv,
        method=method,
        metric=metric,
        n_splits=int(n_splits),
        group_col=group_col,
        save_to=save_to,
        num_chunks=nc,
        chunk_size=chunk_size,
        overlap=overlap,
        aggregate=aggregate,
        topk=topk,
        max_length=max_length,
    )
    import os as _os
    _dir = _os.path.dirname(out_path)
    _report = _os.path.join(_dir, 'calibration_report.json')
    _plot = _os.path.join(_dir, 'style_calibration_reliability.png')
    return {"status": "ok (gpu)", "saved_to": out_path, "report": _report, "plot": _plot, "method": method, "metric": metric, "n_splits": int(n_splits)}


# Local shortcuts (no flags needed)
@app.local_entrypoint()
def calibrate_style():
    return calibrate_style_similarity_remote.remote(
        pairs_per_class=5000,
        use_gpu=True,
        force_generate=True,
    )


@app.local_entrypoint()
def calibrate_style_gpu():
    return calibrate_style_similarity_remote_gpu.remote(
        pairs_per_class=5000,
        force_generate=True,
    )


@app.function(
    image=image_gpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[modal.Secret.from_name("llm-api-keys")],
    gpu=_GPU_H200_1,
    timeout=60 * 60 * 12,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def run_style_benchmark(
    model: str,
    book: str,
    n_samples: int = 1,
    n_excerpts: int = 5,
    concurrency: int = 3,
    # Topic control
    topic: str | None = None,
    topics_file: str | None = None,
    # Scoring params
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    num_chunks = 'auto',
    chunk_size: int = 14,
    overlap: int = 4,
    aggregate: str = "mean",
    topk: int = 5,
    max_length: int = 512,
    # LLM params
    temperature: float | None = None,
    top_p: float = 0.95,
    max_tokens: int = 1200,
    seed: int = 42,
    # Logging
    log_outputs: bool = True,
):
    """Run the LLM style benchmark:
    - Select a 15-sentence excerpt from the chosen book
    - Generate n_samples stories on a topic with the given model
    - Score style similarity using the contrastive matcher (GPU)
    """
    import os as _os
    import json as _json
    from eval.benchmark_style import run_benchmark as _run

    # Resolve book path (abs, or relative to workspace/eval/books)
    book_path = book
    if not _os.path.isabs(book_path):
        cand1 = _os.path.join("/workspace", book_path)
        cand2 = _os.path.join("/workspace/eval/books", book_path)
        for c in (cand1, cand2):
            if _os.path.exists(c):
                book_path = c
                break
    if not _os.path.exists(book_path):
        raise FileNotFoundError(f"Book not found: {book}")

    # Load topics if provided
    topics = None
    if topics_file:
        p = topics_file
        if not _os.path.isabs(p):
            p2 = _os.path.join("/workspace", p)
            if _os.path.exists(p2):
                p = p2
        if not _os.path.exists(p):
            raise FileNotFoundError(f"topics_file not found: {topics_file}")
        try:
            # Try JSON first
            with open(p, 'r', encoding='utf-8') as f:
                data = f.read().strip()
            topics = _json.loads(data)
            if not isinstance(topics, list):
                topics = None
        except Exception:
            # Fallback: newline-separated
            with open(p, 'r', encoding='utf-8') as f:
                topics = [ln.strip() for ln in f if ln.strip()]

    # Determine provider-specific defaults
    _provider = None
    try:
        _provider = (model.split(":", 1)[0] or "").lower()
    except Exception:
        _provider = None
    if temperature is None:
        effective_temperature = 1.0 if _provider == "kimi" else 0.9
    else:
        effective_temperature = float(temperature)

    # Provider-specific concurrency: Kimi tends to rate-limit aggressively.
    # If caller didn't override from the default (30) or asked for higher, cap to 8.
    try:
        effective_concurrency = int(concurrency)
    except Exception:
        effective_concurrency = 3
    if _provider == "kimi":
        if effective_concurrency >= 8:
            effective_concurrency = 8
    # Cap concurrency a bit for GPT-5 to reduce flaky/empty responses
    try:
        if _provider == "openai" and ("gpt-5" in (model or "").lower()):
            if effective_concurrency > 10:
                effective_concurrency = 10
    except Exception:
        pass
    # Final global cap to 3 to reduce flakiness across providers
    if effective_concurrency > 3:
        effective_concurrency = 3
    # Sanity lower bound
    if effective_concurrency < 1:
        effective_concurrency = 1

    print({
        "info": "Running style benchmark",
        "model": model,
        "book": book_path,
        "n_samples": int(n_samples),
        "n_excerpts": int(n_excerpts),
        "concurrency": int(effective_concurrency),
        "temperature": effective_temperature,
        "topic": topic,
        "topics_loaded": (len(topics) if topics else 0),
    })

    streamed = bool(log_outputs)
    results = _run(
        model=model,
        book_path=book_path,
        topics=topics,
        fixed_topic=topic,
        n_samples=int(n_samples),
        n_excerpts=int(n_excerpts),
        concurrency=int(effective_concurrency),
        seed=int(seed),
        model_dir=model_dir,
        num_chunks=num_chunks,
        chunk_size=int(chunk_size),
        overlap=int(overlap),
        aggregate=aggregate,
        topk=int(topk),
        max_length=int(max_length),
        temperature=float(effective_temperature),
        top_p=float(top_p),
        max_tokens=int(max_tokens),
        stream_print=streamed,
    )
    # Optionally log raw LLM outputs exactly as returned
    if log_outputs and not streamed:
        try:
            samples = results.get("samples", [])
            total = len(samples)
            for i, s in enumerate(samples, start=1):
                out = s.get("output", "")
                topic_i = s.get("topic")
                print(f"=== LLM OUTPUT {i}/{total} (topic={topic_i}) ===")
                if isinstance(out, str):
                    print(out)
                else:
                    print(str(out))
                print(f"=== END LLM OUTPUT {i}/{total} ===")
        except Exception:
            pass
    # Print a compact summary
    summary = {
        "model": results.get("model"),
        "book": results.get("book"),
        "aggregate": results.get("aggregate"),
    }
    print({"summary": summary})
    return results


@app.function(
    image=image_cpu,
    volumes={"/vol": artifacts_vol},
    secrets=[modal.Secret.from_name("llm-api-keys")],
    timeout=60 * 5,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def debug_openai_env(model: str = "gpt-4o-mini"):
    """Small diagnostic: checks OpenAI env in the container and attempts a 1-token completion.

    Prints masked key suffix, base_url, org/project, and the result or error of a tiny request.
    """
    import os as _os
    try:
        key = _os.getenv("OPENAI_API_KEY")
        base = _os.getenv("OPENAI_BASE_URL")
        org = _os.getenv("OPENAI_ORG_ID") or _os.getenv("OPENAI_ORGANIZATION")
        proj = _os.getenv("OPENAI_PROJECT")
        print({
            "OPENAI_API_KEY_present": bool(key),
            "OPENAI_API_KEY_suffix": (key[-6:] if key else None),
            "OPENAI_BASE_URL": base or None,
            "OPENAI_ORG_ID": org or None,
            "OPENAI_PROJECT": proj or None,
        })
        if not key:
            return {"ok": False, "error": "OPENAI_API_KEY not set in secret"}
        from openai import OpenAI
        kwargs = {"api_key": key}
        if base:
            kwargs["base_url"] = base
        if org:
            kwargs["organization"] = org
        if proj:
            kwargs["project"] = proj
        client = OpenAI(**kwargs)
        try:
            _is_gpt5 = (model or "").lower().startswith("gpt-5") or "gpt-5" in (model or "").lower()
            if _is_gpt5:
                # Use Responses API for GPT-5. Omit temperature/top_p.
                _input = [
                    {"role": "user", "content": [{"type": "input_text", "text": "Say 'ok'"}]}
                ]
                resp = client.responses.create(
                    model=model,
                    input=_input,
                    max_output_tokens=5,
                )
                txt = (getattr(resp, "output_text", None) or "").strip()
                if not txt:
                    # Fallback to Chat Completions if needed
                    try:
                        cc = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": "Say 'ok'"}],
                            max_completion_tokens=5,
                        )
                        txt = (cc.choices[0].message.content or "").strip()
                    except Exception:
                        pass
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Say 'ok'"}],
                    max_tokens=5,
                    temperature=0.0,
                )
                txt = (resp.choices[0].message.content or "").strip()
            print({"test_call": "ok", "model": model, "output": txt})
            return {"ok": True, "output": txt}
        except Exception as e:
            print({"test_call": "error", "model": model, "error": str(e)})
            return {"ok": False, "error": str(e)}
    except Exception as e:
        print({"diagnostic_error": str(e)})
        return {"ok": False, "error": str(e)}


@app.function(
    image=image_cpu,
    volumes={"/vol": artifacts_vol},
    secrets=[modal.Secret.from_name("llm-api-keys")],
    timeout=60 * 5,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def debug_kimi_env(model: str = "moonshot-v1-8k", base_url: str | None = None):
    """Diagnostic for Kimi/Moonshot env and a tiny chat call.

    Prints masked key suffix, effective base_url, and attempts a 1-token completion.
    """
    import os as _os
    try:
        key = _os.getenv("KIMI_API_KEY") or _os.getenv("MOONSHOT_API_KEY")
        base = base_url or _os.getenv("KIMI_BASE_URL") or "https://api.moonshot.cn/v1"
        print({
            "KIMI_API_KEY_present": bool(key),
            "KIMI_API_KEY_suffix": (key[-6:] if key else None),
            "KIMI_BASE_URL": base,
            "model": model,
        })
        if not key:
            return {"ok": False, "error": "KIMI_API_KEY or MOONSHOT_API_KEY not set in secret"}
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=base)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say 'ok'"}],
                max_tokens=5,
                temperature=0.0,
            )
            txt = (resp.choices[0].message.content or "").strip()
            print({"test_call": "ok", "output": txt})
            return {"ok": True, "output": txt}
        except Exception as e:
            print({"test_call": "error", "error": str(e)})
            return {"ok": False, "error": str(e)}
    except Exception as e:
        print({"diagnostic_error": str(e)})
        return {"ok": False, "error": str(e)}

# Local entrypoints for `modal run`
@app.local_entrypoint()
def prepare(
    training_dir: str = "/input/training",
    chunk_size: int = 14,
    overlap: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_chunks_per_book: int = 800,
    use_hard_negatives: bool = True,
    # Embedding-based hard negatives
    use_embedding_hard_negatives: bool = True,
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    num_chunks_for_embed: int = 80,
    num_hard_negative_books: int = 50,
    n_positive_per_book: int = 20,
    n_negative_per_book: int = 40,
):
    return prepare_remote_gpu.remote(
        training_dir=training_dir,
        chunk_size=chunk_size,
        overlap=overlap,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        max_chunks_per_book=max_chunks_per_book,
        use_hard_negatives=use_hard_negatives,
        use_embedding_hard_negatives=use_embedding_hard_negatives,
        embedding_model=embedding_model,
        num_chunks_for_embed=num_chunks_for_embed,
        num_hard_negative_books=num_hard_negative_books,
        n_positive_per_book=n_positive_per_book,
        n_negative_per_book=n_negative_per_book,
    )


@app.local_entrypoint()
def prepare_gpu(
    training_dir: str = "/input/training",
    chunk_size: int = 14,
    overlap: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_chunks_per_book: int = 800,
    use_hard_negatives: bool = True,
    use_embedding_hard_negatives: bool = True,
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    num_chunks_for_embed: int = 80,
    num_hard_negative_books: int = 50,
    n_positive_per_book: int = 20,
    n_negative_per_book: int = 40,
):
    return prepare_remote_gpu.remote(
        training_dir=training_dir,
        chunk_size=chunk_size,
        overlap=overlap,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        max_chunks_per_book=max_chunks_per_book,
        use_hard_negatives=use_hard_negatives,
        use_embedding_hard_negatives=use_embedding_hard_negatives,
        embedding_model=embedding_model,
        num_chunks_for_embed=num_chunks_for_embed,
        num_hard_negative_books=num_hard_negative_books,
        n_positive_per_book=n_positive_per_book,
        n_negative_per_book=n_negative_per_book,
    )


@app.local_entrypoint()
def benchmark_style(
    model: str,
    book: str,
    n_samples: int = 1,
    n_excerpts: int = 5,
    concurrency: int = 3,
    topic: str | None = None,
    topics_file: str | None = None,
    # Scoring params
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    num_chunks = 'auto',
    chunk_size: int = 14,
    overlap: int = 4,
    aggregate: str = "mean",
    topk: int = 5,
    max_length: int = 512,
    # LLM params
    temperature: float | None = None,
    top_p: float = 0.95,
    max_tokens: int = 1200,
    seed: int = 42,
    log_outputs: bool = True,
):
    return run_style_benchmark.remote(
        model=model,
        book=book,
        n_samples=n_samples,
        n_excerpts=n_excerpts,
        concurrency=concurrency,
        topic=topic,
        topics_file=topics_file,
        model_dir=model_dir,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        overlap=overlap,
        aggregate=aggregate,
        topk=topk,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        log_outputs=log_outputs,
    )


@app.local_entrypoint()
def debug_openai(model: str = "gpt-4o-mini"):
    return debug_openai_env.remote(model=model)


@app.local_entrypoint()
def debug_kimi(model: str = "moonshot-v1-8k", base_url: str | None = None):
    return debug_kimi_env.remote(model=model, base_url=base_url)

@app.local_entrypoint()
def train(
    model: str = "roberta-base",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    warmup_steps: int = 500,
    data_subdir: str = "/vol/data/processed",
    output_subdir: str = "/vol/models/book_matcher",
):
    return train_remote.remote(
        model_name=model,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        data_subdir=data_subdir,
        output_subdir=output_subdir,
    )


@app.local_entrypoint()
def train_gpu(
    model: str = "roberta-base",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    warmup_steps: int = 500,
    data_subdir: str = "/vol/data/processed",
    output_subdir: str = "/vol/models/book_matcher",
):
    return train_remote_gpu.remote(
        model_name=model,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        data_subdir=data_subdir,
        output_subdir=output_subdir,
    )


@app.local_entrypoint()
def train_contrastive(
    model: str = "roberta-large",
    epochs: int = 6,
    batch_size: int = 16,
    lr: float = 1e-5,
    warmup_steps: int = 1000,
    use_style_features: bool = True,
    use_symmetric_features: bool = True,
    data_subdir: str = "/vol/data/processed",
    output_subdir: str = "/vol/models/book_matcher_contrastive",
    contrastive_weight: float = 0.2,
    distill_from_cross: bool = True,
    teacher_model_dir: str = "/vol/models/book_matcher/final",
    distill_weight: float = 0.7,
    distill_temperature: float = 3.0,
    calibrate_for: str = "f1",
    target_acc: float | None = 0.85,
    target_recall: float | None = 0.80,
    pooling: str = "attn",
    use_projection: bool = True,
    label_smoothing: float = 0.05,
    grad_accum_steps: int = 2,
    select_metric: str = "pr_auc",
    classifier: str = "arcface",
    arcface_margin: float = 0.2,
    arcface_scale: float = 30.0,
    contrastive_mode: str = "supcon",
    max_length: int = 256,
    grad_checkpointing: bool = True,
    teacher_on_gpu: bool = False,
    tokenize_workers: int = 4,
):
    return train_contrastive_remote.remote(
        model_name=model,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        use_style_features=use_style_features,
        use_symmetric_features=use_symmetric_features,
        data_subdir=data_subdir,
        output_subdir=output_subdir,
        contrastive_weight=contrastive_weight,
        distill_from_cross=distill_from_cross,
        teacher_model_dir=teacher_model_dir,
        distill_weight=distill_weight,
        distill_temperature=distill_temperature,
        calibrate_for=calibrate_for,
        target_acc=target_acc,
        target_recall=target_recall,
        pooling=pooling,
        use_projection=use_projection,
        label_smoothing=label_smoothing,
        grad_accum_steps=grad_accum_steps,
        select_metric=select_metric,
        classifier=classifier,
        arcface_margin=arcface_margin,
        arcface_scale=arcface_scale,
        contrastive_mode=contrastive_mode,
        max_length=max_length,
        grad_checkpointing=grad_checkpointing,
        teacher_on_gpu=teacher_on_gpu,
        tokenize_workers=tokenize_workers,
    )


@app.local_entrypoint()
def train_contrastive_gpu(
    model: str = "roberta-large",
    epochs: int = 6,
    batch_size: int = 8,
    lr: float = 1e-5,
    warmup_steps: int = 1000,
    use_style_features: bool = True,
    use_symmetric_features: bool = True,
    data_subdir: str = "/vol/data/processed",
    output_subdir: str = "/vol/models/book_matcher_contrastive",
    contrastive_weight: float = 0.3,
    distill_from_cross: bool = True,
    teacher_model_dir: str = "/vol/models/book_matcher/final",
    distill_weight: float = 0.5,
    distill_temperature: float = 3.0,
    calibrate_for: str = "accuracy",
    target_acc: float | None = None,
    target_recall: float | None = 0.85,
    pooling: str = "attn",
    use_projection: bool = True,
    label_smoothing: float = 0.03,
    grad_accum_steps: int = 2,
    select_metric: str = "balanced_accuracy",
    classifier: str = "arcface",
    arcface_margin: float = 0.25,
    arcface_scale: float = 30.0,
    contrastive_mode: str = "supcon",
    max_length: int = 512,
    grad_checkpointing: bool = True,
    teacher_on_gpu: bool = True,
    tokenize_workers: int = 4,
):
    return train_contrastive_remote_gpu.remote(
        model_name=model,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        use_style_features=use_style_features,
        use_symmetric_features=use_symmetric_features,
        data_subdir=data_subdir,
        output_subdir=output_subdir,
        contrastive_weight=contrastive_weight,
        distill_from_cross=distill_from_cross,
        teacher_model_dir=teacher_model_dir,
        distill_weight=distill_weight,
        distill_temperature=distill_temperature,
        calibrate_for=calibrate_for,
        target_acc=target_acc,
        target_recall=target_recall,
        pooling=pooling,
        use_projection=use_projection,
        label_smoothing=label_smoothing,
        grad_accum_steps=grad_accum_steps,
        select_metric=select_metric,
        classifier=classifier,
        arcface_margin=arcface_margin,
        arcface_scale=arcface_scale,
        contrastive_mode=contrastive_mode,
        max_length=max_length,
        grad_checkpointing=grad_checkpointing,
        teacher_on_gpu=teacher_on_gpu,
        tokenize_workers=tokenize_workers,
    )


@app.local_entrypoint()
def train_contrastive_eight_gpu_h200(
    model: str = "roberta-large",
    num_epochs: int = 6,
    batch_size: int = 64,
    lr: float = 2e-5,
    warmup_steps: int = 1000,
    use_style_features: bool = True,
    use_symmetric_features: bool = True,
    data_subdir: str = "/vol/data/processed",
    output_subdir: str = "/vol/models/book_matcher_contrastive",
    # Optional prepare
    prepare_before_train: bool = True,
    prepare_training_dir: str = "/input/training",
    # Tokenization (None -> auto)
    tokenize_workers: int | None = None,
):
    """Local CLI wrapper for 8x H200 training.

    Example:
    modal run modal_app.py::train_contrastive_eight_gpu_h200 -- --prepare-before-train=false --epochs 6 --batch-size 64
    """
    return train_contrastive_remote_eight_gpu_h200.remote(
        model_name=model,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        use_style_features=use_style_features,
        use_symmetric_features=use_symmetric_features,
        data_subdir=data_subdir,
        output_subdir=output_subdir,
        prepare_before_train=prepare_before_train,
        prepare_training_dir=prepare_training_dir,
        tokenize_workers=tokenize_workers,
    )

@app.local_entrypoint()
def pipeline(
    model: str = "roberta-base",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    warmup_steps: int = 500,
):
    prepare_remote_gpu.remote(training_dir="/input/training")
    # Default to GPU training for speed
    train_remote_gpu.remote(
        model_name=model,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup_steps,
    )


# Contrastive one-shot pipeline: prepare + contrastive train (GPU)
@app.local_entrypoint()
def pipeline_contrastive(
    # Prepare args (defaults tuned)
    training_dir: str = "/input/training",
    # Sharding controls: off by default for reliability
    force_sharded: bool = False,
    auto_shard: bool = False,
    sharded_switch_threshold: int = 50000,
    containers: int = 100,
    per_container_workers: int = 8,
    chunk_size: int = 14,
    overlap: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_chunks_per_book: int = 800,
    use_hard_negatives: bool = True,
    use_embedding_hard_negatives: bool = True,
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    num_chunks_for_embed: int = 80,
    num_hard_negative_books: int = 50,
    n_positive_per_book: int = 20,
    n_negative_per_book: int = 40,
    # Train args (defaults tuned)
    model: str = "roberta-large",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-5,
    warmup_steps: int = 1000,
    use_style_features: bool = True,
    data_subdir: str = "/vol/data/processed",
    output_subdir: str = "/vol/models/book_matcher_contrastive",
    contrastive_weight: float = 0.2,
    max_length: int = 512,
):
    # Prepare datasets only if missing
    import os as _os
    _ann_dir = "/vol/models/book_matcher_contrastive/final"
    _ann_ok = _os.path.exists(f"{_ann_dir}/pytorch_model.bin") or _os.path.exists(f"{_ann_dir}/model.safetensors")
    _ds_dir = data_subdir
    _ds_ready = _os.path.exists(_os.path.join(_ds_dir, 'train')) and _os.path.exists(_os.path.join(_ds_dir, 'validation'))

    if not _ds_ready:
        # Decide sharded vs single-container prepare
        # Count only if auto_shard is requested; otherwise skip the RPC for speed
        n_books = 0
        if auto_shard:
            try:
                stat = count_training_texts_remote.remote(training_dir=training_dir)
                n_books = int(stat.get("count", 0))
            except Exception:
                n_books = 0
        use_sharded = bool(force_sharded or (auto_shard and (n_books >= int(sharded_switch_threshold))))

        if use_sharded:
            print({"prepare_mode": "sharded", "books": n_books, "threshold": int(sharded_switch_threshold), "force": bool(force_sharded), "auto": bool(auto_shard)})
            prepare_sharded_remote.remote(
                training_dir=training_dir,
                containers=int(containers),
                per_container_workers=int(per_container_workers),
                chunk_size=chunk_size,
                overlap=overlap,
                max_chunks_per_book=max_chunks_per_book,
                use_hard_negatives=use_hard_negatives,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                use_embedding_hard_negatives=use_embedding_hard_negatives,
                embedding_model=embedding_model,
                num_chunks_for_embed=num_chunks_for_embed,
                num_hard_negative_books=num_hard_negative_books,
                n_positive_per_book=n_positive_per_book,
                n_negative_per_book=n_negative_per_book,
            )
        else:
            print({"prepare_mode": "single", "books": n_books, "auto": bool(auto_shard)})
            try:
                prepare_remote_gpu.remote(
                    training_dir=training_dir,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    max_chunks_per_book=max_chunks_per_book,
                    use_hard_negatives=use_hard_negatives,
                    use_embedding_hard_negatives=use_embedding_hard_negatives,
                    embedding_model=embedding_model,
                    num_chunks_for_embed=num_chunks_for_embed,
                    num_hard_negative_books=num_hard_negative_books,
                    n_positive_per_book=n_positive_per_book,
                    n_negative_per_book=n_negative_per_book,
                    # Only use miners if a prior contrastive checkpoint exists
                    use_model_mined_negatives=_ann_ok,
                    use_ann_chunk_negatives=_ann_ok,
                    ann_miner_model_dir=(_ann_dir if _ann_ok else None),
                    miner_model_dir=(_ann_dir if _ann_ok else None),
                )
            except Exception as e:
                print({"prepare_retry": str(e)})
                prepare_remote_gpu.remote(
                    training_dir=training_dir,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    max_chunks_per_book=max_chunks_per_book,
                    use_hard_negatives=use_hard_negatives,
                    use_embedding_hard_negatives=use_embedding_hard_negatives,
                    embedding_model=embedding_model,
                    num_chunks_for_embed=num_chunks_for_embed,
                    num_hard_negative_books=num_hard_negative_books,
                    n_positive_per_book=n_positive_per_book,
                    n_negative_per_book=n_negative_per_book,
                    use_model_mined_negatives=False,
                    use_ann_chunk_negatives=False,
                    ann_miner_model_dir=None,
                    miner_model_dir=None,
                )
    else:
        print({"prepare": "skipped", "datasets_ready": True, "data_subdir": _ds_dir})
    # Train contrastive on GPU
    train_contrastive_remote_gpu.remote(
        model_name=model,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        use_style_features=use_style_features,
        data_subdir=data_subdir,
        output_subdir=output_subdir,
        contrastive_weight=contrastive_weight,
        max_length=max_length,
        grad_checkpointing=True,
        # Avoid double prepare in pipeline
        prepare_before_train=False,
    )
    # Calibrate contrastive classifier (temperature + threshold) on GPU
    calibrate_contrastive_remote_gpu.remote(
        model_dir=f"{output_subdir}/final",
        data_subdir=data_subdir,
        calibrate_for="accuracy",
        target_recall=0.85,
        batch_size=512,
        num_proc=4,
        num_workers=4,
        max_length=max_length,
    )
    # Calibrate style similarity on GPU (auto-generates pairs if needed)
    calibrate_style_similarity_remote_gpu.remote(
        model_dir=f"{output_subdir}/final",
        pairs_csv="/vol/data/style_pairs_autogen.csv",
        method="auto",
        metric="brier",
        n_splits=5,
        dataset_dir=data_subdir,
        pairs_per_class=5000,
        force_generate=True,
    )
    # Cross-encoder training is triggered by train_contrastive_remote_gpu by default.
    return {"status": "pipeline_contrastive completed: prepared + trained contrastive + calibrated classifier + calibrated style"}


# One-click: prepare -> train contrastive (GPU) -> calibrate style similarity (GPU)
@app.local_entrypoint()
def pipeline_style(
    # Prepare defaults
    training_dir: str = "/input/training",
    # Train defaults
    model: str = "roberta-large",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-5,
    warmup_steps: int = 1000,
    # Scoring/calibration defaults (kept implicit)
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    dataset_dir: str = "/vol/data/processed",
):
    """Runs the full default pipeline with no extra flags required.

    Steps: prepare (defaults) -> train_contrastive_gpu (defaults) -> calibrate_style_similarity (auto).
    Artifacts: datasets at /vol/data/processed, model at /vol/models/book_matcher_contrastive/final,
    calibration at ../style_calibration.json with reliability report and plot.
    """
    # 1) Prepare
    prepare_remote_gpu.remote(training_dir=training_dir)
    # 2) Train contrastive on GPU (defaults tuned)
    train_contrastive_remote_gpu.remote(
        model_name=model,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        data_subdir=dataset_dir,
        output_subdir="/vol/models/book_matcher_contrastive",
        # Avoid double prepare in pipeline
        prepare_before_train=False,
    )
    # 3) Calibrate contrastive classifier (GPU)
    calibrate_contrastive_remote_gpu.remote(
        model_dir=model_dir,
        data_subdir=dataset_dir,
        calibrate_for="accuracy",
        target_recall=0.85,
        batch_size=512,
        num_proc=4,
        num_workers=4,
    )
    # 4) Calibrate style similarity (auto-select, GPU)
    calibrate_style_similarity_remote_gpu.remote(
        model_dir=model_dir,
        pairs_csv="/vol/data/style_pairs_autogen.csv",
        method="auto",
        metric="brier",
        n_splits=5,
        group_col=None,
        dataset_dir=dataset_dir,
        pairs_per_class=5000,
        force_generate=True,
    )
    return {"status": "pipeline_style completed: prepared + trained + calibrated classifier + calibrated style"}
# Inference (contrastive)
@app.function(
    image=image_cpu,
    volumes={"/vol": artifacts_vol},
    secrets=[],
    timeout=60 * 10,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def infer_contrastive_remote(
    text1: str,
    text2: str,
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    threshold: float | None = None,
):
    from inference_contrastive import ContrastiveBookMatcherInference

    matcher = ContrastiveBookMatcherInference(model_dir, threshold=threshold)
    result = matcher.predict(text1, text2)
    print({"result": result})
    return result


@app.local_entrypoint()
def infer_contrastive(
    text1: str,
    text2: str,
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    threshold: float | None = None,
):
    return infer_contrastive_remote.remote(text1=text1, text2=text2, model_dir=model_dir, threshold=threshold)


# Inference (two-stage: contrastive prefilter + cross-encoder rerank)
@app.function(
    image=image_cpu,
    volumes={"/vol": artifacts_vol},
    secrets=[],
    timeout=60 * 10,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def infer_two_stage_remote(
    text1: str,
    text2: str,
    bi_model_dir: str = "/vol/models/book_matcher_contrastive/final",
    cross_model_dir: str = "/vol/models/book_matcher/final",
    prefilter_threshold: float | None = None,
    cross_threshold: float = 0.5,
):
    from inference_two_stage import TwoStageBookMatcher

    matcher = TwoStageBookMatcher(
        bi_model_dir=bi_model_dir,
        cross_model_dir=cross_model_dir,
        prefilter_threshold=prefilter_threshold,
        cross_threshold=cross_threshold,
    )
    result = matcher.predict(text1, text2)
    print({"result": result})
    return result


@app.local_entrypoint()
def infer_two_stage(
    text1: str,
    text2: str,
    bi_model_dir: str = "/vol/models/book_matcher_contrastive/final",
    cross_model_dir: str = "/vol/models/book_matcher/final",
    prefilter_threshold: float | None = None,
    cross_threshold: float = 0.5,
):
    return infer_two_stage_remote.remote(
        text1=text1,
        text2=text2,
        bi_model_dir=bi_model_dir,
        cross_model_dir=cross_model_dir,
        prefilter_threshold=prefilter_threshold,
        cross_threshold=cross_threshold,
    )


@app.local_entrypoint()
def calibrate_contrastive(
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    data_subdir: str = "/vol/data/processed",
    calibrate_for: str = "accuracy",
    target_acc: float | None = None,
    target_recall: float | None = 0.85,
    save_to: str | None = None,
    max_length: int = 512,
):
    return calibrate_contrastive_remote.remote(
        model_dir=model_dir,
        data_subdir=data_subdir,
        calibrate_for=calibrate_for,
        target_acc=target_acc,
        target_recall=target_recall,
        save_to=save_to,
        max_length=max_length,
    )


@app.local_entrypoint()
def calibrate_contrastive_gpu(
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    data_subdir: str = "/vol/data/processed",
    calibrate_for: str = "accuracy",
    target_acc: float | None = None,
    target_recall: float | None = 0.85,
    save_to: str | None = None,
    max_length: int = 512,
):
    return calibrate_contrastive_remote_gpu.remote(
        model_dir=model_dir,
        data_subdir=data_subdir,
        calibrate_for=calibrate_for,
        target_acc=target_acc,
        target_recall=target_recall,
        save_to=save_to,
        max_length=max_length,
    )


@app.local_entrypoint()
def evaluate_contrastive(
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    data_subdir: str = "/vol/data/processed",
    calibration_path: str | None = None,
):
    return evaluate_contrastive_remote.remote(
        model_dir=model_dir,
        data_subdir=data_subdir,
        calibration_path=calibration_path,
        max_length=512,
    )


# -------------------------------
# Gutenberg ingestion (txt only)
# -------------------------------

def _slugify(s: str) -> str:
    import re, unicodedata
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")
    s = s.strip().lower()
    # Replace apostrophes and dashes with space
    s = s.replace("'", " ").replace("\u2019", " ").replace("-", " ")
    # Remove non-alphanumeric except spaces
    s = re.sub(r"[^a-z0-9\s]", "", s)
    # Collapse whitespace and join with underscores
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace(" ", "_") or "untitled"


def _clip_slug(slug: str, max_len: int, salt: str | None = None) -> str:
    """Bound a slug to a safe filename length and add a short hash when truncated.

    Ensures the final component stays well under typical 255-byte limits and
    avoids collisions when multiple long titles share the same prefix.
    """
    import hashlib
    if len(slug) <= max_len:
        return slug
    hsrc = (slug + (salt or "")).encode("utf-8")
    suffix = "__" + hashlib.sha1(hsrc).hexdigest()[:8]
    keep = max(1, max_len - len(suffix))
    return slug[:keep] + suffix


def _languages_from_text(text: str, max_lines: int = 400) -> list[str] | None:
    """Parse the 'Language:' header from PG text; returns a list of lowercased names or None if not found."""
    import re
    head = text.splitlines()[:max_lines]
    lang_re = re.compile(r"^\s*Language:\s*(.+)$", re.I)
    for line in head:
        m = lang_re.search(line)
        if m:
            langs = [x.strip().lower() for x in re.split(r"[,;/]", m.group(1)) if x.strip()]
            return langs or []
    return None


def _languages_from_path(path: str, max_lines: int = 400) -> list[str] | None:
    """Open a file and parse the 'Language:' header; returns list or None if not found."""
    import re
    lang_re = re.compile(r"^\s*Language:\s*(.+)$", re.I)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                m = lang_re.search(line)
                if m:
                    langs = [x.strip().lower() for x in re.split(r"[,;/]", m.group(1)) if x.strip()]
                    return langs or []
    except Exception:
        pass
    return None


def _extract_title_author(path: str, max_lines: int = 400) -> tuple[str | None, str | None]:
    import re
    title = None
    author = None
    start_marker = re.compile(r"\*\*\*\s*START OF THIS PROJECT GUTENBERG EBOOK", re.I)
    title_re = re.compile(r"^\s*Title:\s*(.+)$", re.I)
    author_re = re.compile(r"^\s*Author:\s*(.+)$", re.I)
    byline_re = re.compile(r"^\s*by\s+(.+)$", re.I)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                if start_marker.search(line):
                    break
                m = title_re.search(line)
                if m and not title:
                    title = m.group(1).strip()
                    continue
                m = author_re.search(line)
                if m and not author:
                    author = m.group(1).strip()
                    continue
                # Sometimes the line after Title is a byline
                if title and not author:
                    m = byline_re.search(line)
                    if m:
                        author = m.group(1).strip()
                        continue
    except Exception:
        pass
    # Basic cleanup
    def _clean(v: str | None) -> str | None:
        if not v:
            return None
        v = v.strip().strip("-:;., ")
        # Truncate at common separators
        for sep in ["[", "(", "{" ]:
            if sep in v:
                v = v.split(sep)[0].strip()
        return v or None
    return _clean(title), _clean(author)


def _extract_title_author_from_text(text: str, max_lines: int = 400) -> tuple[str | None, str | None]:
    import re
    title = None
    author = None
    start_marker = re.compile(r"\*\*\*\s*START OF THIS PROJECT GUTENBERG EBOOK", re.I)
    title_re = re.compile(r"^\s*Title:\s*(.+)$", re.I | re.M)
    author_re = re.compile(r"^\s*Author:\s*(.+)$", re.I | re.M)
    byline_re = re.compile(r"^\s*by\s+(.+)$", re.I)
    head = text.splitlines()[:max_lines]
    for line in head:
        if start_marker.search(line):
            break
        if not title:
            m = title_re.search(line)
            if m:
                title = m.group(1).strip()
                continue
        if not author:
            m = author_re.search(line)
            if m:
                author = m.group(1).strip()
                continue
        if title and not author:
            m = byline_re.search(line)
            if m:
                author = m.group(1).strip()
                continue
    def _clean(v: str | None) -> str | None:
        if not v:
            return None
        v = v.strip().strip("-:;., ")
        for sep in ["[", "(", "{"]:
            if sep in v:
                v = v.split(sep)[0].strip()
        return v or None
    return _clean(title), _clean(author)


def _best_pg_urls_for_id(gid: int, prefer_plain_txt_first: bool = True) -> list[str]:
    base = f"https://www.gutenberg.org/cache/epub/{gid}/"
    candidates = [
        f"pg{gid}.txt",
        f"pg{gid}-0.txt",
        f"pg{gid}.txt.utf-8",
        f"pg{gid}-utf8.txt",
        f"pg{gid}-8.txt",
    ]
    if not prefer_plain_txt_first:
        # Move pg{id}.txt lower if we prefer UTF-8 first
        candidates = [f"pg{gid}-0.txt", f"pg{gid}.txt.utf-8", f"pg{gid}-utf8.txt", f"pg{gid}.txt", f"pg{gid}-8.txt"]
    return [base + x for x in candidates]


@app.function(
    image=image_data,
    volumes={"/training": training_vol},
    timeout=60 * 60 * 12,
    # Respect account limit; caller can further gate concurrency in the driver.
    max_containers=100,
    # Give each fetch container modest CPU for cleaning work without overcommitting.
    cpu=2,
    env={"PYTHONPATH": "/workspace"},
)
def gutenberg_fetch_http_remote(
    ids: list[int] | None = None,
    urls: list[str] | None = None,
    # HTTP concurrency per container (network-bound). Tune 16–32.
    max_concurrency: int = 24,
    # Parallelize across IDs within a container; each ID tries its candidates sequentially.
    id_concurrency: int = 32,
    # Cap concurrent writes to the Modal volume per container to reduce contention.
    io_concurrency: int = 8,
    # Simple retry policy for transient mirror failures.
    retries: int = 3,
    # Hard cap per-ID total duration to avoid long-tail stalls (seconds)
    per_id_timeout_s: float = 90.0,
    connect_timeout: float = 10.0,
    read_timeout: float = 30.0,
    raw_root: str = "/training/gutenberg/http_raw",
    out_root: str = "/training/training/gutenberg",
    meta_root: str = "/training/metadata",
    prefer_plain_txt_first: bool = True,
    keep_raw: bool = False,
    skip_if_exists: bool = True,
):
    """Download Gutenberg plain-text files over HTTP and write cleaned copies into the training volume.

    - If `ids` is provided, constructs URL candidates like `.../pg{id}.txt`, with UTF-8 fallbacks.
    - If `urls` is provided, uses those URLs as-is.
    - Saves raw copies under `raw_root` (by id or hash) and cleaned copies under `out_root/{author_slug}/{title_slug}.txt`.
    """
    import asyncio
    import os
    import hashlib
    import httpx
    import time
    import random
    import shutil
    from pathlib import Path
    from standardize_training import clean_text as _clean_text

    # Ensure destination directories
    Path(out_root).mkdir(parents=True, exist_ok=True)
    if keep_raw:
        Path(raw_root).mkdir(parents=True, exist_ok=True)
    meta_path = Path(meta_root) / "index_http.jsonl"
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    # Build worklist
    work: list[tuple[str, int | None]] = []  # (url, id)
    if urls:
        for u in urls:
            work.append((u, None))
    if ids:
        for gid in ids:
            for u in _best_pg_urls_for_id(int(gid), prefer_plain_txt_first=prefer_plain_txt_first):
                work.append((u, int(gid)))

    # De-dup by URL while preserving order
    seen = set()
    uniq_work: list[tuple[str, int | None]] = []
    for url, gid in work:
        if url not in seen:
            uniq_work.append((url, gid))
            seen.add(url)

    headers = {
        "User-Agent": "writing7/1.0 (Modal; contact: please-provide)"
    }

    # Concurrency controls
    net_sem = asyncio.Semaphore(max(1, min(max_concurrency, 128)))
    io_sem = asyncio.Semaphore(max(1, min(io_concurrency, 64)))
    results: list[dict] = []

    async def _http_get_with_retries(client: httpx.AsyncClient, url: str) -> httpx.Response | None:
        delay = 0.5
        for attempt in range(max(1, retries)):
            try:
                r = await client.get(url, follow_redirects=True)
                if r.status_code == 200 and not r.headers.get("Content-Type", "").startswith("text/html"):
                    return r
            except Exception:
                pass
            if attempt < retries - 1:
                # Exponential backoff with jitter
                await asyncio.sleep(delay + random.uniform(0, 0.25))
                delay *= 2
        return None

    async def _atomic_move(src: Path, dst: Path):
        """Move src -> dst atomically if possible, with EXDEV fallback."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(src, dst)
        except OSError:
            shutil.move(str(src), str(dst))

    async def fetch_and_store(client: httpx.AsyncClient, url: str, gid: int | None):
        nonlocal results
        async with net_sem:
            try:
                r = await _http_get_with_retries(client, url)
                if r is None:
                    return False
                data = r.text  # httpx decodes with apparent encoding
            except Exception:
                return False

        # Extract minimal metadata from header
        try:
            title, author = _extract_title_author_from_text(data)
        except Exception:
            title, author = None, None
        # Language filter: include only English when explicitly specified
        langs = None
        try:
            langs = _languages_from_text(data)
        except Exception:
            langs = None
        if langs is not None and any(l != "english" for l in langs):
            return True  # treat as handled/skipped without error

        # Slugify and bound component lengths
        title_slug = _slugify(title or (f"pg{gid}" if gid else hashlib.sha1(url.encode()).hexdigest()[:10]))
        author_slug = _slugify(author or "unknown_author")
        title_slug = _clip_slug(title_slug, 120, salt=str(gid) if gid is not None else url)
        author_slug = _clip_slug(author_slug, 80)

        # Save raw (optional) — gate volume writes
        raw_path_str = None
        if keep_raw:
            raw_name = f"pg{gid}.txt" if gid else hashlib.sha1(url.encode()).hexdigest() + ".txt"
            raw_path = Path(raw_root) / raw_name
            tmp_raw = Path("/tmp/gutenberg_raw") / raw_name
            tmp_raw.parent.mkdir(parents=True, exist_ok=True)
            try:
                tmp_raw.write_text(data, encoding="utf-8", errors="ignore")
                async with io_sem:
                    await _atomic_move(tmp_raw, raw_path)
                raw_path_str = str(raw_path.relative_to(Path(raw_root).parent.parent))
            except Exception:
                pass

        # Clean in memory (CPU-light)
        try:
            cleaned = _clean_text(data)
        except Exception:
            cleaned = data

        # Stage to /tmp, then atomically move into the volume under an IO semaphore
        dst_dir = Path(out_root) / author_slug
        dst = dst_dir / f"{title_slug}.txt"
        # Skip duplicates or suffix on collision
        if dst.exists():
            if skip_if_exists:
                return True
            else:
                suffix = f"__pg{gid}" if gid is not None else "__dup"
                # Ensure filename stays within bounds after suffix is added
                title_with_suffix = _clip_slug(f"{title_slug}{suffix}", 120)
                dst = dst_dir / f"{title_with_suffix}.txt"

        tmp_dst = Path("/tmp/gutenberg_proc") / author_slug / dst.name
        tmp_dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp_dst.write_text(cleaned, encoding="utf-8")
            async with io_sem:
                await _atomic_move(tmp_dst, dst)
        except Exception:
            # Best-effort fallback to direct write under semaphore
            try:
                async with io_sem:
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    dst.write_text(cleaned, encoding="utf-8")
            except Exception:
                return False

        rec = {
            "source": "gutenberg_http",
            "url": url,
            "gutenberg_id": gid,
            "author": author or "unknown",
            "title": title or title_slug.replace("_", " "),
            "author_slug": author_slug,
            "title_slug": title_slug,
            "raw_path": raw_path_str,
            "processed_path": str(dst.relative_to(Path(out_root).parent.parent)),
            "languages": langs,
        }
        results.append(rec)
        return True

    async def main():
        limits = httpx.Limits(
            max_keepalive_connections=max(8, min(max_concurrency * 2, 256)),
            max_connections=max(8, min(max_concurrency * 4, 256)),
        )
        # httpx>=0.27 requires either a default timeout or all individual timeouts.
        timeout = httpx.Timeout(timeout=None, connect=connect_timeout, read=read_timeout, write=30.0, pool=30.0)
        async with httpx.AsyncClient(headers=headers, limits=limits, timeout=timeout, http2=True) as client:
            # If ids were provided, stop after the first successful candidate per id.
            # Run multiple IDs concurrently to avoid long-tail stalls.
            if ids:
                # Group by id
                by_id: dict[int, list[str]] = {}
                for url, gid in uniq_work:
                    if gid is None:
                        continue
                    by_id.setdefault(int(gid), []).append(url)

                id_sem = asyncio.Semaphore(max(1, min(id_concurrency, 128)))

                async def _process_one_id(gid: int, urls_for_id: list[str]):
                    async with id_sem:
                        for url in urls_for_id:
                            try:
                                ok = await asyncio.wait_for(fetch_and_store(client, url, gid), timeout=per_id_timeout_s)
                            except asyncio.TimeoutError:
                                ok = False
                            if ok:
                                break

                await asyncio.gather(*[_process_one_id(g, u) for g, u in by_id.items()])
            # URLs as-is (no id semantics)
            if urls:
                await asyncio.gather(*[fetch_and_store(client, u, None) for u, _ in uniq_work if _ is None])

    asyncio.run(main())

    # Append metadata
    if results:
        # Append once per container to reduce small writes contention
        with open(meta_path, "a", encoding="utf-8") as mf:
            for rec in results:
                import json
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Fetched {len(results)} files. Metadata appended to {meta_path}.")
    else:
        print("No files fetched.")


@app.local_entrypoint()
def gutenberg_fetch_http(
    ids_csv: str | None = None,
    start_id: int | None = None,
    end_id: int | None = None,
    url: str | None = None,
    max_concurrency: int = 16,
    prefer_plain_txt_first: bool = True,
):
    """Fetch Gutenberg plain-text files over HTTP by ids or direct URL.

    Examples:
    - Single: modal run modal_app.py::gutenberg_fetch_http --url https://www.gutenberg.org/cache/epub/1342/pg1342.txt
    - Range: modal run modal_app.py::gutenberg_fetch_http --start-id 1 --end-id 2000
    - CSV:   modal run modal_app.py::gutenberg_fetch_http --ids-csv 12,1342,2701
    """
    ids: list[int] | None = None
    urls: list[str] | None = None
    if url:
        urls = [url]
    if ids_csv:
        ids = [int(x.strip()) for x in ids_csv.split(',') if x.strip().isdigit()]
    elif start_id is not None and end_id is not None:
        ids = list(range(int(start_id), int(end_id) + 1))
    return gutenberg_fetch_http_remote.remote(
        ids=ids,
        urls=urls,
        max_concurrency=max_concurrency,
        prefer_plain_txt_first=prefer_plain_txt_first,
    )


@app.local_entrypoint()
def gutenberg_fetch_all_http(
    start_id: int = 1,
    end_id: int = 80000,
    chunk_size: int = 500,
    # How many containers to keep in-flight concurrently (<= 100 account limit)
    containers: int = 96,
    # Per-container network concurrency (HTTP requests), tuned for mirrors
    per_container_concurrency: int = 24,
    # Per-container volume write concurrency
    io_concurrency: int = 8,
    # Retries for transient fetch errors
    retries: int = 3,
    prefer_plain_txt_first: bool = True,
):
    """Fetch a full range of Gutenberg IDs over HTTP using parallel containers.

    Example:
      modal run modal_app.py::gutenberg_fetch_all_http --start-id 1 --end-id 80000 --chunk-size 500 --containers 96 --per-container-concurrency 24
    """
    ids = list(range(int(start_id), int(end_id) + 1))
    chunks: list[list[int]] = [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]
    n_chunks = len(chunks)
    print(f"Submitting {n_chunks} chunks of size {chunk_size} with up to {containers} containers in flight...")

    # Bound in-flight containers to avoid exceeding account limit and to control mirror load
    inflight: list = []
    submitted = 0
    completed = 0
    for chunk in chunks:
        call = gutenberg_fetch_http_remote.spawn(
            ids=chunk,
            max_concurrency=per_container_concurrency,
            id_concurrency=min(per_container_concurrency, 64),
            io_concurrency=io_concurrency,
            retries=retries,
            prefer_plain_txt_first=prefer_plain_txt_first,
            skip_if_exists=True,
        )
        inflight.append(call)
        submitted += 1
        if len(inflight) >= max(1, min(containers, 100)):
            # Wait for current batch to finish
            for c in inflight:
                try:
                    c.get()
                except Exception as e:
                    print(f"Container failed: {e}")
                finally:
                    completed += 1
            inflight.clear()
            print(f"Progress: {completed}/{n_chunks} chunks done.")

    # Flush remaining
    if inflight:
        for c in inflight:
            try:
                c.get()
            except Exception as e:
                print(f"Container failed: {e}")
            finally:
                completed += 1
        inflight.clear()

    print(f"All chunks completed: {completed}/{n_chunks}.")


@app.function(
    image=image_data,
    volumes={"/training": training_vol},
    timeout=60 * 60 * 12,
)
def cleanup_training_remote(
    keep_paths: list[str] | None = None,
):
    """Remove everything under /training except the specified keep_paths.

    Defaults to keeping '/training/training' and '/training/metadata'.
    """
    from pathlib import Path
    import shutil
    root = Path("/training")
    default_keep = {"training", "metadata"}
    keep = set(keep_paths) if keep_paths else default_keep
    # Normalize keep to top-level names under /training
    keep_names = set()
    for p in keep:
        name = p.strip("/").split("/")[1] if p.startswith("/training/") else p.strip("/").split("/")[0]
        if name:
            keep_names.add(name)
    removed = []
    for child in root.iterdir():
        if child.name in keep_names:
            continue
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
            removed.append(child.name)
        except Exception as e:
            print(f"Failed to remove {child}: {e}")
    print({"removed": removed, "kept": sorted(list(keep_names))})


@app.local_entrypoint()
def cleanup_training():
    return cleanup_training_remote.remote(keep_paths=["/training/training", "/training/metadata"])


def _guess_gutenberg_id(name: str) -> str | None:
    import re
    # Match 12345, 12345-0, pg12345, etc.
    m = re.search(r"(?:pg)?(\d+)", name)
    return m.group(1) if m else None


@app.function(
    image=image_data,
    volumes={"/training": training_vol},
    timeout=60 * 60 * 12,
    max_containers=100,
    cpu=2,
    env={"PYTHONPATH": "/workspace"},
)
def gutenberg_ingest_remote(
    remote: str = "rsync://aleph.gutenberg.org/gutenberg/",
    raw_root: str = "/training/gutenberg/raw",
    out_root: str = "/training/training/gutenberg",
    prefer_utf8_variants: bool = True,
    utf8_only: bool = True,
    exclude_old: bool = True,
    keep_raw: bool = False,
    meta_root: str = "/training/metadata",
    max_files: int | None = None,
    parallelism: int = 0,
    rsync_shards: int = 0,
):
    """Mirror Gutenberg .txt files into the training volume and index by author/title.

    - Mirrors only .txt (with UTF-8 variants) using rsync, preserving directory structure under raw_root.
    - Parses Title and Author from PG header and writes processed copies to out_root/{author_slug}/{title_slug}.txt
    - Appends metadata rows to /training/gutenberg/metadata/index.jsonl
    """
    import os
    import json
    import shutil
    from pathlib import Path
    from subprocess import run, CalledProcessError

    # Ensure dirs
    Path(raw_root).mkdir(parents=True, exist_ok=True)
    Path(meta_root).mkdir(parents=True, exist_ok=True)
    Path(out_root).mkdir(parents=True, exist_ok=True)

    # Build rsync include/exclude patterns
    includes = ["*/"]
    if utf8_only:
        # Strict UTF-8 only
        includes += ["*-0.txt", "*.txt.utf-8", "*-utf8.txt"]
    else:
        if prefer_utf8_variants:
            includes += ["*-0.txt", "*-8.txt", "*.txt.utf-8"]
        includes += ["*.txt"]
    rsync_cmd = [
        "rsync", "-azv", "--delete", "--delete-after", "--no-perms", "--no-owner", "--no-group", "--prune-empty-dirs",
        "--partial", "--timeout=60", "--contimeout=60", "--no-motd",
    ]
    # Order matters: first-match wins.
    # 1) Exclude old/ trees early so we don't descend into them.
    if exclude_old:
        rsync_cmd += ["--exclude", "*/old/**"]
    # 2) Allow directory recursion.
    rsync_cmd += ["--include", "*/"]
    # 3) Include desired file patterns.
    for pat in [p for p in includes if p != "*/"]:
        rsync_cmd += ["--include", pat]
    rsync_cmd += ["--exclude", "*"]
    
    # Try the requested remote, then fall back to a few known-good mirrors
    mirror_candidates = [remote]
    # Append fallbacks if not already present
    fallbacks = [
        "rsync://aleph.gutenberg.org/gutenberg/",
        "rsync://gutenberg.mirror.ac.uk/gutenberg/",
        "rsync://ftp.funet.fi/pub/mirrors/gutenberg/",
        "rsync://mirrors.ocf.berkeley.edu/gutenberg/",
        # MirrorService uses different module paths; try common ones
        "rsync://rsync.mirrorservice.org/mirror/ftp.ibiblio.org/gutenberg/",
        "rsync://rsync.mirrorservice.org/sites/ftp.ibiblio.org/gutenberg/",
    ]
    for fb in fallbacks:
        if fb not in mirror_candidates:
            mirror_candidates.append(fb)

    def _rsync_once(src: str, dst: str) -> bool:
        from subprocess import run, CalledProcessError
        for m in mirror_candidates:
            cmd = rsync_cmd + [src if src.startswith("rsync://") else (m.rstrip('/') + '/' + src.lstrip('/')), dst]
            print("Running:", " ".join(cmd))
            try:
                res = run(cmd, check=True)
                print("rsync exit code:", res.returncode)
                return True
            except CalledProcessError as e:
                print(f"rsync failed for {cmd[-2]}: {e}")
                continue
        return False

    # If sharding requested, sync numeric top-level directories with limited parallelism
    if rsync_shards and rsync_shards > 0:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        digits = [str(d) for d in range(10)]
        # Ensure local shard dirs exist
        for d in digits:
            (Path(raw_root) / d).mkdir(parents=True, exist_ok=True)
        max_workers = min(max(1, rsync_shards), len(digits))
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {}
            for d in digits:
                # Build shard-specific src (relative subpath) and shared destination root
                src = d + "/"
                dst_dir = str((Path(raw_root) / d)) + "/"
                (Path(dst_dir)).mkdir(parents=True, exist_ok=True)
                futs[ex.submit(_rsync_once, src, dst_dir)] = d
            for fut in as_completed(futs):
                d = futs[fut]
                ok = False
                try:
                    ok = fut.result()
                except Exception as e:
                    print(f"Shard {d} failed with exception: {e}")
                results[d] = ok
        ok_count = sum(1 for v in results.values() if v)
        print(f"Shard rsync complete: {ok_count}/{len(digits)} shards succeeded.")
        if ok_count == 0:
            print("All rsync shards failed; proceeding with any existing local files.")
    else:
        # Single stream rsync for entire tree
        synced = _rsync_once(remote, raw_root + "/")
        if not synced:
            print("All rsync mirrors failed; proceeding with any existing local files.")

    # Walk raw mirror and collect candidate files
    raw_files_all = sorted([p for p in Path(raw_root).rglob("*.txt*") if p.is_file()])
    # Select a single main file per Gutenberg ID
    def _gid_for(p: Path) -> str | None:
        # Prefer immediate parent directory name if numeric
        try:
            parent = p.parent.name
            if parent.isdigit():
                return parent
        except Exception:
            pass
        return _guess_gutenberg_id(p.name)

    def _score_name(name: str) -> int:
        ln = name.lower()
        if ln.endswith("-0.txt"):  # new UTF-8 canonical
            return 400
        if ln.endswith(".txt.utf-8") or ln.endswith("-utf8.txt"):
            return 350
        if ln.endswith("-8.txt"):  # ISO-8859-1 older
            return 300
        if ln.endswith(".txt"):
            return 100
        return 0

    selected: dict[str, Path] = {}
    for p in raw_files_all:
        gid = _gid_for(p)
        if not gid:
            continue
        # If utf8_only, only accept UTF-8 variants
        if utf8_only:
            nl = p.name.lower()
            if not (nl.endswith('-0.txt') or nl.endswith('.txt.utf-8') or nl.endswith('-utf8.txt')):
                continue
        cur = selected.get(gid)
        if cur is None or _score_name(p.name) > _score_name(cur.name):
            selected[gid] = p

    raw_files = list(selected.values())
    raw_files.sort()
    if max_files is not None:
        raw_files = raw_files[:max_files]
    print(f"Selected {len(raw_files)} primary files (from {len(raw_files_all)} candidates) under {raw_root}")

    meta_path = Path(meta_root) / "index.jsonl"
    seen_processed = set()
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8", errors="ignore") as mf:
                for line in mf:
                    try:
                        rec = json.loads(line)
                        seen_processed.add(rec.get("processed_path", ""))
                    except Exception:
                        pass
        except Exception:
            pass

    # Process files either sequentially or in parallel map
    appended = 0
    if parallelism and parallelism > 1:
        # Use modal map for parallel processing
        rels = [str(p.relative_to(raw_root)) for p in raw_files]
        # Stream results and append to a single index file from the driver
        with open(meta_path, "a", encoding="utf-8") as mf:
            for i, rec in enumerate(gutenberg_process_file_remote.map(rels), 1):
                if not rec:
                    continue
                if rec.get("processed_path", "") in seen_processed:
                    continue
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                appended += 1
                if i % 500 == 0:
                    print(f"[{i}/{len(rels)}] Indexed: {rec.get('processed_path')}")
    else:
        for i, src in enumerate(raw_files, 1):
            rec = _process_one_gutenberg(str(src), raw_root=raw_root, out_root=out_root)
            if not rec:
                continue
            if rec.get("processed_path", "") in seen_processed:
                continue
            with open(meta_path, "a", encoding="utf-8") as mf:
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            appended += 1
            if i % 500 == 0:
                print(f"[{i}/{len(raw_files)}] Indexed: {rec.get('processed_path')}")

    print(f"Done. Appended {appended} new records to {meta_path}.")
    print(f"Processed output under: {out_root}")
    # Optionally remove raw mirror to keep only processed files
    if not keep_raw:
        try:
            import shutil
            shutil.rmtree(raw_root, ignore_errors=True)
            print(f"Removed raw mirror at {raw_root}")
        except Exception as e:
            print(f"Could not remove raw mirror at {raw_root}: {e}")


def _process_one_gutenberg(src_path: str, raw_root: str, out_root: str) -> dict | None:
    """Helper to process a single raw .txt file and return a metadata record."""
    import shutil
    from pathlib import Path
    src = Path(src_path)
    # Extract metadata and filter non-English when header is present
    title, author = _extract_title_author(str(src))
    langs = _languages_from_path(str(src))
    if langs is not None and any(l != "english" for l in langs):
        return None
    title_slug = _slugify(title or src.stem)
    author_slug = _slugify(author or "unknown_author")
    # Bound slug lengths to avoid filesystem component overflows
    title_slug = _clip_slug(title_slug, 120, salt=str(src))
    author_slug = _clip_slug(author_slug, 80)

    # Destination path (author dir + title filename)
    dst_dir = Path(out_root) / author_slug
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{title_slug}.txt"
    if dst.exists():
        gid = _guess_gutenberg_id(src.name) or ""
        alt = dst_dir / f"{_clip_slug(title_slug + f"__pg{gid or 'dup'}", 120)}.txt"
        dst = alt

    # Copy with cleaning
    try:
        try:
            from standardize_training import clean_text as _clean_text
        except Exception:
            _clean_text = None
        raw_txt = src.read_text(encoding="utf-8", errors="ignore")
        cleaned = _clean_text(raw_txt) if _clean_text else raw_txt
        dst.write_text(cleaned, encoding="utf-8")
    except Exception:
        # Fallback raw copy
        shutil.copy2(src, dst)

    record = {
        "source": "gutenberg",
        "raw_path": str(src.relative_to(Path(raw_root).parent.parent)),
        "processed_path": str(dst.relative_to(Path(out_root).parent.parent)),
        "author": author or "unknown",
        "title": title or title_slug.replace("_", " "),
        "author_slug": author_slug,
        "title_slug": title_slug,
        "gutenberg_id": _guess_gutenberg_id(src.name),
        "size_bytes": src.stat().st_size if src.exists() else None,
        "languages": langs,
    }
    return record


@app.function(
    image=image_data,
    volumes={"/training": training_vol},
    timeout=60 * 60 * 12,
    max_containers=100,
    cpu=2,
    env={"PYTHONPATH": "/workspace"},
)
def gutenberg_process_file_remote(src_rel: str, raw_root: str = "/training/gutenberg/raw", out_root: str = "/training/training/gutenberg") -> dict | None:
    from pathlib import Path
    abs_src = str(Path(raw_root) / src_rel)
    return _process_one_gutenberg(abs_src, raw_root=raw_root, out_root=out_root)


@app.local_entrypoint()
def gutenberg_ingest(
    remote: str = "rsync://aleph.gutenberg.org/gutenberg/",
    raw_root: str = "/training/gutenberg/raw",
    out_root: str = "/training/training/gutenberg",
    prefer_utf8_variants: bool = True,
    max_files: int | None = None,
    parallelism: int = 0,
    utf8_only: bool = True,
    exclude_old: bool = True,
    rsync_shards: int = 0,
):
    """Local entrypoint wrapper for Gutenberg ingest.

    Typical usage:
    - modal run modal_app.py::gutenberg_ingest
    - modal run modal_app.py::gutenberg_ingest --max-files 50000
    """
    return gutenberg_ingest_remote.remote(
        remote=remote,
        raw_root=raw_root,
        out_root=out_root,
        prefer_utf8_variants=prefer_utf8_variants,
        max_files=max_files,
        parallelism=parallelism,
        utf8_only=utf8_only,
        exclude_old=exclude_old,
        rsync_shards=rsync_shards,
    )


# -------------------------------
# Volume wiping utilities
# -------------------------------

@app.function(
    image=image_data,
    volumes={"/vol": artifacts_vol, "/training": training_vol},
    timeout=60 * 60 * 12,
)
def wipe_volumes_remote(
    dry_run: bool = True,
    preserve_hf_cache: bool = True,
    preserve_models: bool = False,
    confirm: bool = False,
):
    """Delete contents of the training and artifacts volumes.

    - When `dry_run=True`, prints what would be removed.
    - `preserve_hf_cache`: keeps `/vol/hf` if True.
    - `preserve_models`: keeps `/vol/models` if True.
    - `confirm` must be True to actually delete.
    """
    import shutil
    from pathlib import Path

    def _collect_targets(root: Path, keep: set[str]) -> list[Path]:
        if not root.exists():
            return []
        out = []
        for p in root.iterdir():
            if p.name in keep:
                continue
            out.append(p)
        return out

    # Training volume: clear everything under /training
    train_root = Path("/training")
    keep_train: set[str] = set()
    train_targets = _collect_targets(train_root, keep_train)

    # Artifacts volume: selectively preserve cache/models
    vol_root = Path("/vol")
    keep_vol: set[str] = set()
    if preserve_hf_cache:
        keep_vol.add("hf")
    if preserve_models:
        keep_vol.add("models")
    vol_targets = _collect_targets(vol_root, keep_vol)

    print("Training volume targets:")
    for p in train_targets:
        print("  ", p)
    print("Artifacts volume targets:")
    for p in vol_targets:
        print("  ", p)

    if dry_run:
        print("Dry run: no files deleted. Set dry_run=False and confirm=True to proceed.")
        return {"train": len(train_targets), "artifacts": len(vol_targets), "deleted": 0}
    if not confirm:
        print("Refusing to delete without confirm=True.")
        return {"deleted": 0}

    deleted = 0
    for p in train_targets + vol_targets:
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)
            deleted += 1
        except Exception as e:
            print(f"Failed to delete {p}: {e}")
    print(f"Deleted {deleted} items.")
    return {"deleted": deleted}


@app.local_entrypoint()
def wipe_volumes(
    dry_run: bool = True,
    preserve_hf_cache: bool = True,
    preserve_models: bool = False,
    confirm: bool = False,
):
    return wipe_volumes_remote.remote(
        dry_run=dry_run,
        preserve_hf_cache=preserve_hf_cache,
        preserve_models=preserve_models,
        confirm=confirm,
    )


# -------------------------------
# Word count utilities (Volumes)
# -------------------------------

@app.function(
    image=image_data,
    volumes={"/training": training_vol, "/vol": artifacts_vol},
    timeout=60 * 60 * 2,
)
def count_words_remote(
    roots: list[str] | None = None,
    pattern: str = "*.txt",
    verbose: bool = False,
):
    """Walk given roots and count words in matching text files.

    - `roots`: directories to search recursively. Defaults to ['/training'].
    - `pattern`: glob for files to include (default '*.txt').
    - `verbose`: if True, prints per-file counts as it goes.
    """
    import re
    from pathlib import Path

    word_re = re.compile(r"\b\w+\b")
    roots = roots or ["/training"]

    total_words = 0
    total_files = 0
    per_root: dict[str, int] = {}

    def count_file_words(p: Path) -> int:
        cnt = 0
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    cnt += len(word_re.findall(line))
        except Exception:
            return 0
        return cnt

    for root in roots:
        root_path = Path(root)
        root_total = 0
        if not root_path.exists():
            if verbose:
                print(f"Skip missing root: {root}")
            per_root[root] = 0
            continue
        for p in root_path.rglob(pattern):
            if not p.is_file():
                continue
            n = count_file_words(p)
            root_total += n
            total_words += n
            total_files += 1
            if verbose:
                print(f"{p}: {n}")
        per_root[root] = root_total

    print("Word count summary:")
    for r, c in per_root.items():
        print(f"  {r}: {c} words")
    print(f"Total files: {total_files}")
    print(f"TOTAL WORDS: {total_words}")
    return {"total_words": total_words, "total_files": total_files, "per_root": per_root}


@app.local_entrypoint()
def count_words(
    roots_csv: str = "/training",
    pattern: str = "*.txt",
    verbose: bool = False,
):
    """Local wrapper. Example:
    - modal run modal_app.py::count_words --roots-csv /training/training/gutenberg
    - modal run modal_app.py::count_words --roots-csv "/training,/vol"
    """
    roots = [x.strip() for x in roots_csv.split(",") if x.strip()]
    return count_words_remote.remote(roots=roots, pattern=pattern, verbose=verbose)


# -------------------------------
# Diagnostics: count total chunks
# -------------------------------

@app.function(
    image=image_cpu,
    volumes={"/vol": artifacts_vol},
    timeout=60 * 60,
    env={"PYTHONPATH": "/workspace"},
)
def count_chunks_remote(
    shards_dir: str = "/vol/tmp/prepare_shards",
    manifest_path: str | None = None,
):
    """Return total books and chunks found across shard JSONLs.

    - If a manifest JSON is provided, it should contain {"shards": [paths...]}
    - Otherwise, globs shards_dir for files matching shard_*.jsonl
    """
    from pathlib import Path as _Path
    import json as _json
    from prepare_data import load_shards_jsonl

    sdir = _Path(shards_dir)
    shards: list[_Path] = []
    manifest_files: list[str] = []
    if manifest_path:
        mp = _Path(manifest_path)
        if mp.exists():
            try:
                data = _json.loads(mp.read_text(encoding="utf-8"))
                manifest_files = [str(p) for p in data.get("shards", [])]
                shards = [ _Path(p) for p in manifest_files if _Path(p).exists() ]
            except Exception:
                pass
    if not shards:
        shards = sorted(p for p in sdir.glob("shard_*.jsonl") if p.is_file())

    if not shards:
        result = {"books": 0, "chunks": 0, "shards": 0, "from_manifest": bool(manifest_files)}
        print(result)
        return result

    book_chunks, _book_metadata = load_shards_jsonl(shards)
    books = len(book_chunks)
    chunks = sum(len(v) for v in book_chunks.values())
    result = {"books": books, "chunks": chunks, "shards": len(shards), "from_manifest": bool(manifest_files)}
    print(result)
    return result


@app.local_entrypoint()
def count_chunks(
    shards_dir: str = "/vol/tmp/prepare_shards",
    manifest_path: str | None = None,
):
    """Local CLI entrypoint to return the total number of chunks.

    Example:
      modal run modal_app.py::count_chunks
      modal run modal_app.py::count_chunks --shards-dir /vol/tmp/prepare_shards --manifest-path /vol/tmp/prepare_shards/shards_manifest.json
    """
    return count_chunks_remote.remote(
        shards_dir=shards_dir,
        manifest_path=manifest_path,
    )
