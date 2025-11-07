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
TRAINING_VOL_NAME = "writing7-training2"

app = modal.App(APP_NAME)
artifacts_vol = modal.Volume.from_name(ARTIFACT_VOL_NAME, create_if_missing=True)
training_vol = modal.Volume.from_name(TRAINING_VOL_NAME, create_if_missing=True)


# Images
image_cpu = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
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

image_gpu = (
    modal.Image.debian_slim()
    # Install base requirements first
    .pip_install_from_requirements("requirements.txt")
    # Then override torch with CUDA wheels from PyTorch index
    .run_commands(
        "python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121"
    )
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


def _ensure_dirs():
    os.makedirs("/vol/data/processed", exist_ok=True)
    os.makedirs("/vol/models/book_matcher", exist_ok=True)
    os.makedirs("/vol/models/book_matcher_contrastive", exist_ok=True)
    os.makedirs("/vol/hf", exist_ok=True)


COMMON_ENV = {
    "HF_HOME": "/vol/hf",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    # Reduce CUDA fragmentation per PyTorch docs when VRAM is tight
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


@app.function(
    image=image_cpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    timeout=60 * 60 * 6,
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
    gpu="H200",
    timeout=60 * 60 * 4,
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
    gpu="H200",
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
                    use_ann_chunk_negatives=ann_ok,
                    ann_miner_model_dir=(ann_dir if ann_ok else None),
                )
            except Exception as e:
                print({"warning": f"prepare_remote failed or unavailable: {e}"})

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
    )
    print("Contrastive CPU training complete.")
    print({"test_results": test_results})


@app.function(
    image=image_gpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    gpu="H200",
    timeout=60 * 60 * 12,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def train_contrastive_remote_gpu(
    model_name: str = "roberta-large",
    num_epochs: int = 6,
    batch_size: int = 8,
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
    teacher_on_gpu: bool = True,
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
                    use_ann_chunk_negatives=ann_ok,
                    ann_miner_model_dir=(ann_dir if ann_ok else None),
                )
            except Exception as e:
                print({"warning": f"prepare_remote failed or unavailable: {e}"})

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
    )
    print("Contrastive GPU training complete.")
    print({"test_results": test_results})


@app.function(
    image=image_cpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    timeout=60 * 60 * 2,
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
    gpu="H200",
    timeout=60 * 60,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def calibrate_contrastive_remote_gpu(
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    data_subdir: str = "/vol/data/processed",
    calibrate_for: str = "accuracy",
    target_acc: float | None = None,
    target_recall: float | None = 0.85,
    save_to: str | None = None,
    batch_size: int = 256,
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
    )
    return {"status": "calibration complete (GPU)", "model_dir": model_dir}


@app.function(
    image=image_cpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    timeout=60 * 30,
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
    timeout=60 * 15,
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
    gpu="H200",
    timeout=60 * 10,
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
    timeout=60 * 45,
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
    cluster_method: str = "none",
    n_clusters: int = 12,
    hdbscan_min_cluster_size: int = 10,
    hdbscan_min_samples: int | None = None,
    drop_outliers: bool = False,
    prune_top_pct: float = 0.0,
    prune_knn_k: int = 10,
    num_chunks = 'auto',
    chunk_size: int = 14,
    overlap: int = 4,
    max_chars: int | None = None,
    embed_batch_size: int = 64,
    auto_tune_embed_batch: bool = False,
    max_length: int = 384,
):
    """Generate a 2D style map (CSV + PNG) for books under books_dir (CPU)."""
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

    return {"status": "ok", "csv": str(out_csv), "png": str(out_png), "books": len(labels)}


@app.function(
    image=image_gpu,
    volumes={"/vol": artifacts_vol, "/input": training_vol},
    secrets=[],
    gpu="H200",
    # 10k books can take longer; allow up to 2 hours
    timeout=60 * 120,
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
    interactive_html: bool = False,
    cluster_method: str = "none",
    n_clusters: int = 12,
    hdbscan_min_cluster_size: int = 10,
    hdbscan_min_samples: int | None = None,
    drop_outliers: bool = False,
    prune_top_pct: float = 0.0,
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

    # Save CSV
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
    interactive_html: bool = False,
    cluster_method: str = "none",
    n_clusters: int = 12,
    hdbscan_min_cluster_size: int = 10,
    hdbscan_min_samples: int | None = None,
    drop_outliers: bool = False,
    prune_top_pct: float = 0.0,
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
    timeout=60 * 30,
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
    gpu="H200",
    timeout=60 * 20,
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
    gpu="H200",
    timeout=60 * 90,
    env={**COMMON_ENV, "PYTHONPATH": "/workspace"},
)
def run_style_benchmark(
    model: str,
    book: str,
    n_samples: int = 3,
    n_excerpts: int = 10,
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
    n_samples: int = 3,
    n_excerpts: int = 10,
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
    # Prepare datasets
    # Auto-enable experimental ANN miner if a prior contrastive checkpoint exists
    import os as _os
    _ann_dir = "/vol/models/book_matcher_contrastive/final"
    _ann_ok = _os.path.exists(f"{_ann_dir}/pytorch_model.bin") or _os.path.exists(f"{_ann_dir}/model.safetensors")

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
        use_ann_chunk_negatives=_ann_ok,
        ann_miner_model_dir=(_ann_dir if _ann_ok else None),
    )
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
    return {"status": "pipeline_contrastive completed: prepared + trained contrastive + trained cross-encoder"}


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
    # 3) Calibrate style similarity (auto-select, GPU)
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
    return {"status": "pipeline_style completed: prepared + trained + calibrated"}
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
