# Writing7 on Modal: Data Prep, Training, and Inference

This guide documents the end‑to‑end workflow to prepare data, train transformer models (baseline and contrastive), and run inference on Modal with GPU support and persistent storage. It also covers recent upgrades: calibration, supervised InfoNCE loss, and embedding‑based hard negatives.

## Overview

- Data prep: Build Hugging Face datasets from raw book `.txt` files with fast sentence splitting and chunking, saved to a persistent Modal Volume.
- Training (baseline): Sequence‑pair classifier using Hugging Face `Trainer`.
- Training (contrastive): Siamese RoBERTa with style features and supervised InfoNCE in‑batch contrastive loss.
- Inference: Contrastive model inference function with optional calibrated probabilities and thresholding.
- Storage: Separate Volumes for raw input and artifacts (datasets/models/HF cache) to keep uploads small and reuse outputs across runs.

## Key Files

- `modal_app.py`: Modal app entrypoints
  - `prepare_remote`, `train_remote`, `train_remote_gpu`
  - `train_contrastive_remote`, `train_contrastive_remote_gpu`
  - `infer_contrastive_remote`
  - Local entrypoints for `modal run`: `prepare`, `train`, `train_gpu`, `train_contrastive`, `train_contrastive_gpu`, `infer_contrastive`, `pipeline`
- `prepare_data.py`: Loads `.txt` books, splits sentences (spaCy sentencizer or regex), creates overlapping chunks, builds HF `DatasetDict` of pairs (pos/neg), and saves to disk. Optional embedding‑based hard negative mining via Sentence‑Transformers.
- `train.py`: Baseline text‑pair classifier using HF `Trainer`.
- `train_contrastive.py`: Siamese RoBERTa + style features; custom collator and trainer; supervised InfoNCE loss; calibration saved to `calibration.json`.
- `inference_contrastive.py`: Loads trained contrastive model (PyTorch or `safetensors`), temperature‑scales logits, applies threshold, and predicts.
- `.modalignore`: Excludes large/local directories from code upload to speed up builds.

## Modal App & Images

- App name: `writing7`
- Volumes:
  - `writing7-training2` mounted at `/input` (raw `.txt` books)
  - `writing7-artifacts` mounted at `/vol` (datasets, models, HF cache)
- Storage conventions:
  - Datasets: `/vol/data/processed`
  - Models: `/vol/models/book_matcher` and `/vol/models/book_matcher_contrastive`
  - HF cache: `/vol/hf` (via `HF_HOME`)
- Images:
  - CPU image: installs `requirements.txt` and spaCy model; includes only necessary code files.
  - GPU image: installs `requirements.txt` then CUDA PyTorch `2.5.1+cu121`.

## Prerequisites

- Modal CLI installed and logged in: `pip install modal-client` then `modal token new`
- Create Volumes once:
  - `modal volume create writing7-training2`
  - `modal volume create writing7-artifacts`
- Upload raw `.txt` books to the training volume (root becomes `/input` at runtime):
  - `modal volume put writing7-training2 ./training /training`
  - Your files end up at `/input/training/*.txt` inside the container.

## Prepare Datasets

Runs sentence splitting and chunking, then creates train/val/test pairs and saves to `/vol/data/processed`.

- Defaults (tuned): `chunk_size=14`, `overlap=4`, `max_chunks_per_book=800`, embedding-based hard negatives ON with `num_chunks_for_embed=80`, `num_hard_negative_books=50`, and `n_negative_per_book=40`.
- Quick run with defaults:
  - `modal run modal_app.py::prepare --training-dir /input/training`
- Optional: add model‑mined negatives (expensive on CPU):
  - `modal run modal_app.py::prepare --training-dir /input/training --use-model-mined-negatives true --miner-model contrastive --n-mined-trials 200 --n-mined-keep 20`
- Notes:
  - Optimized one‑pass sentence splitting via spaCy sentencizer; falls back to regex.
  - Resources: `prepare_remote` requests `cpu=8`, `memory=16GiB`, timeout 6h.

## Train Models

Baseline (GPU recommended)
- `modal run modal_app.py::train_gpu --model roberta-base --epochs 3 --batch-size 16`
- Outputs to `/vol/models/book_matcher`

Contrastive (GPU recommended)
- Defaults (tuned): `model=roberta-large`, `epochs=5`, `batch_size=8`, `lr=1e-5`, `warmup_steps=1000`, `contrastive_weight=0.2`, `calibrate_for=accuracy`.
- Distillation: If `/vol/models/book_matcher/final` (cross-encoder) is missing, it is trained first (3 epochs, roberta-large) and used as a teacher; otherwise the existing cross-encoder is reused.
- Quick run with defaults:
  - `modal run modal_app.py::train_contrastive_gpu`
  - Note: This command now auto-runs data prep by default and, if a prior contrastive checkpoint exists at `/vol/models/book_matcher_contrastive/final`, enables the experimental ANN hard-negative mining during prepare with no extra flags.
- Increase separation via InfoNCE weight:
  - `modal run modal_app.py::train_contrastive_gpu --contrastive-weight 0.3`
- Outputs to `/vol/models/book_matcher_contrastive` and uses `/vol/models/book_matcher` as teacher.

## One-Shot Pipeline (Prepare + Train Contrastive)

Run the full pipeline in one command (uses tuned defaults above):

- `modal run modal_app.py::pipeline_contrastive --training-dir /input/training`

You can override any of the prepare/training flags, e.g.:

- `modal run modal_app.py::pipeline_contrastive --training-dir /input/training --num-hard-negative-books 80 --n-negative-per-book 50 --contrastive-weight 0.25`
- Implementation highlights:
  - Siamese encoders (shared RoBERTa, no pooling layer) with mean pooling.
  - Style features per side: `[type_token_ratio, punct_ratio, avg_sentence_len]`.
  - Supervised InfoNCE in‑batch contrastive loss + CE loss (alpha=0.1).
  - Custom data collator handles `input_ids_1/_2`, `attention_mask_1/_2`, and style features.
  - Post‑training calibration on validation set: temperature scaling + threshold search saved to `calibration.json`.

Common Tips
- OOM on GPU: lower `--batch-size` (e.g., 8).
- More training: increase `--epochs` to 5 (watch overfitting) or try a larger base (e.g., `roberta-large`).
- Longer chunks: increase `--chunk-size` (30–40) for stronger style signal.

## Inference (Contrastive)

- Local entrypoint:
  - `modal run modal_app.py::infer_contrastive --text1 "..." --text2 "..." --model-dir /vol/models/book_matcher_contrastive/final`
- Options:
  - `--threshold <float>` override; otherwise uses `calibration.json` if present, then 0.5.
  - Calibration file is expected at `../calibration.json` relative to `final/` folder.
- Loader supports both `pytorch_model.bin` and `model.safetensors`.
- Output fields: `same_book` (bool), `confidence` (0–1), `probability` (P[same_book]).

## Inference (Two-Stage: Bi-encoder + Cross-encoder)

- Local entrypoint:
  - `modal run modal_app.py::infer_two_stage --text1 "..." --text2 "..." \\
     --bi-model-dir /vol/models/book_matcher_contrastive/final \\
     --cross-model-dir /vol/models/book_matcher/final`
- Behavior:
  - Runs contrastive model first; if it predicts negative (below its threshold), returns negative immediately.
  - If positive, runs cross-encoder and returns its score/decision as final.
  - This is the recommended default inference path after `train_contrastive_gpu`.
- Options:
  - `--prefilter-threshold <float>`: override the bi-encoder threshold (else use calibration/defaults).
  - `--cross-threshold <float>`: threshold for cross-encoder decision (default 0.5).
  - To use accuracy‑oriented behavior, consider lowering the prefilter threshold to admit more pairs to the cross-encoder re-ranker.

## Deploy and Call Remotely

Deploy once so you can call functions without re‑uploading code:
- `modal deploy modal_app.py`

Then call functions:
- Prepare: `modal function call writing7.prepare_remote --kwargs '{"training_dir":"/input/training"}'`
- Train (GPU contrastive): `modal function call writing7.train_contrastive_remote_gpu --kwargs '{"model_name":"roberta-base","num_epochs":3,"batch_size":16}'`
- Infer: `modal function call writing7.infer_contrastive_remote --kwargs '{"text1":"...","text2":"...","model_dir":"/vol/models/book_matcher_contrastive/final"}'`

## Managing Volumes

- List artifacts: `modal volume ls writing7-artifacts /vol/models`
- Download models locally: `modal volume get writing7-artifacts /vol/models/book_matcher_contrastive ./models/book_matcher_contrastive`
- Inspect datasets: `modal volume ls writing7-artifacts /vol/data/processed`

## Performance Notes

- Data prep speedups:
  - One‑pass sentence splitting with spaCy sentencizer (fallback to regex).
  - Configurable limits via `--max-chunks-per-book`.
  - More CPUs and memory requested for the prepare function.
- Optional: fully parallel prepare using Modal `map` to shard per‑book processing across workers.

## Troubleshooting

- Argument parsing with `modal run`: pass options directly after the function (no extra `--`).
- Siamese collator: custom collator used to avoid HF default expecting `input_ids` only.
- Shape mismatches: aligned classifier input as `hidden*2 + 2*style_dim`.
- Missing weights at inference: loader supports both PyTorch and `safetensors` formats.
- GPU OOM: reduce `--batch-size`.
- Kernel version warning: informational in Modal; training proceeds.

## Environment & Caching

- HF cache: `HF_HOME=/vol/hf` to persist downloads.
- `TOKENIZERS_PARALLELISM=false` to avoid warnings (set true for faster tokenization if desired).
- Transformers advisory warnings disabled.

## Ignore Large Local Dirs

`.modalignore` excludes heavy/irrelevant directories to speed up code upload:
- `venv/`, `.git/`, `__pycache__/`, local `data/`, `models/`, `logs/`, and raw `training/` (since we mount it via a Volume).

## Reproducibility

- Seeds are set in data prep.
- Keep chunking params and splits stable for fair comparisons.

## Results Snapshot

Recent contrastive run (3 epochs, `roberta-base`) achieved:
- `F1 0.81`, `AUC 0.85`, `Accuracy 0.79`
- Saved at `/vol/models/book_matcher_contrastive/final` with `calibration.json` in parent folder.

---

Next ideas to consider:
- Parallelized data prep across many workers
- HTTP endpoint for inference
- Baseline (non‑contrastive) inference function
- Evaluation scripts to compare model variants and plot calibration
