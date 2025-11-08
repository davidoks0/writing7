# Writing7: System Architecture, Data Flow, and Operations

This document describes how the Writing7 system works end‑to‑end: ingestion → data prep → training → calibration → inference and benchmarking. It complements `MODAL.md` (quick commands) with deeper design, performance guidance, and troubleshooting.

## Overview

- Goal: train and serve a book‑text matcher with a contrastive bi‑encoder (plus optional cross‑encoder teacher), along with a calibrated style similarity score.
- Compute: [Modal](https://modal.com) functions orchestrate CPU/GPU workloads against persistent volumes.
- Storage:
  - Raw input volume: `writing7-training` mounted read‑only at `/input` (some utilities use `/training`).
  - Artifacts volume: `writing7-artifacts` mounted read‑write at `/vol` for datasets, models, HF cache.
- Pipelines:
  - Prepare datasets (single‑container GPU default; optional sharded CPU chunking + GPU merge).
  - Train contrastive model (and cross‑encoder teacher if missing).
  - Calibrate classifier threshold/temperature and style similarity mapping.
  - Evaluate and optionally benchmark style with an LLM (local code, remote keys).

## Repo Map (key files)

- `modal_app.py`: Modal functions, images, pipelines, orchestration.
- `prepare_data.py`: Data standardization hooks, sentence splitting, chunking, dedup, hard‑negative mining, dataset building, sharding helpers.
- `train_contrastive.py`: Model, tokenizer, Trainer config, distillation, style features, adversarial/topic head, training loop.
- `calibrate_contrastive.py`: Temperature/threshold calibration for classifier.
- `calibrate_style_similarity.py`: Map raw cosine → [0,1] style score.
- `inference_contrastive.py`: Inference wrapper for classifier + style similarity.
- `inference_two_stage.py`: Bi‑encoder prefilter + cross‑encoder reranker.
- `evaluate_contrastive.py`: Test metrics (baseline and calibrated).
- `standardize_training.py`: Boilerplate cleaning and normalization utilities.
- `eval/`: Style benchmark and helpers (LLM required for text generation).
- `MODAL.md`: Practical “how to run” guide.

## Storage Layout and Paths

- Volumes:
  - `writing7-training` → mounted at `/input` (many functions) or `/training` (ingestion/utilities).
  - `writing7-artifacts` → mounted at `/vol`.
- Important subpaths on `/vol`:
  - Datasets: `/vol/data/processed` (HuggingFace `DatasetDict` with `train/validation/test`).
  - Models:
    - Cross‑encoder: `/vol/models/book_matcher/final`.
    - Contrastive: `/vol/models/book_matcher_contrastive/final`.
  - HF cache: `/vol/hf`.
  - Prepare shards (if used): `/vol/tmp/prepare_shards/shard_*.jsonl`.
  - Style calibration: `/vol/models/book_matcher_contrastive/style_calibration.json`.

## Modal Images and Environment

- Images:
  - `image_cpu`: debian‑slim + uv + requirements + spaCy model. Used for CPU prepare, ingestion, misc.
  - `image_gpu`: same, plus CUDA torch (`cu121`). Used for GPU prepare/merge and training.
  - `image_data`: minimal data tools (rsync, httpx) for ingestion/wiping.
- Env vars (`COMMON_ENV`):
  - `HF_HOME=/vol/hf` (cache on persistent volume),
  - `TOKENIZERS_PARALLELISM=false`,
  - `PYTORCH_ALLOC_CONF=expandable_segments:True` (reduce fragmentation).

## Ingestion

Two paths:

1) HTTP fetch by Gutenberg ID (asynchronous, resilient):
   - Functions: `gutenberg_fetch_http_remote`, `gutenberg_fetch_all_http`.
   - Features: per‑ID concurrency (`id_concurrency`), per‑ID timeouts (`per_id_timeout_s`), backoff, `skip_if_exists` to avoid re‑downloads.
   - Writes to `/training/training/gutenberg/{author}/{title}.txt` (note: these helpers mount training volume at `/training`).

2) Manual upload of `.txt` into the training volume.

Standardization:
- `standardize_training.clean_text` performs unicode normalization and boilerplate removal. It is applied in prepare in two ways:
  - Single‑container prepare: pre‑clean full directory into `/input/training_clean` before chunking.
  - Sharded prepare: per‑file cleaning inside worker chunking (fast) without a full pre‑copy (optional `standardize=True` available).

## Data Preparation (single‑container, default)

Function: `prepare_remote_gpu` (GPU) or `prepare_remote` (CPU only).

Steps:

1) Discover `.txt` recursively under `training_dir` (default `/input/training`).

2) Standardize (GPU path): copy + clean to `/input/training_clean` for consistent input.

3) Chunking:
   - Sentence splitting: spaCy sentencizer if available, else regex splitter.
   - Create overlapping chunks of `chunk_size` sentences, `overlap` step; min length heuristic and a lightweight English filter.
   - Sample at most `max_chunks_per_book` chunks per book.

4) Cross‑book dedup (fingerprint): remove repeated chunks seen across many books.

5) Hard‑negative mining (embedding‑based):
   - For each book, sample up to `num_chunks_for_embed` chunks → embed with `sentence-transformers/all-MiniLM-L6-v2`.
   - Compute book‑level centroids by averaging chunk embeddings.
   - Efficient neighbor search:
     - Cross‑book batching of chunk embedding on GPU (large microbatches keep GPU saturated).
     - Top‑K neighbor mining with block matmul on GPU, computing similarities in `bfloat16` with float32 normalization.
     - Fallback to blocked CPU `argpartition` when GPU is unavailable.

6) Pair creation and split:
   - Shuffle books; split to train/val/test by `train_ratio`/`val_ratio`.
   - Positives: two chunks from the same book.
   - Negatives: chunks from different books, biased by neighbor lists (hard negatives) and simple topic heuristics.
   - Optional extras (off by default in merge): model‑mined negatives, experimental ANN chunk‑level negatives.
   - Each pair includes: `text1`, `text2`, `label`, `book1`, `book2`, `pair_type`, `topic1/2`, `same_topic`.

7) Save HF `DatasetDict` to `/vol/data/processed`.

## Sharded Prepare (optional)

Status: Off by default for reliability; enable with flags (see Pipelines). Use this for very large corpora when single prepare becomes a bottleneck.

Design:

- Stage A (CPU fan‑out): `prepare_chunk_shard_remote`
  - Orchestrator splits the file list across containers.
  - Each worker chunks its subset and writes a JSONL shard (`/vol/tmp/prepare_shards/shard_XXXX.jsonl`).
  - Atomic shard write (`*.tmp` then `os.replace`) to avoid partial reads.

- Stage B (GPU merge/build): `prepare_merge_shards_remote_gpu`
  - Fresh GPU container reads a manifest of shard paths written by the orchestrator and waits until all exist.
  - Loads shards → merges `book_chunks` and `book_metadata` in memory.
  - Runs the same dedup + hard‑negatives + pair generation as the single‑container path.
  - Saves the dataset to `/vol/data/processed`.

Notes and pitfalls:
- Modal volume visibility can lag across containers; the GPU merge reads via a manifest and polls until shard files are present.
- Pre‑standardization of the full tree is disabled by default for sharded mode; files are cleaned during chunking.
- Auto‑shard is disabled by default. See Pipelines to opt in.

## Training

Primary path: `train_contrastive_remote_gpu` (H200 GPU).

Model features (see `train_contrastive.py`):
- Base: HuggingFace encoder (default `roberta-large`).
- Projection head + contrastive loss (SupCon; configurable) with optional symmetric features and simple style features.
- Optional topic adversary and independence penalty (conservative defaults enabled for robustness).
- Distillation: if `distill_from_cross=True` and the cross‑encoder teacher is absent, the pipeline triggers `train_remote_gpu` to produce it, then trains the contrastive model with distillation.
- Memory/Speed:
  - bf16/amp where available; gradient checkpointing on encoder; dataloader workers; TF32 allowed.
  - Evaluation/saving cadence tuned for long runs (can be increased to reduce I/O).

Outputs:
- `.../book_matcher_contrastive/final`: final checkpoint.
- Calibration: `calibration.json` (temperature + threshold) written next to the model.

## Calibration and Evaluation

- Classifier calibration: `calibrate_contrastive.py` (wrapped by Modal function) runs a forward pass on validation, optimizes temperature and decision threshold for a target metric (default accuracy).

- Style similarity calibration: `calibrate_style_similarity.py` learns a monotone mapping from cosine to [0,1] probability using a labeled dev set (auto‑generated or provided). Saved to `style_calibration.json`.

- Evaluation: `evaluate_contrastive.py` prints baseline and calibrated metrics on test split.

## Inference

- Contrastive only: `inference_contrastive.py`
  - Loads `final` checkpoint, optional `calibration.json` and `style_calibration.json`.
  - Provides `score(text1,text2)` and `style_similarity(text1,text2)`.

- Two‑stage: `inference_two_stage.py`
  - Uses contrastive bi‑encoder to prefilter, then optional cross‑encoder to rerank or make final decisions.

## Pipelines (one‑click)

Local entrypoints in `modal_app.py`:

- Contrastive end‑to‑end (default: no sharding):

  ```bash
  modal run modal_app.py::pipeline_contrastive --training-dir /input/training
  ```

  Optional flags:
  - `--force-sharded`: force the sharded prepare path.
  - `--auto-shard`: enable auto switch to sharded by count; threshold via `--sharded-switch-threshold`.
  - Common prepare knobs: `--chunk-size`, `--overlap`, `--max-chunks-per-book`, `--num-chunks-for-embed`, `--num-hard-negative-books`.

- Prepare only (single container, GPU):

  ```bash
  modal run modal_app.py::prepare_remote_gpu -- --training-dir /input/training
  ```

- Prepare only (sharded; opt‑in):

  ```bash
  modal run modal_app.py::prepare_sharded -- --training-dir /input/training \
    --containers 100 --per-container-workers 8
  ```

## Performance Guidance

Implemented speedups:
- Cross‑book batching for centroid embedding on GPU (large microbatches); fewer encode calls.
- Top‑K neighbor mining with block matmul up to 2048 rows per block; bfloat16 similarity compute.
- Atomic shard writes and manifest‑driven merge (when sharded).

High‑leverage knobs:
- Reduce embed work: `--num-chunks-for-embed 40`, `--num-hard-negative-books 30`.
- Reduce dataset size: `--max-chunks-per-book 400`, `--n-positive-per-book 12`, `--n-negative-per-book 24`.
- Training cadence: increase `eval_steps`/`save_steps` to reduce checkpoint overhead.
- Sharded: use `--per-container-workers 6–8`, watch volume I/O; prefer single container if reliable/fast enough.

Backlog/ideas:
- FAISS‑GPU neighbor search for 100k+ books.
- Fused AdamW optimizer and even sparser eval save cadence.

Expected runtime (≈59k books; defaults):
- Prepare: 40–85 min (single container, GPU). Sharded A+B: ~25–60 min depending on I/O.
- Train: 3–6 h (teacher adds 1–2 h if missing).
- Calibration: 10–20 min.

## Dataset Format

Saved via `datasets.save_to_disk('/vol/data/processed')` with splits `train/validation/test`.

Fields per example:
- `text1`, `text2` (strings)
- `label` (0/1)
- `book1`, `book2` (IDs like `author_slug/title_slug` if tracked)
- `pair_type` (`positive` or `negative`), `neg_type` (optional)
- `topic1`, `topic2` (0..4), `same_topic` (bool)

Training adds derived features at tokenize‑time (style features, symmetric features), not persisted in the dataset.

## Troubleshooting

- “No shard files found” during sharded merge:
  - Orchestrator manifests are written to `/vol/tmp/prepare_shards/shards_manifest.json`; the GPU merge polls for those files on a fresh mount.
  - If shards are very slow, increase `--shard-wait-timeout-s` (default 2h) in `prepare_sharded_remote` call path.
  - Consider default single‑container prepare if sharding is flaky in your environment.

- “No training examples were created”:
  - Verify `.txt` files exist in the training volume path (`modal volume ls writing7-training /training`).
  - Check minimum chunk thresholds or too aggressive dedup.

- Teacher missing / distillation:
  - The pipeline will train the cross‑encoder teacher first (GPU) if absent. This adds ~1–2h for large datasets.

- Volume I/O contention:
  - Reduce shard workers, increase batch sizes, or remove sharding.
  - Keep eval/save cadence sparse during training.

- HF cache issues:
  - Cache is under `/vol/hf`. If corrupted, remove it and re‑run; model downloads will repopulate.

## Security and Costs

- Do not embed secrets in code. For LLM benchmarks, use Modal secrets (`llm-api-keys`).
- GPU hours dominate cost; prefer single prepare when viable; reduce training epochs or model size (`roberta-base`) for quick passes.
- Volumes persist across runs and across containers; wiping them deletes datasets / models; use wipe utilities with care.

## Extensibility

- New miners: integrate FAISS/HNSW in `prepare_data.py` (neighbor search) and expose knobs via Modal functions.
- New ingestion sources: add remotes in `modal_app.py` that write to the training volume under a consistent `{author}/{title}.txt` layout.
- New features: extend `train_contrastive.py` (model heads, losses) and adapt inference wrappers accordingly.

## Runbooks

1) First‑time setup
   - Create volumes and upload raw texts (see `MODAL.md`).
   - Optionally populate using HTTP ingest helpers.

2) One‑shot training
   - `modal run modal_app.py::pipeline_contrastive --training-dir /input/training`
   - Inspect logs; final model at `/vol/models/book_matcher_contrastive/final`.

3) Inference (local)
   - `python inference_contrastive.py --model /vol/models/book_matcher_contrastive/final --style-sim` and provide two texts.

4) Sharded prepare (only if necessary)
   - `modal run modal_app.py::prepare_sharded -- --training-dir /input/training --containers 100 --per-container-workers 8`
   - Confirm `shards_manifest.json` is written and the merge sees all shards.

