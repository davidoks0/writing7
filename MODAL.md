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
  - `writing7-training` mounted at `/input` (raw `.txt` books)
  - `writing7-artifacts` mounted at `/vol` (datasets, models, HF cache)
- Storage conventions:
  - Datasets: `/vol/data/processed`
  - Models: `/vol/models/book_matcher` and `/vol/models/book_matcher_contrastive`
  - HF cache: `/vol/hf` (via `HF_HOME`)
- Images:
  - CPU image: uses `uv` to install `requirements.txt` (faster resolver/downloader) and the spaCy model; includes only necessary code files.
  - GPU image: uses `uv` and installs CUDA PyTorch `2.5.1+cu121` first, then the rest of `requirements.txt`.

## Prerequisites

- Modal CLI installed and logged in: `pip install modal-client` then `modal token new`
- Create Volumes once:
  - `modal volume create writing7-training`
  - `modal volume create writing7-artifacts`
- Upload raw `.txt` books to the training volume (root becomes `/input` at runtime):
  - `modal volume put writing7-training ./training /training`
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

### Sharded Prepare (50k+ books)

For large corpora, shard the CPU‑heavy chunking step across many containers, then finish the neighbor mining and pairing on GPU.

- Run sharded prepare with tuned defaults (100 containers × 8 CPU workers each):
  - `modal run modal_app.py::prepare_sharded --training-dir /input/training`
- Flags:
  - `--containers 100` (default): number of containers (capped at 100).
  - `--per-container-workers 8` (default): CPU processes per container during chunking.
  - `--chunk-size 14 --overlap 4 --max-chunks-per-book 800` as usual.
  - Downstream build on GPU uses the same hard‑negative settings as `prepare_gpu`.
- Output: same dataset folder at `/vol/data/processed`.
- Internals:
  - Stage 1 (CPU): per‑container chunking writes shards under `/vol/tmp/prepare_shards/shard_*.jsonl`.
  - Stage 2 (GPU): waits for all shards, merges, dedups, mines neighbors on GPU, and builds pairs.

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

## One-Shot Pipeline (GPU: Prepare → Train → Calibrate)

Run the full pipeline on GPU end-to-end (prepare datasets, train contrastive, calibrate classifier + style):

 - `modal run modal_app.py::pipeline_contrastive --training-dir /input/training`
  - Auto-switches to sharded prepare when large: if book count ≥ 50k under `training_dir`, it runs the distributed prepare (96 containers by default) and then merges on GPU automatically. Override with `--sharded-switch-threshold`.

What it does:
- Prepare (GPU): sentence splitting, chunking, dedup, and hard-negative mining (book embeddings on GPU).
- Train contrastive (GPU): bf16/FP16 + TF32 enabled, gradient checkpointing; dataloader workers tuned for throughput.
- Calibrate contrastive (GPU): temperature + decision threshold on validation; writes `calibration.json`.
- Calibrate style (GPU): cosine→probability mapping (logistic/isotonic via CV); writes `style_calibration.json`.

You can override any prepare/training flags, e.g.:

- `modal run modal_app.py::pipeline_contrastive --num-hard-negative-books 80 --n-negative-per-book 50 --contrastive-weight 0.25`

Implementation highlights:
- Siamese encoders (shared RoBERTa) with attention pooling.
- Style features per side: `[type_token_ratio, punct_ratio, avg_sentence_len]`.
- Supervised InfoNCE in‑batch contrastive loss + CE loss.
- GPU optimizations: bf16/FP16, TF32, grad checkpointing; dataloader workers=8.

Common Tips
- OOM on GPU: lower `--batch-size` (e.g., 8).
- More training: increase `--epochs` to 5 (watch overfitting) or try a larger base (e.g., `roberta-large`).
- Longer chunks: increase `--chunk-size` (30–40) for stronger style signal.
 - Sharding knobs (only used when auto-switched or calling `prepare_sharded`): `--containers` (max 100) and `--per-container-workers`.

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

### Wipe Volumes (Start Fresh)

Use the built-in function to clear contents safely:

- Preview (no deletion):
  - `modal run modal_app.py::wipe_volumes --dry-run true`
- Delete everything except HF cache (default):
  - `modal run modal_app.py::wipe_volumes --dry-run false --confirm true`
- Delete everything, including HF cache and models:
  - `modal run modal_app.py::wipe_volumes --dry-run false --preserve-hf-cache false --preserve-models false --confirm true`

Notes:
- Training volume (`writing7-training`): removes everything under `/training`.
- Artifacts volume (`writing7-artifacts`): removes everything under `/vol`, optionally preserving `/vol/hf` (HF cache) and `/vol/models`.
- Keeping the HF cache avoids re-downloading models/tokenizers, which speeds up subsequent runs.

## Gutenberg Ingest (TXT → Training Volume)

- Mirror all Gutenberg `.txt` (preferring UTF‑8 variants) and index by author/title:
  - `modal run modal_app.py::gutenberg_ingest`
  - For parallel processing of cleaning/metadata (faster): `modal run modal_app.py::gutenberg_ingest --parallelism 64`
  - To reduce download size and speed things up:
    - Exclude historical duplicates: `--exclude-old true`
    - Only fetch UTF‑8 variants: `--utf8-only true` (may miss a few books)
    - Example: `modal run modal_app.py::gutenberg_ingest --parallelism 64 --exclude-old true --utf8-only true`
  - To parallelize the download itself (sharded rsync across top-level digits):
    - `modal run modal_app.py::gutenberg_ingest --rsync-shards 6`  # runs up to 6 parallel rsyncs over 0..9 shards
    - Combine with a close mirror: `--remote rsync://gutenberg.mirror.ac.uk/gutenberg/ --rsync-shards 6`

### Direct HTTP Fetch (pg{id}.txt)

If you just want the “pg{id}.txt” style files over HTTP (e.g., `https://www.gutenberg.org/cache/epub/1342/pg1342.txt`) and to place cleaned copies into the volume by author/title:

- Single URL:
  - `modal run modal_app.py::gutenberg_fetch_http --url https://www.gutenberg.org/cache/epub/1342/pg1342.txt`
- List of IDs:
  - `modal run modal_app.py::gutenberg_fetch_http --ids-csv 12,1342,2701`
- Range of IDs (tries `pg{id}.txt`, then UTF‑8 fallbacks):
  - `modal run modal_app.py::gutenberg_fetch_http --start-id 1 --end-id 5000`

Fetch everything via HTTP with container parallelism
- End-to-end range (chunked across containers):
  - `modal run modal_app.py::gutenberg_fetch_all_http --start-id 1 --end-id 80000 --chunk-size 500 --containers 96 --per-container-concurrency 24`
- Notes:
  - Caps at `--containers` in-flight (account limit 100). Default 96.
  - Per-container HTTP concurrency via `--per-container-concurrency` (default 24) and volume write throttling with `--io-concurrency` (default 8) to reduce NFS contention.
  - Built-in retries with exponential backoff + jitter; tries candidate URLs per ID in priority order (pg{id}.txt → `-0.txt` → `.txt.utf-8` → `-utf8.txt` → `-8.txt`).
  - Safe to re-run; metadata appends and processed files use collision-safe names (suffix `__pg{id}` when needed).

Notes:
- The fetcher extracts Title/Author from the header and cleans PG boilerplate before writing.
- Only-English filter: entries with a `Language:` header present and not equal to `English` are skipped.
- Cleaned files land in `/training/training/gutenberg/{author_slug}/{title_slug}.txt`.
- Metadata index (HTTP fetch): `/training/metadata/index_http.jsonl`.
- Metadata index (rsync ingest): `/training/metadata/index.jsonl`.
- Notes:
  - Cleans each file at ingest using `standardize_training.clean_text` to remove PG headers/footers, transcriber notes, and normalize whitespace.
  - File names reflect Title: e.g., `PRIDE AND PREJUDICE` → `pride_and_prejudice.txt` inside the author folder.
  - Collisions append `__pg{id}` to ensure uniqueness.
  - Use a closer rsync mirror via `--remote` for more bandwidth; default is `rsync://aleph.gutenberg.org/gutenberg/`.
  - No model architecture changes required: Title/Author are saved for future hard‑negative mining and sampling only.
  - You can limit ingestion during testing: `--max-files 20000`.

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
