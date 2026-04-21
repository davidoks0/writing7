# Standalone Setup

This document describes the production Modal workflow for the Gutenberg style system in this repo.

## Prerequisites

- Modal CLI authenticated for your target environment
- Hugging Face token available if you want private model downloads/uploads
- enough Modal GPU quota for transformer training

## Install Local Dependencies

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.lock.txt
```

## Create Modal Objects

```bash
modal volume create stylebench-corpus
modal volume create stylebench-artifacts
modal volume create stylebench-hf-cache
modal secret create stylebench-huggingface HF_TOKEN=YOUR_TOKEN
```

Optional benchmark-provider secret:

```bash
modal secret create stylebench-providers \
  OPENAI_API_KEY=YOUR_OPENAI_KEY \
  ANTHROPIC_API_KEY=YOUR_ANTHROPIC_KEY
```

You can also let the app create missing volumes:

```bash
modal run modal_app.py::setup_modal_objects
```

## Corpus Acquisition

Fetch the official Gutenberg machine-readable catalog first if you want stronger metadata during raw-file discovery:

```bash
modal run modal_app.py::corpus_fetch_gutenberg_catalog
modal run modal_app.py::build_gutenberg_catalog_index
```

For full-corpus work, prefer rsync:

```bash
modal run modal_app.py::corpus_ingest_gutenberg_rsync
```

Use `--max-files 50000` only for a bounded dry run.

For a sharded bulk ingest:

```bash
modal run modal_app.py::corpus_ingest_gutenberg_rsync --rsync-shards 10
```

For incremental HTTP fetches over a numeric Gutenberg id range:

```bash
modal run modal_app.py::corpus_fetch_all_gutenberg_http \
  --start-id 1 \
  --end-id 80000 \
  --chunk-size 500 \
  --containers 64 \
  --per-container-concurrency 12
```

## Build The Corpus

```bash
modal run modal_app.py::build_corpus_manifests --corpus-config configs/corpus_v1.json
modal run modal_app.py::freeze_corpus_splits --corpus-config configs/corpus_v1.json
modal run modal_app.py::build_passage_pools --corpus-config configs/corpus_v1.json
```

The corpus builder scans fetched raw files under `/corpus/raw/gutenberg/http` and `/corpus/raw/gutenberg/rsync`, optionally enriches them from `/corpus/meta/gutenberg_catalog_v1.jsonl`, and emits the canonical cleaned/manifold artifacts under `/corpus/meta` and `/corpus/splits`.

When you bootstrap from a local input manifest, the builder also copies those raw source files under `/corpus/raw/local_manifest` so the resulting corpus stays self-contained and the manifests only store root-relative paths.

## Train The Style Scorer

```bash
modal run modal_app.py::build_scorer_dataset --config configs/scorer_train_v1.json
modal run modal_app.py::train_style_scorer --config configs/scorer_train_v1.json
modal run modal_app.py::calibrate_style_scorer --config configs/scorer_train_v1.json
```

The production config uses:

- `roberta-large`
- attention pooling
- projection head
- semantic content clustering with `sentence-transformers/all-MiniLM-L6-v2` when available
- topic-adversarial training with gradient reversal
- dual-view inference over `raw` and `style_masked_v1`
- chunked multi-window scoring with configurable overlap and top-k aggregation
- transformer training backend
- Modal GPU execution for training

The default production config now enables topic-adversarial training. Local smoke mode still falls back cleanly to the bag-of-words scorer path if heavyweight model bootstrap fails.

Scorer artifacts now record:

- the training `text_view`
- inference `score_text_views`
- `blend_weights`
- chunking settings
- whether the topic adversary was enabled
- the semantic teacher model id used for content labels

Training also writes `diagnostics_v1.json` next to the final scorer so you can inspect same-author accuracy, topic-confusable negatives, and raw-vs-masked behavior after each run.

## Build Benchmark Manifests

```bash
modal run modal_app.py::build_benchmark_manifests --benchmark-config configs/benchmark_v1.json
```

Artifacts are written under `/artifacts/benchmark/manifests`.

The benchmark manifest build now also writes target reference-group sidecars for originality checks, including full target-book text where available.

## Run And Aggregate The Benchmark

```bash
modal run modal_app.py::run_benchmark --track author --split test --model stub:fixed_prose
modal run modal_app.py::aggregate_benchmark --input results/benchmark_runs/author_test_stub_fixed_prose.jsonl
```

For hosted providers, either export credentials in your shell before `modal run` or point the app at the named secret you created:

```bash
export STYLEBENCH_PROVIDER_SECRET_NAME=stylebench-providers
modal run modal_app.py::run_benchmark --track author --split test --model openai:gpt-4o-mini
```

The benchmark runner currently supports `stub:`, `openai:`, and `anthropic:` model prefixes.

## Hugging Face Artifact Management

Upload:

```bash
python scripts/push_scorer_to_hf.py \
  --model-dir /path/to/scorer/final \
  --repo-id your-org/your-style-scorer
```

Download:

```bash
python scripts/download_scorer_from_hf.py \
  --repo-id your-org/your-style-scorer \
  --out-dir models/style_embedder/final
```

The uploaded/downloaded scorer directory is expected to be self-contained for benchmarking, including
`style_calibration_v1.json`, `diagnostics_v1.json`, `config.json`, and `scorer_manifest.json`.

## Local Smoke Validation

The local smoke path stays intentionally lightweight:

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m unittest discover -s tests -v
```

This verifies the end-to-end scaffolding without requiring Modal, GPUs, or model downloads.
