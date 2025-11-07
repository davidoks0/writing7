# Calibrated Style Similarity

This document explains the embedding-based style similarity scorer, how we reduce topic leakage during training, and how we calibrate the cosine similarity to a [0,1] score suitable for benchmarks where the same style is written about different topics.

## Overview

- Purpose: Score stylistic similarity between two texts while minimizing topic influence.
- Signal: Cosine similarity between pooled, L2-normalized encoder embeddings, aggregated across chunks when texts are long.
- Calibration: Map raw cosine to a probability-like [0,1] score via a labeled dev set using logistic or isotonic calibration (or `auto` to select between them by CV Brier/ECE).

Outputs (from Modal endpoints and the Python helper):
- `cosine`: Raw cosine in [-1, 1].
- `score_0_1`: Naive linear map `(cos+1)/2`.
- `score_calibrated`: Calibrated [0,1] score (requires `style_calibration.json`).
- `aggregate`: Aggregation method for multi-chunk scoring (default `mean`).
- `pairs`: Number of chunk pairs used.

## Topic Robustness: How Style Is Disentangled

Training-time measures to reduce topic leakage:
- Supervised contrastive objective (SupCon): Pulls same-book pairs together in embedding space while pushing different-book pairs apart.
- Style features: Lightweight, hand-crafted features concatenated to embedding features (optional, enabled by default).
- Symmetric features: Elementwise ops (diff, abs, prod) across the pair to stabilize relational cues.
- Topic adversary with gradient reversal (GRL): Predicts coarse topic labels from the joint representation; GRL pushes embeddings to be topic-invariant.
- Adversary scheduling: `AdversarySchedulerCallback` gradually increases GRL scale and `adv_lambda` from 0 to target (warmup + ramp), stabilizing early training.

Implementation touchpoints:
- `train_contrastive.py` → `ContrastiveBookMatcher`, GRL-based topic head, and `AdversarySchedulerCallback` (CLI flags: `--adv-warmup-ratio`, `--adv-ramp-ratio`, `--adv-lambda`, `--grl-max-scale`, `--no-topic-adversary`).
- `evaluate_contrastive.py` adds topic-sliced negative metrics to detect topic leakage (`same_topic` vs `different_topic` negatives).

## Scoring: Embedding + Cosine with Auto‑Chunking

Code: `inference_contrastive.py` → `ContrastiveBookMatcherInference.style_similarity`.

- Encoder: RoBERTa (base/large) with attention/mean pooling; optional projection head.
- Embeddings: Pooled per text, L2-normalized. If chunked, compute all pairwise chunk cosines.
- Aggregation: Default `mean`. Diagnostic `topk_mean` available (e.g., `--aggregate topk_mean --topk 5`).
- Auto‑chunking (`num_chunks='auto'`, default):
  - If text is short (≤512 tokens and ≤chunk_size sentences) → single pass.
  - Otherwise, split into sentence windows of `chunk_size` with `overlap` and limit chunks (up to a cap).
  - Tokenization respects 512 max length per chunk and avoids tokenizer warnings by using the fast tokenizer backend when available.

Key parameters:
- `num_chunks`: `'auto'` (default) or integer.
- `chunk_size`: Sentences per chunk (default 14).
- `overlap`: Sentence overlap between chunks (default 4).
- `aggregate`: `mean` (default) or `topk_mean`.
- `max_length`: Token cap per chunk (default 512).

## Calibration: Cosine → [0,1]

Goal: Convert raw cosine to a calibrated [0,1] score reflecting the probability of “same style” according to your labeled dev set.

Workflow:
1. Prepare a CSV of labeled pairs: `text1,text2,label` where `label` ∈ {0,1}.
2. Compute cosines with the exact inference path used at runtime (same chunking/aggregation) to avoid train/test mismatch.
3. Fit a 1D mapping from cosine to [0,1]:
   - `logistic`: Parametric sigmoid; robust and simple, usually sufficient.
   - `isotonic`: Non-parametric, monotonic; can fit arbitrary shapes on enough data.
   - `auto`: Compare logistic vs isotonic by cross-validated Brier (default) or ECE and pick the best; supports GroupKFold when a group column (e.g., `book1`) is present.
4. Save to `style_calibration.json` next to the model; inference auto-loads it and returns `score_calibrated`.

Implementation:
- `calibrate_style_similarity.py` provides the calibration routine.
- `inference_contrastive.py` automatically loads `../style_calibration.json` relative to the model dir and applies it when present.

JSON format (example):
```json
{
  "style_calibration": {"method": "logistic", "coef": 4.12, "intercept": -0.87},
  "meta": {
    "n_samples": 3000,
    "num_chunks": "auto",
    "chunk_size": 14,
    "overlap": 4,
    "aggregate": "mean",
    "topk": 5,
    "max_length": 512,
    "model_dir": "/vol/models/book_matcher_contrastive/final"
  }
}
```

## Modal Usage

Calibration (GPU by default; auto-generates pairs if missing):
```
modal run modal_app.py::calibrate_style_similarity_remote
# auto-select by Brier across 5 folds
modal run modal_app.py::calibrate_style_similarity_remote --method auto --metric brier --n-splits 5
# or choose method
modal run modal_app.py::calibrate_style_similarity_remote --method isotonic
# custom pairs CSV
modal run modal_app.py::calibrate_style_similarity_remote --pairs /vol/data/style_pairs.csv
# force CPU
modal run modal_app.py::calibrate_style_similarity_remote --use_gpu false
```

Scoring (GPU):
```
modal run modal_app.py::style_similarity_remote_gpu --text1 "..." --text2 "..."
```

Scoring (CPU):
```
modal run modal_app.py::style_similarity_remote --text1 "..." --text2 "..."
```

Python API:
```python
from inference_contrastive import ContrastiveBookMatcherInference
m = ContrastiveBookMatcherInference("/vol/models/book_matcher_contrastive/final")
res = m.style_similarity(text_a, text_b)  # dict with cosine, calibrated, etc.
```

## CPU vs GPU Guidance

- Calibration on thousands of pairs → use GPU (10–30x faster with chunking).
- One-off or tiny dev sets → CPU is fine.
- Style scoring for long texts benefits from GPU due to multiple chunk forward passes.

## Interpreting Scores

- `cosine`: Raw similarity; good for diagnostics and internal sanity checks.
- `score_0_1`: Naive 0–1 mapping; not calibrated and can be misleading.
- `score_calibrated`: Preferred; interpretable and tuned to your dev distribution. Thresholds are preference/metric-dependent; choose per use case.

## Recalibration Guidance

Recalibrate when:
- You substantially change the encoder, pooling, projection, or chunking strategy.
- You switch domains or dev data distribution.
- You adjust sentence chunking parameters (chunk size, overlap, or `num_chunks`).

Keep the dev set representative and balanced. For isotonic, aim for enough pairs (e.g., ~1–3k) to avoid overfitting.

## Implementation Map

- Calibration: `calibrate_style_similarity.py`
- Inference + Scoring: `inference_contrastive.py` (`ContrastiveBookMatcherInference.style_similarity`)
- Modal endpoints: `modal_app.py`
  - `calibrate_style_similarity_remote` (defaults to GPU)
  - `calibrate_style_similarity_remote_gpu`
  - `style_similarity_remote` (CPU) / `style_similarity_remote_gpu`
- Training: `train_contrastive.py` (GRL topic adversary, scheduler, pooling, projection, ArcFace)
- Evaluation: `evaluate_contrastive.py` (topic-sliced negative metrics)

## Known Limits and Next Steps

- Calibration is dev-set-specific; verify on a held-out set if possible.
- Auto-chunking is heuristic; extreme texts (very short or highly fragmented) may need manual `num_chunks` or `topk_mean`.
- Additional disentangling ideas (optional): topic-aware sampling, independence penalties (HSIC/MMD), style robustness metrics (trimmed mean, coverage@τ).

## Changelog (high level)

- Added GRL topic adversary with warmup/ramp scheduling and defaults enabled in training.
- Implemented embedding-based style similarity with auto-chunking and mean aggregation.
- Introduced style similarity calibration (logistic/isotonic) and automatic loading at inference.
- Added Modal endpoints for calibration and scoring; made GPU and auto-generated pairs the defaults for calibration.
