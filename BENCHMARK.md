# LLM Style Benchmark: What It Is and How It Works

This document explains how the style similarity scoring and calibration work and how they power the LLM style benchmark in this repository. It also proposes concrete ways to improve calibration quality and discrimination.

## TL;DR

- We train a contrastive encoder that learns book/style representations while reducing topic leakage (style ≠ topic).
- Style similarity is cosine between pooled, L2‑normalized embeddings of two texts, optionally aggregated across sentence chunks.
- We calibrate raw cosine to a probability‑like [0,1] score using a labeled dev set. You can report 0–100 by multiplying by 100.
- The benchmark: sample an excerpt from a book → ask an LLM to write on a topic → score style similarity between excerpt and LLM output → aggregate across multiple samples.

## What “Style Similarity” Means Here

Style similarity is measured by embedding proximity, not content overlap. Several training choices push the representation to carry stylistic regularities and minimize topic information:

- Supervised contrastive objective (SupCon): pulls same‑book pairs together; pushes different‑book pairs apart.
- Topic adversary (GRL): predicts coarse topic labels; gradients are reversed to discourage topic signals in the encoder features.
- Optional style and symmetric features in the head to stabilize pairwise comparisons.

At inference, we do not use labels or topic supervision; we only embed text(s) and compare their embeddings.

## Scoring: Embeddings → Cosine

Code: `inference_contrastive.py::ContrastiveBookMatcherInference.style_similarity`

- Encoder: RoBERTa base/large with pooling (attention or mean) and optional projection head.
- For long texts, we split into sentence windows (chunking) and compute pairwise chunk cosines, then aggregate (default: mean). See parameters below.
- Output includes:
  - `cosine` in [-1,1]
  - `score_0_1` = (cosine+1)/2 (naive)
  - `score_calibrated` ∈ [0,1] if a calibration file is present
  - `aggregate` and `pairs` for transparency

Key parameters:
- `num_chunks`: `'auto'` (default) or integer
- `chunk_size`: sentences per chunk (default 14)
- `overlap`: sentence overlap (default 4)
- `aggregate`: `mean` or `topk_mean` (with `topk`)
- `max_length`: token cap per chunk (default 512)

## Calibration: Cosine → [0,1]

Goal: convert raw cosine to a calibrated [0,1] score that tracks the probability of “same style” as defined by your labeled dev set distribution.

Workflow:
1. Construct a CSV with `text1,text2,label` where `label` ∈ {0,1} (same/different style/book). We autogenerate from the processed dataset if none provided.
2. Compute cosines with the exact inference parameters (chunking/aggregate) used at runtime.
3. Fit a 1D mapping:
   - Logistic (Platt scaling): robust and simple; preferred default.
   - Isotonic: non‑parametric monotonic function; more flexible on large datasets.
4. Save `style_calibration.json`; inference auto‑loads and returns `score_calibrated`.

Commands (Modal):
```
modal run modal_app.py::calibrate_style_similarity_remote -- --use-gpu true
# pick automatically by CV Brier (default)
modal run modal_app.py::calibrate_style_similarity_remote -- --method auto --metric brier --n-splits 5
# or force a method
modal run modal_app.py::calibrate_style_similarity_remote -- --method isotonic
# custom pairs CSV
modal run modal_app.py::calibrate_style_similarity_remote -- --pairs /vol/data/style_pairs.csv
```

## LLM Benchmark Pipeline

Code: `eval/benchmark_style.py` and `modal_app.py::run_style_benchmark`

1. Load a `.txt` book, pick a random 15‑sentence excerpt.
2. Build a prompt: keep the reference style, but write an original story on a chosen topic (avoid copying content/entities).
3. Generate with an LLM (`openai:...`, `anthropic:...`, `gemini:...`, `kimi:...`).
4. Score style similarity between the excerpt and the output; repeat for `n_samples` with different topics/seeds.
5. Aggregate per‑run metrics (mean/median of cosine, naive 0–1, calibrated).

Run (example):
```
modal run modal_app.py::benchmark_style -- --model anthropic:claude-3-5-sonnet-20241022 \
  --book eval/books/<book>.txt
```
Defaults: 1 sample per excerpt, across 5 different excerpts (5 total generations).

Automatic empty-output retry
- The benchmark retries empty or too-short generations (default: <50 chars) up to 2 extra attempts with new seeds.
- Configure via flags in code: `retry_empty=True`, `retry_attempts=2`, `min_chars=50`.

## Is the description “divorced from topic” accurate?

Mostly: the training explicitly discourages topic information in the representation via a GRL topic adversary and contrastive supervision across different topics, so the cosine reflects style rather than topical overlap. It is not perfectly topic‑free (no method is), but in practice we observe improved topic robustness versus naive embedding cosine.

## Improving Calibration Quality and Discrimination

Below are practical levers to make the calibrated score sharper and more reliable for LLM benchmarking.

1) Calibration Data Quality
- Balance: roughly equal positive/negative pairs.
- Hard negatives: include near‑style but different authors; within‑genre cross‑author; within‑book cross‑section negatives.
- Topic separation: ensure negatives span both same‑topic and different‑topic cases to stress topic‑invariance.
- LLM outputs in calibration: include a slice of LLM generations paired with their reference excerpts, to match the benchmark distribution.

2) Chunking/Aggregation Consistency
- Calibrate with the same `num_chunks`, `chunk_size`, `overlap`, and `aggregate` you plan to use during benchmarking. Mismatch reduces calibration fidelity.
- Consider `topk_mean` for long/heterogeneous texts; calibrate with that setting if you use it at inference.

3) Calibration Method and Diagnostics
- Try both logistic and isotonic; select by validation Brier score or ECE (expected calibration error).
- Reliability plots: visualize predicted probability vs empirical frequency; check for S‑curve or flat spots.
- Cross‑validation: k‑fold across authors/books to avoid overfitting to a subset.

4) Feature‑Aware Calibration (Lightweight)
- Add simple covariates to the calibrator: e.g., pair length stats, number of chunk pairs, or aggregated variance across chunk cosines. Train a small logistic model on `[cosine, n_pairs, len_ratio, chunk_var]` → probability, constrained to be monotonic in cosine via regularization.

5) Robust Aggregation
- Trimmed mean: drop top/bottom 10% of chunk cosines to reduce outlier impact; calibrate using the same rule.
- Coverage‑aware: require a minimum number of valid chunk pairs; otherwise report “uncertain”.

6) Training‑Time Improvements (Topic Disentangling)
- Stronger adversary: increase GRL scale gradually and/or add multi‑head topic prediction at different layers.
- Independence penalties: HSIC/MMD penalties between embeddings and topic labels.
- Data curation: more cross‑topic positives (same book, different chapter themes) and near‑miss negatives (similar era/genre).

7) Benchmark‑Specific Tuning
- Per‑topic banding: if you evaluate repeated topics (e.g., “AI data center”), fit a small per‑topic calibration offset based on a few labeled pairs.
- Mixture calibration: learn separate isotonic maps for short vs long outputs, then pick based on text length at inference.

8) Reporting and Thresholds
- Optimize for metric of interest (AUC/PR‑AUC, F1 at threshold) on a held‑out set when picking the logistic threshold for labeling.
- Monitor drift by periodically recalibrating with fresh pairs, especially if you rotate in new books or change chunking.

## Quick Pointers to Code

- Style scoring: `inference_contrastive.py::ContrastiveBookMatcherInference.style_similarity`
- Calibration routine: `calibrate_style_similarity.py` (logistic/isotonic) → writes `style_calibration.json`
- Modal endpoints: `modal_app.py::calibrate_style_similarity_remote(_gpu)`, `style_similarity_remote(_gpu)`
- Benchmark orchestration: `eval/benchmark_style.py`, `modal_app.py::run_style_benchmark`

## Interpreting the Calibrated Score

- Prefer `score_calibrated` for comparisons across models/topics; it is distribution‑aware.
- If absent (uncalibrated), use `cosine` for diagnostics and `score_0_1` only as a naive proxy.
- To report 0–100, use `round(100 * score_calibrated)`.

## Next Steps (Optional Enhancements)

- Add save flag to persist JSON of each run (e.g., `/vol/benchmarks/…`).
- Publish weights on Hugging Face and add an auto‑download step.
- Add a reliability plot script over a labeled evaluation set.
- Provide a “matrix runner” to evaluate many (model × topic × book) tuples and export CSV/JSONL.
