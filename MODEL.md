# Model Architecture and Training: Writing7 Book Matching

This document explains how the book–text matching models in this repo work, how the data is built, what losses and architectural choices we use, and how inference and calibration operate in practice.

## Problem

Given two text chunks, predict whether they come from the same book (label 1) or from different books (label 0).

We implement two complementary models:
- Cross‑encoder baseline (sequence pair classification; `train.py`).
- Contrastive bi‑encoder (recommended; siamese with style features, contrastive training, ArcFace head, optional adversarial topic head; `train_contrastive.py`).

---

## Data Pipeline

Code: `prepare_data.py` (plus optional cleaning in `standardize_training.py`).

- Input: `.txt` book files under `training/` (can be nested). Optional cleaning removes Project Gutenberg boilerplate and normalizes whitespace.
- Sentence splitting: spaCy rule‑based sentencizer if available; else a regex splitter.
- Chunking: Sliding window of sentences, default `chunk_size=14`, `overlap=4`, with a minimum character length filter. This balances context with sample count.
- Deduplication across books (optional, default on): fingerprint chunks (lowercased alnum text) and drop those appearing in many books to reduce boilerplate leakage.
- Splits by book, not by pair: train/val/test are disjoint on the book level to prevent leakage.
- Pair generation per split:
  - Positives: two chunks sampled from the same book.
  - Negatives: by default “harder” negatives:
    - Embedding neighbors (default): build book‑level centroids with `sentence-transformers/all-MiniLM-L6-v2`, pick nearest books as negative sources.
    - Heuristic similarity (fallback): match by coarse period/genre inferred from keywords.
    - Optional model‑mined negatives: sample candidate pairs and keep those the current model scores as most similar. Useful for sharpening the decision boundary.
- Output: HuggingFace `DatasetDict` with `train/validation/test`, each having rows of `{text1, text2, label}` saved to `data/processed/`.

---

## Cross‑Encoder Baseline (sequence pair)

Code: `train.py`, `inference.py`.

- Architecture: RoBERTa (base by default) fine‑tuned for 2‑way sequence pair classification. Inputs are concatenated with special tokens; output is a 2‑class softmax.
- Tokenization: `AutoTokenizer` with `max_length=512`, padding to max length.
- Objective: cross‑entropy on the two classes.
- Training utilities: Early stopping, mixed precision on GPU, best‑checkpoint selection by F1.
- Outputs: Saved under `models/book_matcher/final`.

When to use: strong re‑ranking precision, slower per pair. Often used as stage 2 in two‑stage inference.

---

## Contrastive Bi‑Encoder (recommended)

Code: `train_contrastive.py`, `inference_contrastive.py`.

### High‑Level Architecture

- Siamese encoders: two identical RoBERTa encoders (weight‑sharing). Each text chunk is encoded independently.
- Pooling: configurable
  - `attn` (default): a lightweight attention MLP over sequence hidden states.
  - `mean` or `cls` as alternatives.
- Optional projection: small MLP to refine pooled embeddings before classification.
- Symmetric features: concatenate `[h1, h2, |h1−h2|, h1*h2]` to encourage order‑invariance and richer interactions.
- Style features (optional; default on): 3 hand‑crafted features per side computed from raw text
  - Type–token ratio proxy
  - Punctuation ratio
  - Average sentence length
  These are concatenated to the classifier input and help capture stylistic fingerprints.
- Classifier head:
  - `mlp`: 2‑layer MLP → logits.
  - `arcface` (default): feature head → cosine logits with ArcFace margin; improves margin and calibration in embedding space.
- Topic adversary (optional; default on): a small head predicts coarse topics (religious, historical, adventure, romance, general) from each side. During training, a Gradient Reversal Layer makes the encoder invariant to this signal, reducing topic bias.

### Losses

Let CE be the main classification loss, CL the contrastive loss (only positives supervise alignment), ADV the topic adversary loss, and KD the optional distillation loss.

- CE: cross‑entropy with optional class weights and label smoothing.
- Contrastive (choose one):
  - SupCon (default): build a 2B×2B similarity matrix from normalized embeddings of both sides; supervise aligned views among positive pairs.
  - InfoNCE: dual direction in‑batch negatives (1→2 and 2→1) on positive rows.
  Temperature is a learned parameter.
- Adversary: cross‑entropy for topic heads (both sides), multiplied by `adv_lambda` through a gradient reversal (penalizes topic predictability, promoting invariance).
- Distillation (optional): KL divergence between student logits and a cross‑encoder teacher’s logits with temperature `T`, scaled by `T^2` and weighted by `distill_weight`.

Total loss (schematically):
```
L = CE + α·CL + λ·ADV + β·KD
```
where α=`contrastive_weight`, λ=`adv_lambda`, β=`distill_weight`.

### Training Details

- Tokenization: each side tokenized separately to `max_length=512`; a custom collator stacks tensors and passes style features and weak topic labels.
- Class imbalance: optional per‑class weights computed from training labels.
- Efficiency: gradient checkpointing, TF32 on Ampere, bf16/fp16 autocast as available, gradient accumulation.
- Early stopping: based on `select_metric` (default `balanced_accuracy`).
- Checkpoints and logs written under `models/book_matcher_contrastive/`.

### Calibration

After training, we calibrate on the validation set:
- Temperature scaling: fit a single scalar to minimize NLL (LBFGS).
- Threshold search: grid‑search probability threshold to maximize a target metric (`accuracy`, `balanced_accuracy`, `f1`, `f0.5`, `f2`) under optional accuracy/recall constraints.
- Results saved to `calibration.json` alongside the `final/` directory and automatically used in inference.

### Inference

`ContrastiveBookMatcherInference` auto‑detects architecture choices from checkpoint weights (base vs large encoder, presence of projection/attention/style/symmetric features, classifier type, topic head). It loads the saved temperature and decision threshold if present and returns:

```
{
  same_book: bool,
  probability: float,   # P(same)
  confidence: float     # probability if positive else 1 - probability
}
```

---

## Two‑Stage Inference (optional)

Code: `inference_two_stage.py`.

Pipeline:
1) Run bi‑encoder (fast). If below threshold, exit negative.
2) If above threshold, run the cross‑encoder re‑ranker and take its probability/decision as final.

This pattern uses the bi‑encoder to filter most negatives cheaply, reserving the slower cross‑encoder for likely positives to improve precision.

---

## Hard‑Negative Mining (experimental)

Code: `prepare_data.py` (+ `hard_negative_mining.py`).

- Options in `prepare_data.prepare_datasets` allow adding ANN‑mined chunk‑level negatives using the current contrastive model’s encoder.
- Flow: sample per‑book chunks → embed with the contrastive encoder → nearest‑neighbor search (cosine) → keep only neighbors from other books with high similarity but low model probability (`prob_max`) to avoid false negatives.
- Controlled by `use_ann_chunk_negatives` and related `ann_*` parameters. Disabled by default.
- NOTE: This is experimental and may be removed later; use when you have a stable prior contrastive checkpoint to mine against.

On Modal, pass the flags through `prepare_remote(...)` to enable.

---

## Files of Interest

- Data
  - `prepare_data.py`: build datasets, hard negatives, dedup, splits.
  - `standardize_training.py`: clean/normalize raw texts (PG boilerplate removal).
- Cross‑encoder
  - `train.py`: baseline training.
  - `inference.py`: baseline inference helper.
- Contrastive bi‑encoder
  - `train_contrastive.py`: model, losses, trainer, calibration.
  - `inference_contrastive.py`: architecture auto‑detect + calibrated inference.
  - `evaluate_contrastive.py`: side‑by‑side argmax vs calibrated metrics.
  - `calibrate_contrastive.py`: temperature + threshold optimization only.
- Modal
  - `modal_app.py`: run prepare/train/calibrate on Modal with persistent volumes and optional GPU.

---

## Quick Commands

- Prepare data:
```bash
python prepare_data.py
```

- Train contrastive model (recommended):
```bash
python train_contrastive.py --model roberta-base --epochs 5 --batch-size 16
```

- Calibrate only:
```bash
python calibrate_contrastive.py \
  --model models/book_matcher_contrastive/final \
  --data data/processed \
  --calibrate-for f1 --target-acc 0.85 --target-recall 0.80
```

- Inference (contrastive):
```bash
python inference_contrastive.py \
  --model models/book_matcher_contrastive/final \
  --text1 "..." --text2 "..."
```

---

## Design Rationale (short)

- Bi‑encoder with symmetric + style cues captures stylistic fingerprints and scales to many comparisons.
- Contrastive supervision sharpens alignment for positive pairs beyond pure CE.
- ArcFace improves decision margins in the embedding space and typically calibrates well with temperature scaling.
- Topic adversary reduces shortcuts where “topic similarity” masquerades as “same book”.
- Calibration decouples operating point from training loss and supports metric‑specific thresholds with constraints.
