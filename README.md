# Style Similarity Model

A neural model for scoring how stylistically similar two pieces of text are (0-100 scale). Designed for evaluating how well LLMs mimic writing style.

## Overview

This project trains a style embedding model using:
- **Contrastive learning** (SupCon): Learns embeddings where same-author texts cluster together
- **Topic adversary**: Pushes content/topic information out, leaving only style
- **Calibration**: Maps cosine similarity to interpretable 0-100 scores

## Quick Start

### Training on Modal (Recommended)

Prerequisites:
```bash
pip install modal
modal token new
modal volume create writing7-training
modal volume create writing7-artifacts
# Upload your book .txt files:
modal volume put writing7-training /path/to/books /training_clean/gutenberg
```

Train the model (2x H200 GPUs, ~8-12 hours, ~$30-50):
```bash
modal run modal_app.py::train_contrastive_remote_multi_gpu
```

That's it. The defaults are configured for optimal style learning:
- `contrastive_only=True` - Pure embedding learning, no classifier
- `use_topic_adversary=True` - Disentangles style from content
- `adv_lambda=0.4` - Adversary weight
- `batch_size=32`, `lr=2e-5`, `epochs=6`

Single GPU alternative:
```bash
modal run modal_app.py::train_contrastive_remote_gpu
```

### Inference

After training, get style similarity scores:

```python
from inference_contrastive import ContrastiveBookMatcherInference

model = ContrastiveBookMatcherInference("/path/to/model/final")

# Get style similarity (0-1 scale, multiply by 100 for percentage)
result = model.style_similarity(text1, text2)
score = result["score_0_1"] * 100  # 0-100 scale

print(f"Style similarity: {score:.1f}%")
```

Or from command line:
```bash
python inference_contrastive.py \
    --model models/style_embedder/final \
    --text1 "First text here..." \
    --text2 "Second text here..."
```

### Calibration

After training, calibrate to get well-scaled 0-1 scores:
```bash
modal run modal_app.py::calibrate_style_similarity_gpu \
    --model-dir /vol/models/style_embedder/final \
    --pairs-csv /vol/data/style_pairs_autogen.csv
```

## How It Works

### Training Objective

1. **Contrastive Loss**: Pull same-author chunk embeddings together, push different-author chunks apart
2. **Topic Adversary**: A semantic encoder extracts content features; gradient reversal forces the style encoder to NOT encode content

This teaches the model: "What's consistent within an author's writing that ISN'T about content?"

Answer: Style (sentence structure, rhythm, vocabulary patterns, punctuation habits).

### Architecture

```
Text → RoBERTa Encoder → Attention Pooling → Projection → L2 Normalize → Style Embedding
                                                    ↓
                                            Topic Adversary (GRL)
                                                    ↓
                                            "Don't encode this"
```

### Why This Works for LLM Evaluation

When evaluating LLM style mimicry:
- Original: "It was the best of times..." (Dickens)
- LLM output: "The morning dawned grey..." (attempting Dickens style)

Content is different, so the model **must** rely on style features to find similarity.

## Local Training

```bash
python train_contrastive.py \
    --model roberta-large \
    --contrastive-only \
    --contrastive-mode supcon \
    --pooling attn \
    --epochs 6 \
    --batch-size 32 \
    --lr 2e-5 \
    --output models/style_embedder
```

To enable topic adversary (recommended):
```bash
python train_contrastive.py \
    --model roberta-large \
    --contrastive-only \
    --contrastive-mode supcon \
    --pooling attn \
    --adv-lambda 0.4 \
    --epochs 6 \
    --batch-size 32 \
    --lr 2e-5 \
    --output models/style_embedder
```

To disable adversary:
```bash
python train_contrastive.py ... --no-topic-adversary
```

### Fine-tuning from Checkpoint

To load weights from a previous training run:
```bash
python train_contrastive.py \
    --init-from models/style_embedder/final \
    --epochs 3 \
    --output models/style_embedder_v2
```

## Modal Commands Reference

| Command | Description |
|---------|-------------|
| `modal run modal_app.py::train_contrastive_remote_multi_gpu` | Train on 2x H200 (recommended) |
| `modal run modal_app.py::train_contrastive_remote_gpu` | Train on 1x H200 |
| `modal run modal_app.py::prepare_remote_gpu` | Prepare training data |
| `modal run modal_app.py::calibrate_style_similarity_gpu` | Calibrate model scores |

### Modal Volumes

- `writing7-training`: Raw book files (mounted at `/input`)
- `writing7-artifacts`: Models and processed data (mounted at `/vol`)
  - Datasets: `/vol/data/processed`
  - Models: `/vol/models/style_embedder`

## Directory Structure

```
writing7/
├── train_contrastive.py           # Training script (model + loss + trainer)
├── inference_contrastive.py       # Inference wrapper
├── prepare_data.py                # Data preparation + chunking
├── calibrate_style_similarity.py  # Cosine → 0-1 score calibration
├── evaluate_contrastive.py        # Test-set evaluation
├── hard_negative_mining.py        # ANN-based hard negative mining
├── standardize_training.py        # Gutenberg text cleaning
├── style_map.py                   # 2D style embedding visualization
├── modal_app.py                   # Modal cloud GPU entrypoints
├── eval/                          # Style benchmark + book fixtures
├── scripts/                       # Utilities (HF push, UMAP sweep)
├── docs/                          # Architecture documentation
├── training/                      # Raw book .txt files (not committed)
├── data/processed/                # Processed datasets (not committed)
└── models/                        # Trained model checkpoints (not committed)
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--contrastive-only` | `True` | Pure embedding learning (no classifier) |
| `--use-topic-adversary` | `True` | Enable style/content disentanglement |
| `--adv-lambda` | `0.4` | Adversary loss weight |
| `--supcon-temperature` | `0.07` | Contrastive loss temperature |
| `--pooling` | `attn` | Attention pooling over sequence |
| `--model` | `roberta-large` | Base transformer model |
