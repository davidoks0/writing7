# Book-Text Matching Classifier

An end-to-end RoBERTa-based transformer model for determining whether a text snippet belongs to a specific book.

## Overview

This project trains a binary classifier to predict if two text chunks come from the same book (positive) or different books (negative). It uses a RoBERTa-base transformer fine-tuned on pairs of text chunks from a collection of books.

## Installation

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

The training data consists of book text files in the `training/` directory. Each book is split into overlapping chunks, and positive/negative pairs are created:

- **Positive pairs**: Two chunks from the same book
- **Negative pairs**: Two chunks from different books

```bash
python prepare_data.py
```

This will:
1. Load all books from `training/`
2. Split them into chunks
3. Create train/val/test splits
4. Generate positive and negative pairs
5. Save processed data to `data/processed/`

## Training

Train the model with default settings:

```bash
python train.py
```

Customize training:

```bash
python train.py \
    --model roberta-base \
    --output models/book_matcher \
    --epochs 5 \
    --batch-size 32 \
    --lr 2e-5
```

### Contrastive Training (Recommended)

For better style similarity learning, use the contrastive approach:

```bash
python train_contrastive.py --model roberta-base --epochs 3 --batch-size 16
```

The contrastive model includes:
- **Siamese architecture** with twin encoders for style embedding
- **Contrastive loss** to learn fine-grained distinctions
- **Style features**: type-token ratio, sentence complexity, punctuation patterns, formality
- **512-token context** with hierarchical pooling

Customize:
```bash
python train_contrastive.py \
    --model roberta-base \
    --output models/book_matcher_contrastive \
    --epochs 5 \
    --batch-size 16 \
    --lr 2e-5 \
    --calibrate-for f1 \
    --target-acc 0.85 --target-recall 0.80 \
    --no-style-features  # Disable style features if desired
```

## Inference

Make predictions on new text pairs:

```bash
python inference.py \
    --model models/book_matcher/final \
    --text1 "First text chunk here..." \
    --text2 "Second text chunk here..."
```

Or use in Python:

```python
from inference import BookMatcher

matcher = BookMatcher('models/book_matcher/final')
result = matcher.predict(text1, text2)

if result['same_book']:
    print(f"Texts belong to the same book (confidence: {result['confidence']:.2%})")
```

### Contrastive Inference

For contrastive models:

```bash
python inference_contrastive.py \
    --model models/book_matcher_contrastive/final \
    --text1 "First text chunk here..." \
    --text2 "Second text chunk here..."

### Calibration-Only (no retraining)

Re-optimize temperature and decision threshold on the validation split without retraining:

```bash
python calibrate_contrastive.py \
  --model models/book_matcher_contrastive/final \
  --data data/processed \
  --calibrate-for f1 \
  --target-acc 0.85 --target-recall 0.80
```
This writes `calibration.json` next to the `final/` directory.
```

## Run on Modal

Run data prep and training on Modal with persistent storage and optional GPU.

Prerequisites:
- Install CLI: `pip install modal`
- Authenticate: `modal token new`

Quick start (baseline model):
- Create a Volume for raw training texts and upload files:
  - `modal volume create writing7-training`
  - `modal volume put writing7-training training`
- Prepare datasets from the training Volume:
  - `modal run modal_app.py::prepare`  # reads from /input/training (writing7-training)
- Train on CPU:
  - `modal run modal_app.py::train -- --model roberta-base --epochs 3`
- Train on GPU (recommended):
  - `modal run modal_app.py::train_gpu -- --model roberta-base --epochs 3`
- Pipeline (prepare → CPU train):
  - `modal run modal_app.py::pipeline -- --model roberta-base`

Contrastive model:
- CPU: `modal run modal_app.py::train_contrastive -- --model roberta-base --epochs 3`
- GPU: `modal run modal_app.py::train_contrastive_gpu -- --model roberta-base --epochs 3`
 - Calibrate: `modal run modal_app.py::calibrate_contrastive -- --model-dir /vol/models/book_matcher_contrastive/final --calibrate-for f1 --target-acc 0.85 --target-recall 0.80`

Details:
- Raw input Volume: `writing7-training` mounted at `/input` (contains your `.txt` books)
- Artifact Volume: `writing7-artifacts` mounted at `/vol`
  - Datasets: `/vol/data/processed`
  - Models/logs: `/vol/models/...`
  - HF cache: `/vol/hf`
- The image bakes this repo at `/workspace` and attempts to download spaCy `en_core_web_sm`; if unavailable, code falls back to a simple splitter.

Data flow:
- `prepare` reads raw `.txt` files from `/input`, writes processed datasets to `/vol/data/processed`.
- `train` and `train_*` read datasets from `/vol/data/processed` and write models to `/vol/models/...`.

Deploy and call (optional):
- `modal deploy modal_app.py`
- Then:
  - `modal call writing7.prepare --training-dir training`
  - `modal call writing7.train --model roberta-base`
  - `modal call writing7.train_gpu --model roberta-base`
  - `modal call writing7.train_contrastive --model roberta-base`
  - `modal call writing7.train_contrastive_gpu --model roberta-base`

GPU guidance:
- Use GPU for any substantial training run (T4 default). CPU is fine for smoke tests or tiny datasets.

## Model Architecture

### Standard Classifier

- **Base Model**: RoBERTa-base (12-layer transformer)
- **Task**: Binary sequence classification
- **Input**: Two text chunks (concatenated with special tokens)
- **Output**: Probability that chunks are from the same book

### Contrastive Model (Recommended)

- **Architecture**: Siamese twin encoders (shared RoBERTa weights)
- **Features**:
  - Embeddings: Mean-pooled RoBERTa hidden states
  - Style features: Type-token ratio, sentence length, punctuation frequency, formality
  - Concatenation: Embeddings + style features → classifier
- **Loss**: Classification loss + contrastive loss (10% weight)
- **Context**: 512 tokens per text

## Evaluation

The model is evaluated on:
- **Accuracy**: Overall correctness
- **F1 Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve
- **Precision/Recall**: Detailed classification metrics

## Example Output

```
Prediction: SAME BOOK
Confidence: 87.32%
Probability: 0.8732
```

## Directory Structure

```
writing7/
├── training/           # Raw book text files
├── data/
│   └── processed/      # Processed datasets
├── models/
│   └── book_matcher/   # Trained model
├── prepare_data.py     # Data preparation script
├── train.py            # Training script
├── inference.py        # Inference script
└── requirements.txt    # Dependencies
```
