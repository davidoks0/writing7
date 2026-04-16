# Contrastive-Only Training Mode

## Summary

This document describes the new `--contrastive-only` training mode, which trains the model using **pure contrastive loss** without a classification head. This mode is recommended for style similarity applications.

## The Problem with the Original Architecture

The original model was trained as a **binary classifier** ("same book" vs "different book") with contrastive loss as a minor auxiliary term (10% weight). This created a fundamental mismatch:

### Training Objective vs. Intended Use

| Training Objective | Intended Use |
|-------------------|--------------|
| Classification accuracy | Style similarity scoring |
| Decision boundary separation | Embedding space geometry |
| CE Loss (90%) + Contrastive (10%) | Cosine similarity on embeddings |

### Why This Matters

**Cross-Entropy (CE) Loss** optimizes for:
- Finding a decision boundary between classes
- Maximizing margin at the boundary
- Does NOT care about intra-class structure or embedding geometry

**Contrastive Loss** optimizes for:
- Meaningful embedding geometry
- Similar items cluster together, dissimilar items push apart
- Preserves relative distances in the space

When CE dominates (90%), the model learns representations optimized for boundary separation, not for similarity computation. Two texts from the same author could have very different embeddings—as long as they're on the correct side of the decision boundary.

### Visual Intuition

With CE-dominant training:
```
Embedding Space:
     [Book A chunks]  |  [Book B chunks]
          ●●●        |      ○○○
          ●●         |       ○○
                    boundary
```

With contrastive-dominant training:
```
Embedding Space:
    ●●●              ○○○
     ●●●            ○○○
   (tight cluster)  (tight cluster)
```

## The Solution: Contrastive-Only Mode

The new `--contrastive-only` flag trains the model with **pure contrastive loss**:

```bash
python train_contrastive.py \
    --model roberta-large \
    --contrastive-only \
    --contrastive-mode supcon \
    --output models/book_matcher_contrastive_only
```

### What Changes

| Aspect | Original Mode | Contrastive-Only Mode |
|--------|--------------|----------------------|
| Classification head | ArcFace/MLP | None |
| Primary loss | Cross-Entropy | SupCon/InfoNCE |
| Contrastive weight | 10-30% | 100% |
| Style features | Used | Disabled |
| Symmetric features | Used | Disabled |
| Topic adversary | Optional | Supported (recommended: enabled) |
| Output | Logits | Embeddings |

### Why Disable Style/Symmetric Features?

In contrastive-only mode:
- **Style features** (type-token ratio, punctuation, sentence length) are not needed because we're learning embeddings directly for similarity
- **Symmetric features** (`[h1, h2, |h1-h2|, h1*h2]`) are designed for classification heads, not embedding learning
- The model learns to encode style information directly into the embeddings

### Topic Adversary in Contrastive-Only Mode

The topic adversary is supported and recommended in contrastive-only mode. It uses gradient reversal to push content/topic information out of the style embeddings:
- Without it, the contrastive loss may learn shortcuts based on content similarity
- With it, the model is forced to rely on style features (sentence structure, rhythm, vocabulary patterns)
- Enable with `--adv-lambda 0.4` (default)

## How It Works

### Training

The contrastive loss pulls positive pairs (same book) together and pushes negative pairs (different books) apart:

```python
# Normalize embeddings
emb1_norm = F.normalize(emb1_pooled, p=2, dim=1)
emb2_norm = F.normalize(emb2_pooled, p=2, dim=1)

# Build similarity matrix
Z = torch.cat([emb1_norm, emb2_norm], dim=0)
sim = Z @ Z.T / temperature

# SupCon loss: pull same-book pairs together
books = torch.cat([book_ids_1, book_ids_2], dim=0)
P = (books.unsqueeze(0) == books.unsqueeze(1))  # Positive pairs mask
loss = -log(sum(exp(sim_positive)) / sum(exp(sim_all)))
```

### Inference

The model returns normalized embeddings that can be used directly for cosine similarity:

```python
from inference_contrastive import ContrastiveBookMatcherInference

matcher = ContrastiveBookMatcherInference("models/book_matcher_contrastive_only/final")

# Get style similarity
result = matcher.style_similarity(text1, text2)
print(f"Cosine similarity: {result['cosine']}")
print(f"Score [0,1]: {result['score_0_1']}")

# Or use predict() - automatically uses embedding similarity
result = matcher.predict(text1, text2)
print(f"Method: {result['method']}")  # 'embedding' for contrastive-only
print(f"Cosine: {result['cosine']}")
```

## Recommended Configuration

```bash
python train_contrastive.py \
    --model roberta-large \
    --contrastive-only \
    --contrastive-mode supcon \
    --supcon-temperature 0.07 \
    --pooling attn \
    --epochs 6 \
    --batch-size 32 \
    --lr 2e-5 \
    --output models/style_embedder
```

### Key Parameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `--contrastive-mode` | `supcon` | Multi-positive SupCon works best with book-level grouping |
| `--supcon-temperature` | 0.07-0.1 | Lower temperature = sharper similarity distribution |
| `--pooling` | `attn` | Attention pooling learns what to focus on |
| `--batch-size` | 32+ | Larger batches = more in-batch negatives |

## Evaluation

For contrastive-only models, traditional classification metrics (accuracy, F1) are less meaningful. Instead, evaluate:

1. **Style Similarity Correlation**: Does cosine similarity correlate with human judgments of style similarity?

2. **Retrieval Metrics**: Given a query chunk, can the model retrieve other chunks from the same book?
   - Recall@K
   - Mean Reciprocal Rank (MRR)
   - Normalized Discounted Cumulative Gain (NDCG)

3. **Clustering Quality**: Do embeddings cluster by book/author?
   - Silhouette score
   - Adjusted Rand Index (ARI)

## Migration Guide

### From Classification to Contrastive-Only

If you have an existing classification model and want to switch:

1. **Retrain** with `--contrastive-only`:
   ```bash
   python train_contrastive.py --contrastive-only --output models/new_style_embedder
   ```

2. **Update inference code**:
   ```python
   # Old: Classification-based
   result = matcher.predict(text1, text2)
   is_same = result['same_book']  # Boolean from classifier

   # New: Embedding-based (auto-detected)
   result = matcher.predict(text1, text2)
   is_same = result['same_book']  # Boolean from cosine threshold
   similarity = result['cosine']  # Raw cosine similarity
   ```

3. **Calibrate threshold**: The default threshold (0.5) maps to cosine=0.0. You may need to tune this for your use case.

## Technical Details

### Model Architecture (Contrastive-Only)

```
Input Text 1 ──┐                      ┌── Embedding 1 (normalized)
               ├── RoBERTa (shared) ──┼── Projection ──┼── Cosine Similarity
Input Text 2 ──┘                      └── Embedding 2 (normalized)
```

Components:
- **Encoder**: RoBERTa-base or RoBERTa-large
- **Pooling**: Mean, CLS, or Attention
- **Projection** (optional): Linear → GELU → LayerNorm → Dropout
- **Output**: L2-normalized embeddings

### Files Modified

- `train_contrastive.py`: Added `contrastive_only` parameter to `ContrastiveBookMatcher` and `train_contrastive()`
- `inference_contrastive.py`: Auto-detects contrastive-only models and uses embedding similarity

### Backward Compatibility

- Existing classification models continue to work unchanged
- Inference code auto-detects model type by checking for classifier weights
- All CLI arguments remain available (some are ignored in contrastive-only mode)

---

## Additional Improvements (v2)

The following improvements were also implemented:

### 1. Non-Overlapping Positive Pairs

**Problem**: With overlapping chunks (4-sentence overlap), positive pairs could share actual sentences, teaching the model to detect textual overlap rather than style.

**Solution**: New `sample_non_overlapping_pair()` function in `prepare_data.py` ensures sampled positive pairs don't share any sentences.

### 2. Author-Level Grouping

**Problem**: "Same book" ≠ "Same style". Books by the same author share style but were treated as negatives.

**Solution**: SupCon loss now uses author-level grouping when `author_ids` are available:
- Same author = positive (even if different books)
- This teaches style recognition, not book identity

### 3. Online Hard Negative Mining

**Problem**: Pre-mined hard negatives become "stale" as the model improves.

**Solution**: Added `hard_negative_weight` parameter:
- Values > 1.0 emphasize harder negatives (high similarity, wrong label)
- Applied dynamically during training based on current batch similarities

### 4. Fixed Temperature

**Problem**: Learnable temperature can be unstable and adds complexity.

**Solution**: Temperature is now a fixed buffer (default 0.07), not an `nn.Parameter`.

---

## References

- [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) (Khosla et al., 2020)
- [A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709) (SimCLR, Chen et al., 2020)
- [Understanding Contrastive Representation Learning](https://arxiv.org/abs/2102.10411) (Wang & Isola, 2020)
