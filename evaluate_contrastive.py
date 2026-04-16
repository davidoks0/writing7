"""
Evaluate a trained contrastive model on the test split using cosine similarity.
Prints metrics at various cosine thresholds.

Usage:
  python evaluate_contrastive.py \
    --model models/style_embedder/final \
    --data data/processed
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from datasets import load_from_disk

from train_contrastive import tokenize_pair, SiameseDataCollator
from inference_contrastive import ContrastiveBookMatcherInference


@torch.no_grad()
def collect_cosines_labels(model, dataloader) -> tuple[np.ndarray, np.ndarray]:
    """Collect cosine similarities and labels from all batches."""
    device = next(model.parameters()).device
    model.eval()
    cosines_list, labels_list = [], []
    for batch in dataloader:
        tbatch = {}
        for k, v in batch.items():
            tbatch[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        out = model(
            input_ids_1=tbatch['input_ids_1'],
            attention_mask_1=tbatch['attention_mask_1'],
            input_ids_2=tbatch['input_ids_2'],
            attention_mask_2=tbatch['attention_mask_2'],
        )
        # Compute cosine similarity from embeddings
        emb1 = torch.nn.functional.normalize(out['emb1_pooled'], p=2, dim=1)
        emb2 = torch.nn.functional.normalize(out['emb2_pooled'], p=2, dim=1)
        cos = (emb1 * emb2).sum(dim=1)
        cosines_list.append(cos.detach().cpu().numpy())
        labels_list.append(tbatch['labels'].detach().cpu().numpy())
    cosines = np.concatenate(cosines_list, 0)
    labels = np.concatenate(labels_list, 0)
    return cosines, labels


def _metrics_from_preds(y_true, y_score, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float('nan')
    try:
        pr_auc = average_precision_score(y_true, y_score)
    except Exception:
        pr_auc = float('nan')
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': p,
        'recall': r,
        'f1': f1,
        'auc': auc,
        'pr_auc': pr_auc,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate_contrastive(
    model_dir: str,
    data_dir: str = 'data/processed',
    threshold: float = 0.5,
    max_length: int = 512,
):
    infer = ContrastiveBookMatcherInference(model_dir)
    tokenizer = infer.tokenizer
    model = infer.model

    datasets = load_from_disk(data_dir)
    if 'label' in datasets['test'].column_names:
        datasets = datasets.rename_column('label', 'labels')
    tokenized = datasets['test'].map(
        lambda x: tokenize_pair(x, tokenizer, max_length=max_length),
        batched=True, remove_columns=[],
    )
    loader = DataLoader(tokenized, batch_size=32, shuffle=False, collate_fn=SiameseDataCollator())

    cosines, labels = collect_cosines_labels(model, loader)

    # Metrics at specified threshold
    preds = (cosines >= threshold).astype(int)
    metrics = _metrics_from_preds(labels, cosines, preds)

    # Also sweep thresholds to find best F1
    best_f1, best_thr = 0.0, threshold
    for thr in np.linspace(0.0, 1.0, 101):
        p = (cosines >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels, p, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    best_preds = (cosines >= best_thr).astype(int)
    best_metrics = _metrics_from_preds(labels, cosines, best_preds)

    out = {
        'threshold': float(threshold),
        'metrics_at_threshold': metrics,
        'best_f1_threshold': float(best_thr),
        'best_f1_metrics': best_metrics,
        'cosine_stats': {
            'mean': float(cosines.mean()),
            'std': float(cosines.std()),
            'pos_mean': float(cosines[labels == 1].mean()) if (labels == 1).any() else None,
            'neg_mean': float(cosines[labels == 0].mean()) if (labels == 0).any() else None,
        },
    }

    print(json.dumps(out, indent=2))
    return out


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Evaluate contrastive model on test set using cosine similarity')
    p.add_argument('--model', type=str, required=True, help='Path to model final dir')
    p.add_argument('--data', type=str, default='data/processed', help='Path to processed datasets')
    p.add_argument('--threshold', type=float, default=0.5, help='Cosine similarity threshold for classification')
    p.add_argument('--max-length', type=int, default=512, help='Tokenizer max_length')
    args = p.parse_args()
    evaluate_contrastive(args.model, data_dir=args.data, threshold=args.threshold, max_length=args.max_length)
