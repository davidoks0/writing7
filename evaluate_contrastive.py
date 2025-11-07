"""
Evaluate a trained contrastive model on the test split using calibrated
temperature + threshold. Prints both argmax (baseline) and calibrated metrics.

Usage:
  python evaluate_contrastive.py \
    --model models/book_matcher_contrastive/final \
    --data data/processed
"""
from __future__ import annotations

import os
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
def collect_logits_labels(model, dataloader) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    device = next(model.parameters()).device
    model.eval()
    logits_list, labels_list = [], []
    tlog1_list, tlog2_list = [], []
    tlab1_list, tlab2_list = [], []
    for batch in dataloader:
        tbatch = {}
        for k, v in batch.items():
            tbatch[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        out = model(
            input_ids_1=tbatch['input_ids_1'],
            attention_mask_1=tbatch['attention_mask_1'],
            input_ids_2=tbatch['input_ids_2'],
            attention_mask_2=tbatch['attention_mask_2'],
            style_features_1=tbatch.get('style_features_1'),
            style_features_2=tbatch.get('style_features_2'),
            topic_labels_1=tbatch.get('topic_labels_1'),
            topic_labels_2=tbatch.get('topic_labels_2'),
        )
        logits_list.append(out['logits'].detach().cpu().numpy())
        labels_list.append(tbatch['labels'].detach().cpu().numpy())
        if out.get('topic_logits_1') is not None and out.get('topic_logits_2') is not None and tbatch.get('topic_labels_1') is not None:
            tlog1_list.append(out['topic_logits_1'].detach().cpu().numpy())
            tlog2_list.append(out['topic_logits_2'].detach().cpu().numpy())
            tlab1_list.append(tbatch['topic_labels_1'].detach().cpu().numpy())
            tlab2_list.append(tbatch['topic_labels_2'].detach().cpu().numpy())
    logits = np.concatenate(logits_list, 0)
    labels = np.concatenate(labels_list, 0)
    tlog1 = np.concatenate(tlog1_list, 0) if tlog1_list else None
    tlog2 = np.concatenate(tlog2_list, 0) if tlog2_list else None
    tlab1 = np.concatenate(tlab1_list, 0) if tlab1_list else None
    tlab2 = np.concatenate(tlab2_list, 0) if tlab2_list else None
    return logits, labels, tlog1, tlog2, tlab1, tlab2


def _metrics_from_preds(y_true, y_prob, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    # Guard AUCs when only one class is present in y_true
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    try:
        pr_auc = average_precision_score(y_true, y_prob)
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
    calibration_path: Optional[str] = None,
    max_length: int = 512,
):
    # Load model + tokenizer via inference wrapper (auto-detects pooling/projection)
    infer = ContrastiveBookMatcherInference(model_dir, calibration_path=calibration_path)
    tokenizer = infer.tokenizer
    model = infer.model
    temperature = infer.temperature
    threshold = infer.threshold

    # Load dataset and tokenize test split
    datasets = load_from_disk(data_dir)
    if 'label' in datasets['test'].column_names:
        datasets = datasets.rename_column('label', 'labels')
    tokenized = datasets['test'].map(lambda x: tokenize_pair(x, tokenizer, max_length=max_length), batched=True, remove_columns=[])
    loader = DataLoader(tokenized, batch_size=32, shuffle=False, collate_fn=SiameseDataCollator())

    # Collect logits and compute metrics
    logits, labels, tlog1, tlog2, tlab1, tlab2 = collect_logits_labels(model, loader)
    # Argmax baseline
    argmax_pred = logits.argmax(axis=1)
    prob = torch.softmax(torch.tensor(logits) / max(temperature, 1e-6), dim=1)[:, 1].numpy()
    cal_pred = (prob >= threshold).astype(int)

    argmax_metrics = _metrics_from_preds(labels, prob, argmax_pred)
    cal_metrics = _metrics_from_preds(labels, prob, cal_pred)

    out = {
        'temperature': float(temperature),
        'threshold': float(threshold),
        'argmax': argmax_metrics,
        'calibrated': cal_metrics,
    }
    # Optional: topic adversary evaluation (lower is better invariance)
    try:
        if tlog1 is not None and tlab1 is not None:
            import numpy as np
            tp1 = tlog1.argmax(axis=1)
            tp2 = tlog2.argmax(axis=1)
            tacc1 = float((tp1 == tlab1).mean())
            tacc2 = float((tp2 == tlab2).mean())
            out['topic_adversary'] = {
                'acc_1': tacc1,
                'acc_2': tacc2,
                'mean_acc': float((tacc1 + tacc2) / 2.0),
            }
    except Exception:
        pass
    # Topic-sliced negative metrics (how often negatives are correctly rejected)
    try:
        if tlab1 is not None and tlab2 is not None:
            import numpy as np
            labels_np = np.asarray(labels)
            same_topic = (np.asarray(tlab1) == np.asarray(tlab2))
            neg_mask = (labels_np == 0)
            # Build helper to compute negative accuracy
            def _neg_acc(preds, mask):
                if mask.sum() == 0:
                    return float('nan')
                return float((preds[mask] == 0).mean())

            neg_same_argmax = _neg_acc(argmax_pred, neg_mask & same_topic)
            neg_diff_argmax = _neg_acc(argmax_pred, neg_mask & (~same_topic))

            neg_same_cal = _neg_acc(cal_pred, neg_mask & same_topic)
            neg_diff_cal = _neg_acc(cal_pred, neg_mask & (~same_topic))

            out['negatives_by_topic'] = {
                'counts': {
                    'negatives_total': int(neg_mask.sum()),
                    'same_topic_negatives': int((neg_mask & same_topic).sum()),
                    'diff_topic_negatives': int((neg_mask & (~same_topic)).sum()),
                },
                'argmax': {
                    'same_topic_negative_accuracy': float(neg_same_argmax),
                    'diff_topic_negative_accuracy': float(neg_diff_argmax),
                },
                'calibrated': {
                    'same_topic_negative_accuracy': float(neg_same_cal),
                    'diff_topic_negative_accuracy': float(neg_diff_cal),
                }
            }
    except Exception:
        pass

    print(json.dumps(out, indent=2))
    return out


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Evaluate contrastive model on test set using calibration.json')
    p.add_argument('--model', type=str, required=True, help='Path to model final dir (…/final)')
    p.add_argument('--data', type=str, default='data/processed', help='Path to processed datasets')
    p.add_argument('--calibration', type=str, default=None, help='Optional path to calibration.json')
    p.add_argument('--max-length', type=int, default=512, help='Tokenizer max_length for evaluation (default: 512)')
    args = p.parse_args()
    evaluate_contrastive(args.model, data_dir=args.data, calibration_path=args.calibration, max_length=args.max_length)
