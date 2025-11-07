"""
Calibrate a trained contrastive model by optimizing temperature and decision threshold
on the validation split, with support for accuracy/F1/balanced-accuracy and constraints.

Usage:
  python calibrate_contrastive.py \
      --model models/book_matcher_contrastive/final \
      --data data/processed \
      --calibrate-for accuracy \
      --target-recall 0.85 \
      --batch-size 128 --num-proc 4 --num-workers 2 --max-length 512
"""
import os
import json
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import load_from_disk

from train_contrastive import (
    tokenize_pair,
    SiameseDataCollator,
    _optimize_temperature,
    _search_best_threshold,
)
from inference_contrastive import ContrastiveBookMatcherInference


@torch.no_grad()
def _collect_validation_logits(model, dataloader) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    logits_list = []
    labels_list = []
    for batch in dataloader:
        # Move tensors to device
        tbatch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                tbatch[k] = v.to(device)
            else:
                tbatch[k] = v
        out = model(
            input_ids_1=tbatch['input_ids_1'],
            attention_mask_1=tbatch['attention_mask_1'],
            input_ids_2=tbatch['input_ids_2'],
            attention_mask_2=tbatch['attention_mask_2'],
            style_features_1=tbatch.get('style_features_1'),
            style_features_2=tbatch.get('style_features_2'),
        )
        logits_list.append(out['logits'].detach().cpu().numpy())
        labels_list.append(tbatch['labels'].detach().cpu().numpy())
    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return logits, labels


def calibrate_contrastive(
    model_dir: str,
    data_dir: str,
    calibrate_for: str = 'accuracy',
    target_acc: Optional[float] = None,
    target_recall: Optional[float] = 0.85,
    save_to: Optional[str] = None,
    batch_size: int = 64,
    num_proc: int = 1,
    num_workers: int = 0,
    max_length: int = 512,
):
    inf = ContrastiveBookMatcherInference(model_dir)
    tokenizer = inf.tokenizer
    model = inf.model

    datasets = load_from_disk(data_dir)
    if 'label' in datasets['validation'].column_names:
        datasets = datasets.rename_column('label', 'labels')

    tokenized_val = datasets['validation'].map(
        lambda x: tokenize_pair(x, tokenizer, max_length=max_length),
        batched=True,
        num_proc=num_proc if num_proc and num_proc > 1 else None,
        remove_columns=[],
    )

    collator = SiameseDataCollator()
    # Larger batches speed up forward pass on both CPU and GPU
    val_loader = DataLoader(
        tokenized_val,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        collate_fn=collator,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
    )

    logits, labels = _collect_validation_logits(model, val_loader)

    T = _optimize_temperature(logits, labels)
    thr, row = _search_best_threshold(
        logits,
        labels,
        temperature=T,
        metric=calibrate_for,
        target_acc=target_acc,
        target_recall=target_recall,
    )

    out_dir = save_to or os.path.abspath(os.path.join(model_dir, os.pardir))
    os.makedirs(out_dir, exist_ok=True)
    calib = {
        'temperature': float(T),
        'threshold': float(thr),
        'opt_metric': calibrate_for,
        'selection_metrics': row,
    }
    with open(os.path.join(out_dir, 'calibration.json'), 'w') as f:
        json.dump(calib, f)
    print(f"Saved calibration to {os.path.join(out_dir, 'calibration.json')}: {calib}")


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Calibrate contrastive model threshold and temperature on validation set')
    p.add_argument('--model', type=str, required=True, help='Path to model dir (e.g., models/.../final)')
    p.add_argument('--data', type=str, default='data/processed', help='Path to datasets directory')
    p.add_argument('--calibrate-for', type=str, default='accuracy', choices=['accuracy','balanced_accuracy','f1','f0.5','f2'], help='Metric to optimize threshold on')
    p.add_argument('--target-acc', type=float, default=None, help='Minimum accuracy constraint for threshold selection')
    p.add_argument('--target-recall', type=float, default=0.85, help='Minimum recall constraint for threshold selection')
    p.add_argument('--save-to', type=str, default=None, help='Directory to write calibration.json (default: parent of --model)')
    p.add_argument('--batch-size', type=int, default=64, help='Eval batch size for calibration forward pass')
    p.add_argument('--num-proc', type=int, default=1, help='Processes for HF map() tokenization')
    p.add_argument('--num-workers', type=int, default=0, help='DataLoader workers (0 = main process)')
    p.add_argument('--max-length', type=int, default=512, help='Tokenizer max_length for calibration (default: 512)')

    args = p.parse_args()
    calibrate_contrastive(
        model_dir=args.model,
        data_dir=args.data,
        calibrate_for=args.calibrate_for,
        target_acc=args.target_acc,
        target_recall=args.target_recall,
        save_to=args.save_to,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        num_workers=args.num_workers,
        max_length=args.max_length,
    )
