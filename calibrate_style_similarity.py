"""
Calibrate style similarity (cosine) to a [0,1] score using a labeled dev set.

Input CSV format (header required):
  Required: text1,text2,label
  Optional (used for grouped CV): group,book1,book2,author,provider,chunk_count,topic1,topic2
  - If a `group` column is present, it is used for GroupKFold.
  - Else, if `book1` (and optionally `book2`) is present, `group=book1` is used.
  - Otherwise a standard StratifiedKFold is used.

Usage:
  python calibrate_style_similarity.py \
    --model models/book_matcher_contrastive/final \
    --pairs path/to/pairs.csv \
    --method logistic \
    --save-to models/style_calibration.json

The file is saved next to your model by default as ../style_calibration.json.
"""
from __future__ import annotations

import os
import json
import csv
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
import matplotlib.pyplot as plt

from inference_contrastive import ContrastiveBookMatcherInference


def _ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15, strategy: str = 'quantile') -> float:
    """Expected Calibration Error using absolute gap, with fixed or quantile bins."""
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = probs.size
    if n == 0:
        return float('nan')
    if strategy == 'quantile':
        # Equal-mass bins by quantiles (guard duplicates)
        qs = np.linspace(0.0, 1.0, num=n_bins + 1)
        edges = np.quantile(probs, qs, method='linear')
        # De-duplicate edges to avoid empty bins
        edges = np.unique(edges)
        if edges.size <= 1:
            return 0.0
    else:
        edges = np.linspace(0.0, 1.0, num=n_bins + 1)
    ece = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        m = int(mask.sum())
        if m == 0:
            continue
        conf = float(probs[mask].mean())
        acc = float(y[mask].mean())
        ece += (m / n) * abs(acc - conf)
    return float(ece)


def _brier(probs: np.ndarray, y: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean((probs - y) ** 2))


def _load_pairs_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            t1 = r.get('text1') or r.get('t1') or ''
            t2 = r.get('text2') or r.get('t2') or ''
            lab = r.get('label') or r.get('y') or r.get('style_match') or '0'
            group = (
                r.get('group')
                or r.get('book1')
                or r.get('author')
                or ''
            )
            # Keep optional metadata if present
            meta = {
                'book1': r.get('book1') or '',
                'book2': r.get('book2') or '',
                'provider': r.get('provider') or '',
                'topic1': r.get('topic1'),
                'topic2': r.get('topic2'),
                'same_topic': r.get('same_topic'),
                'neg_type': r.get('neg_type'),
            }
            try:
                y = int(lab)
            except Exception:
                y = 0
            rows.append({
                'text1': t1,
                'text2': t2,
                'label': 1 if y else 0,
                'group': group,
                **meta,
            })
    if not rows:
        raise ValueError(f"No rows loaded from {path}. Expect header with columns text1,text2,label")
    return rows


def _fit_logistic(cos: np.ndarray, y: np.ndarray) -> dict:
    X = cos.reshape(-1, 1)
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(X, y)
    coef = float(clf.coef_.ravel()[0])
    intercept = float(clf.intercept_.ravel()[0])
    return {
        'method': 'logistic',
        'coef': coef,
        'intercept': intercept,
    }


def _fit_isotonic(cos: np.ndarray, y: np.ndarray) -> dict:
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(cos, y)
    # Store thresholds and values for lightweight mapping
    return {
        'method': 'isotonic',
        'x_thresholds': ir.X_thresholds_.tolist(),
        'y_values': ir.y_thresholds_.tolist(),
    }


def _apply_calibration(mapping: dict, x: np.ndarray) -> np.ndarray:
    """Apply a fitted mapping dict to cosine array to get probabilities."""
    method = mapping.get('method')
    x = np.asarray(x, dtype=np.float64)
    if method == 'logistic':
        a = float(mapping.get('coef', 0.0))
        b = float(mapping.get('intercept', 0.0))
        z = a * x + b
        # stable sigmoid
        probs = np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z)),
        )
        return probs.astype(np.float64)
    if method == 'isotonic':
        xs = mapping.get('x_thresholds') or []
        ys = mapping.get('y_values') or []
        if not xs or not ys or len(xs) != len(ys):
            return np.zeros_like(x) + np.nan
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        # Vectorized piecewise-linear interpolation with clipping
        idx = np.searchsorted(xs, x, side='left')
        idx = np.clip(idx, 1, len(xs) - 1)
        x0 = xs[idx - 1]
        x1 = xs[idx]
        y0 = ys[idx - 1]
        y1 = ys[idx]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (x - x0) / (x1 - x0)
        t = np.where((x1 == x0), 0.0, t)
        y = y0 + t * (y1 - y0)
        # clip to [0,1]
        return np.clip(y, 0.0, 1.0)
    # unknown
    return np.zeros_like(x) + np.nan


def _cv_select(
    cos: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    candidates: List[str],
    metric: str = 'brier',
    n_splits: int = 5,
) -> Tuple[str, Dict[str, float]]:
    """Cross-validated model selection over candidate calibrators.

    Returns (best_method, scores_by_method)."""
    y = np.asarray(y, dtype=np.int32)
    cos = np.asarray(cos, dtype=np.float64)
    # Build splitter
    if groups is not None and groups.size and (len(np.unique(groups)) >= n_splits):
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(cos, y, groups)
    else:
        # Stratify by label to balance classes
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = splitter.split(cos, y)

    def _score(method: str) -> float:
        scores = []
        for train_idx, val_idx in splits:
            xtr, ytr = cos[train_idx], y[train_idx]
            xva, yva = cos[val_idx], y[val_idx]
            if method == 'logistic':
                mapping = _fit_logistic(xtr, ytr)
            elif method == 'isotonic':
                mapping = _fit_isotonic(xtr, ytr)
            else:
                raise ValueError('unknown method')
            p = _apply_calibration(mapping, xva)
            if metric == 'ece':
                m = _ece(p, yva, n_bins=15, strategy='quantile')
            else:
                m = _brier(p, yva)
            scores.append(float(m))
        return float(np.mean(scores)) if scores else float('inf')

    scores_by_method: Dict[str, float] = {}
    for m in candidates:
        # Need fresh splits iterator each time (GroupKFold yields must be re-created)
        if groups is not None and groups.size and (len(np.unique(groups)) >= n_splits):
            splitter = GroupKFold(n_splits=n_splits)
            splits = splitter.split(cos, y, groups)
        else:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = splitter.split(cos, y)
        scores_by_method[m] = _score(m)

    # Pick best (lower is better)
    best = min(scores_by_method.items(), key=lambda kv: kv[1])[0]
    return best, scores_by_method


def calibrate_style_similarity(
    model_dir: str,
    pairs_csv: str,
    method: str = 'logistic',
    metric: str = 'brier',
    n_splits: int = 5,
    group_col: Optional[str] = None,
    save_to: str | None = None,
    num_chunks = 'auto',
    chunk_size: int = 14,
    overlap: int = 4,
    aggregate: str = 'mean',
    topk: int = 5,
    max_length: int = 512,
):
    pairs = _load_pairs_csv(pairs_csv)
    infer = ContrastiveBookMatcherInference(model_dir)

    cos_vals: List[float] = []
    labels: List[int] = []
    groups: List[str] = []
    for row in pairs:
        t1, t2, lab = row['text1'], row['text2'], int(row['label'])
        sim = infer.style_similarity(
            t1, t2,
            num_chunks=num_chunks,
            chunk_size=chunk_size,
            overlap=overlap,
            aggregate=aggregate,
            topk=topk,
            max_length=max_length,
        )
        cos = float(sim.get('cosine', float('nan')))
        if not np.isfinite(cos):
            continue
        cos_vals.append(cos)
        labels.append(int(lab))
        # Determine group for CV selection if available
        g = ''
        if group_col and row.get(group_col):
            g = str(row.get(group_col) or '')
        else:
            # autodetect
            g = str(row.get('group') or row.get('book1') or '')
        groups.append(g)

    if not cos_vals:
        raise ValueError("No finite cosine values computed; check your inputs.")

    cos_arr = np.asarray(cos_vals, dtype=np.float64)
    y_arr = np.asarray(labels, dtype=np.int32)
    groups_arr = np.asarray(groups, dtype=object)

    # Auto-select calibrator if requested
    selected_method = method
    scores_by_method: Dict[str, float] | None = None
    if method == 'auto':
        candidates = ['logistic', 'isotonic']
        best, scores = _cv_select(cos_arr, y_arr, groups_arr if groups_arr.size else None, candidates, metric=metric, n_splits=n_splits)
        selected_method = best
        scores_by_method = scores
    if selected_method == 'logistic':
        calib = _fit_logistic(cos_arr, y_arr)
    elif selected_method == 'isotonic':
        calib = _fit_isotonic(cos_arr, y_arr)
    else:
        raise ValueError("method must be 'logistic', 'isotonic', or 'auto'")

    meta = {
        'n_samples': int(len(cos_vals)),
        'num_chunks': num_chunks,
        'chunk_size': int(chunk_size),
        'overlap': int(overlap),
        'aggregate': aggregate,
        'topk': int(topk),
        'max_length': int(max_length),
        'model_dir': model_dir,
        'selection_metric': metric,
        'n_splits': int(n_splits),
        'method_requested': method,
    }
    payload = {
        'style_calibration': calib,
        'meta': meta,
    }
    if scores_by_method is not None:
        payload['selection'] = {
            'scores': scores_by_method,
            'chosen': selected_method,
        }

    if save_to is None:
        # Default next to model, as sibling of final/ directory
        save_to = os.path.normpath(os.path.join(model_dir, '..', 'style_calibration.json'))

    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    with open(save_to, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    # Compute fitted probabilities for reporting/plots
    probs = _apply_calibration(calib, cos_arr)
    report = {
        'method': calib.get('method'),
        'n_samples': int(meta['n_samples']),
        'metric': metric,
        'cv_scores': (scores_by_method or {}),
        'brier': _brier(probs, y_arr),
        'ece': _ece(probs, y_arr, n_bins=15, strategy='quantile'),
    }
    # Save a reliability plot
    try:
        fig, ax = plt.subplots(figsize=(5, 5))
        # Quantile bins
        n_bins = 15
        qs = np.linspace(0.0, 1.0, num=n_bins + 1)
        edges = np.unique(np.quantile(probs, qs, method='linear'))
        xs, ys, ns = [], [], []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            mask = (probs >= lo) & (probs <= hi if i == len(edges) - 2 else probs < hi)
            if not mask.any():
                continue
            xs.append(float(probs[mask].mean()))
            ys.append(float(y_arr[mask].mean()))
            ns.append(int(mask.sum()))
        ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
        ax.plot(xs, ys, marker='o')
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Empirical frequency')
        ax.set_title('Reliability (global)')
        ax.grid(True, alpha=0.2)
        plot_path = os.path.join(os.path.dirname(save_to), 'style_calibration_reliability.png')
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        report['plot_path'] = plot_path
    except Exception as e:
        report['plot_error'] = str(e)
    # Save report JSON
    report_path = os.path.join(os.path.dirname(save_to), 'calibration_report.json')
    try:
        with open(report_path, 'w', encoding='utf-8') as rf:
            json.dump(report, rf)
    except Exception:
        pass
    print({'saved_to': save_to, 'report': report_path, 'method': calib.get('method'), 'n_samples': meta['n_samples']})
    return save_to


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Calibrate style similarity (cosine) to [0,1] score')
    p.add_argument('--model', type=str, required=True, help='Path to contrastive model final dir (…/final)')
    p.add_argument('--pairs', type=str, required=True, help='CSV file with columns text1,text2,label')
    p.add_argument('--method', type=str, default='logistic', choices=['logistic','isotonic','auto'])
    p.add_argument('--metric', type=str, default='brier', choices=['brier','ece'], help='Model selection metric when --method auto')
    p.add_argument('--n-splits', type=int, default=5, help='CV splits when using --method auto')
    p.add_argument('--group-col', type=str, default=None, help='Optional CSV column to use for GroupKFold')
    p.add_argument('--save-to', type=str, default=None, help='Where to save style_calibration.json (default: sibling of model dir)')
    p.add_argument('--num-chunks', default='auto', help="Chunks per text: integer or 'auto'")
    p.add_argument('--chunk-size', type=int, default=14)
    p.add_argument('--overlap', type=int, default=4)
    p.add_argument('--aggregate', type=str, default='mean', choices=['mean','topk_mean'])
    p.add_argument('--topk', type=int, default=5)
    p.add_argument('--max-length', type=int, default=512)
    args = p.parse_args()

    # Parse num_chunks which can be 'auto' or int
    try:
        nc = int(args.num_chunks)
    except Exception:
        nc = 'auto'

    calibrate_style_similarity(
        model_dir=args.model,
        pairs_csv=args.pairs,
        method=args.method,
        metric=args.metric,
        n_splits=int(args.n_splits),
        group_col=args.group_col,
        save_to=args.save_to,
        num_chunks=nc,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        aggregate=args.aggregate,
        topk=args.topk,
        max_length=args.max_length,
    )
