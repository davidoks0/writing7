#!/usr/bin/env python3
"""
Compute reference distributions (percentiles) for style similarity scores
from a labeled pairs CSV. Produces a JSON file with global and per-domain
percentiles for positives (same author) and negatives (different author).

Input CSV should follow the schema emitted by scripts/build_mixed_pairs.py:
  text1,text2,label,group,domain,author1,author2,doc1,doc2,topic1,topic2

Usage:
  python eval/build_reference_distributions.py \
    --pairs data/mixed_pairs.csv \
    --model models/book_matcher_contrastive/final \
    --out eval/reference_distributions.json

Notes:
- If style_calibration.json exists next to the model, calibrated scores are used.
- Otherwise, a naive 0..1 mapping ((cos+1)/2) is used (still useful, but less interpretable).
- Chunking/aggregation parameters should match your runtime defaults.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List

import numpy as np

from inference_contrastive import ContrastiveBookMatcherInference


def _score_pairs(
    model_dir: str,
    pairs_csv: Path,
    *,
    num_chunks='auto',
    chunk_size=14,
    overlap=4,
    aggregate='mean',
    topk=5,
    max_length=512,
):
    infer = ContrastiveBookMatcherInference(model_dir)

    def _score(t1: str, t2: str) -> Dict[str, float | None]:
        res = infer.style_similarity(
            t1, t2,
            num_chunks=num_chunks,
            chunk_size=chunk_size,
            overlap=overlap,
            aggregate=aggregate,
            topk=topk,
            max_length=max_length,
        )
        cos = float(res.get('cosine', float('nan')))
        cal = res.get('calibrated')
        s01 = (cos + 1.0) / 2.0 if math.isfinite(cos) else float('nan')
        return {
            'cosine': cos,
            'calibrated': float(cal) if cal is not None else None,
            'score_0_1': s01,
        }

    records: List[Dict] = []
    with open(pairs_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            t1 = r.get('text1') or ''
            t2 = r.get('text2') or ''
            lab = int(r.get('label') or 0)
            dom = r.get('domain') or 'unknown'
            sc = _score(t1, t2)
            records.append({'label': lab, 'domain': dom, **sc})
    return records


def _percentiles(values: List[float], qs=(1,5,10,25,50,75,90,95,99)) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    out: Dict[str, float] = {}
    for q in qs:
        try:
            out[f'p{q}'] = float(np.nanquantile(arr, q/100.0, method='linear'))
        except Exception:
            out[f'p{q}'] = float('nan')
    out['mean'] = float(np.nanmean(arr))
    out['median'] = float(np.nanmedian(arr))
    return out


def build_reference(model_dir: str, pairs_csv: Path, out_path: Path, **style_kwargs) -> Path:
    recs = _score_pairs(model_dir, pairs_csv, **style_kwargs)
    # Prefer calibrated score if available; fallback to score_0_1
    def _get_sc(r):
        return (r.get('calibrated') if r.get('calibrated') is not None else r.get('score_0_1'))

    pos = [float(_get_sc(r)) for r in recs if int(r.get('label') or 0) == 1 and _get_sc(r) is not None]
    neg = [float(_get_sc(r)) for r in recs if int(r.get('label') or 0) == 0 and _get_sc(r) is not None]

    by_dom_pos: Dict[str, List[float]] = defaultdict(list)
    by_dom_neg: Dict[str, List[float]] = defaultdict(list)
    for r in recs:
        sc = _get_sc(r)
        if sc is None:
            continue
        dom = str(r.get('domain') or 'unknown')
        if int(r.get('label') or 0) == 1:
            by_dom_pos[dom].append(float(sc))
        else:
            by_dom_neg[dom].append(float(sc))

    out = {
        'meta': {
            'model_dir': model_dir,
            'pairs_csv': str(pairs_csv),
            'n_pairs': len(recs),
        },
        'global': {
            'positive': _percentiles(pos),
            'negative': _percentiles(neg),
        },
        'by_domain': {},
    }
    domains = set(list(by_dom_pos.keys()) + list(by_dom_neg.keys()))
    for d in sorted(domains):
        out['by_domain'][d] = {
            'positive': _percentiles(by_dom_pos.get(d, [])),
            'negative': _percentiles(by_dom_neg.get(d, [])),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print({'saved_to': str(out_path), 'n_pairs': len(recs)})
    return out_path


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Build reference percentile distributions from a labeled pairs CSV')
    ap.add_argument('--pairs', type=Path, required=True)
    ap.add_argument('--model', type=str, required=True, help='Path to contrastive model final dir (…/final)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--num-chunks', default='auto')
    ap.add_argument('--chunk-size', type=int, default=14)
    ap.add_argument('--overlap', type=int, default=4)
    ap.add_argument('--aggregate', type=str, default='mean', choices=['mean','topk_mean'])
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--max-length', type=int, default=512)
    args = ap.parse_args()

    try:
        nc = int(args.num_chunks)
    except Exception:
        nc = 'auto'

    build_reference(
        args.model,
        args.pairs,
        args.out,
        num_chunks=nc,
        chunk_size=int(args.chunk_size),
        overlap=int(args.overlap),
        aggregate=args.aggregate,
        topk=int(args.topk),
        max_length=int(args.max_length),
    )

