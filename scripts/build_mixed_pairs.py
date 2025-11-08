#!/usr/bin/env python3
"""
Build a mixed-domain style calibration CSV from local text corpora.

Directory layout (simple and reproducible):

  corpora_root/
    <domain>/
      <author_id>/
        doc1.txt
        doc2.txt
        ...

Examples of <domain>:
  archaic_pg, legal, government, technical, fiction, stackoverflow

What this script does:
- Loads documents grouped by domain/author.
- Splits each document into sentence chunks (default 14 sentences, overlap 4).
- Builds labeled pairs:
  - Positives (label=1): two chunks from different docs by the same author.
  - Negatives (label=0): two chunks from different authors within the same domain
    (topic-matched heuristics optional; see --topic-keywords).
- Balances per-author sampling and caps to avoid dominance.
- Writes a CSV suitable for calibrate_style_similarity.py:
    text1,text2,label,group,domain,author1,author2,doc1,doc2,topic1,topic2

Notes:
- This script is intentionally light on dependencies and online fetching.
- If you have explicit topics per document, add a file topics.json under each author dir:
    {
      "doc1.txt": "sailing",
      "doc2.txt": "love"
    }
  The values will appear in topic1/topic2 for better topic-matching.

Usage:
  python scripts/build_mixed_pairs.py \
    --root corpora_root \
    --out data/mixed_pairs.csv \
    --pos-per-author 10 --neg-per-author 20 \
    --chunk-size 14 --overlap 4 --min-chars 200

You can run it multiple times with different corpora roots and append to the same CSV
with --append to grow your calibration set iteratively.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _split_sentences_simple(text: str) -> List[str]:
    # Similar to helpers used elsewhere in the repo; avoids heavy deps
    sentences: List[str] = []
    pattern = r'[.!?]+["\')]*\s+(?=[A-Z])'
    parts = re.split(pattern, text)
    for part in parts:
        part = part.strip()
        if len(part) > 20:
            sentences.append(part)
    if not sentences and len(text.strip()) > 20:
        sentences.append(text.strip())
    return sentences


def _make_chunks(sentences: List[str], *, chunk_size: int = 14, overlap: int = 4, min_chars: int = 200) -> List[str]:
    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(sentences), step):
        chunk = ' '.join(sentences[i : i + chunk_size]).strip()
        if len(chunk) >= int(min_chars):
            chunks.append(chunk)
    if not chunks and sentences:
        chunk = ' '.join(sentences[:chunk_size]).strip()
        if len(chunk) >= int(min_chars // 2):
            chunks.append(chunk)
    return chunks


def _read_topics_json(author_dir: Path) -> Dict[str, str]:
    p = author_dir / 'topics.json'
    if p.exists():
        try:
            with open(p, 'r', encoding='utf-8') as f:
                d = json.load(f)
            if isinstance(d, dict):
                return {str(k): str(v) for k, v in d.items()}
        except Exception:
            pass
    return {}


@dataclass
class Doc:
    domain: str
    author: str
    doc_path: Path
    topic: str = ''
    chunks: List[str] = None  # type: ignore


def _load_docs(root: Path, *, chunk_size: int, overlap: int, min_chars: int, max_docs_per_author: int, rng: random.Random) -> Dict[str, Dict[str, List[Doc]]]:
    """Return mapping: domain -> author -> list[Doc]."""
    data: Dict[str, Dict[str, List[Doc]]] = {}
    for domain_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        domain = domain_dir.name
        by_author: Dict[str, List[Doc]] = {}
        for author_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
            author = author_dir.name
            topics_map = _read_topics_json(author_dir)
            docs: List[Doc] = []
            txts = sorted(author_dir.glob('*.txt'))
            if max_docs_per_author > 0 and len(txts) > max_docs_per_author:
                txts = rng.sample(txts, max_docs_per_author)
            for fp in txts:
                try:
                    raw = fp.read_text(encoding='utf-8', errors='ignore')
                    sents = _split_sentences_simple(raw)
                    chunks = _make_chunks(sents, chunk_size=chunk_size, overlap=overlap, min_chars=min_chars)
                    if not chunks:
                        continue
                    topic = topics_map.get(fp.name, '')
                    docs.append(Doc(domain=domain, author=author, doc_path=fp, topic=topic, chunks=chunks))
                except Exception:
                    continue
            if docs:
                by_author[author] = docs
        if by_author:
            data[domain] = by_author
    return data


def _topic_bucket(text: str, topic_keywords: Dict[str, List[str]]) -> str:
    tl = text.lower()
    best = ''
    best_score = 0
    for name, keys in topic_keywords.items():
        score = sum(tl.count(k) for k in keys)
        if score > best_score:
            best, best_score = name, score
    return best


def build_pairs(
    data: Dict[str, Dict[str, List[Doc]]],
    *,
    pos_per_author: int,
    neg_per_author: int,
    rng: random.Random,
    topic_keywords: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    topic_keywords = topic_keywords or {
        'religious': ['god','church','prayer','scripture','lord','faith','holy','sacred','divine'],
        'historical': ['prince','king','castle','knight','lord','duke','empire','queen','court'],
        'adventure': ['captain','ship','sea','voyage','island','expedition','jungle','desert','treasure'],
        'romance': ['love','heart','soul','passion','beloved','darling','kiss','romance','marriage'],
        'legal': ['court','appeal','plaintiff','defendant','precedent','statute','opinion','justice','jury'],
        'technical': ['algorithm','data','model','experiment','theorem','equation','system','network','performance'],
        'government': ['agency','policy','regulation','federal','program','initiative','public','official','department'],
        'fiction': ['dialogue','whispered','suddenly','shadow','door','eyes','voice','silence','room'],
    }
    # Iterate per domain
    for domain, by_author in data.items():
        authors = list(by_author.keys())
        if len(authors) < 2:
            continue
        for author, docs in by_author.items():
            # Positives: same author, different docs
            if len(docs) >= 2:
                for _ in range(pos_per_author):
                    d1, d2 = rng.sample(docs, 2)
                    c1 = rng.choice(d1.chunks)
                    c2 = rng.choice(d2.chunks)
                    t1 = d1.topic or _topic_bucket(c1, topic_keywords)
                    t2 = d2.topic or _topic_bucket(c2, topic_keywords)
                    pairs.append({
                        'text1': c1,
                        'text2': c2,
                        'label': '1',
                        'group': author,
                        'domain': domain,
                        'author1': author,
                        'author2': author,
                        'doc1': d1.doc_path.name,
                        'doc2': d2.doc_path.name,
                        'topic1': t1,
                        'topic2': t2,
                    })
            # Negatives: different authors within domain (prefer topic-matched)
            others = [a for a in authors if a != author and by_author.get(a)]
            if not others:
                continue
            for _ in range(neg_per_author):
                a2 = rng.choice(others)
                d1 = rng.choice(docs)
                d2 = rng.choice(by_author[a2])
                c1 = rng.choice(d1.chunks)
                # Try topic match by sampling a few candidates from a2
                t1 = d1.topic or _topic_bucket(c1, topic_keywords)
                c2_candidates = rng.sample(d2.chunks, min(6, len(d2.chunks)))
                scored: List[Tuple[int, str]] = []
                for c2 in c2_candidates:
                    t2c = d2.topic or _topic_bucket(c2, topic_keywords)
                    score = 1 if t2c == t1 and t1 != '' else 0
                    scored.append((score, c2))
                scored.sort(key=lambda x: x[0], reverse=True)
                c2 = scored[0][1] if scored else rng.choice(d2.chunks)
                t2 = d2.topic or _topic_bucket(c2, topic_keywords)
                pairs.append({
                    'text1': c1,
                    'text2': c2,
                    'label': '0',
                    'group': author,  # group for CV can be author of text1
                    'domain': domain,
                    'author1': author,
                    'author2': a2,
                    'doc1': d1.doc_path.name,
                    'doc2': d2.doc_path.name,
                    'topic1': t1,
                    'topic2': t2,
                })
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description='Build mixed-domain style calibration CSV from corpora directory tree')
    ap.add_argument('--root', type=Path, required=True, help='Root directory (domain/author/*.txt)')
    ap.add_argument('--out', type=Path, required=True, help='Output CSV path')
    ap.add_argument('--append', action='store_true', help='Append to existing CSV instead of overwriting')
    ap.add_argument('--pos-per-author', type=int, default=10)
    ap.add_argument('--neg-per-author', type=int, default=20)
    ap.add_argument('--max-docs-per-author', type=int, default=10)
    ap.add_argument('--chunk-size', type=int, default=14)
    ap.add_argument('--overlap', type=int, default=4)
    ap.add_argument('--min-chars', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--topic-keywords', type=Path, default=None, help='Optional JSON with topic -> [keywords] mapping')
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    topic_keywords = None
    if args.topic_keywords and args.topic_keywords.exists():
        try:
            topic_keywords = json.loads(args.topic_keywords.read_text(encoding='utf-8'))
        except Exception:
            topic_keywords = None

    data = _load_docs(
        args.root,
        chunk_size=int(args.chunk_size),
        overlap=int(args.overlap),
        min_chars=int(args.min_chars),
        max_docs_per_author=int(args.max_docs_per_author),
        rng=rng,
    )
    if not data:
        raise SystemExit(f'No documents found under {args.root}. Expect domain/author/*.txt')

    pairs = build_pairs(
        data,
        pos_per_author=int(args.pos_per_author),
        neg_per_author=int(args.neg_per_author),
        rng=rng,
        topic_keywords=topic_keywords,
    )
    if not pairs:
        raise SystemExit('No pairs constructed. Ensure at least some authors have >=2 docs.')

    # Write CSV
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = True
    if args.append and out.exists():
        write_header = False
    with open(out, 'a' if args.append else 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['text1','text2','label','group','domain','author1','author2','doc1','doc2','topic1','topic2']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for row in pairs:
            w.writerow(row)
    print(f'Wrote {len(pairs)} pairs -> {out}')


if __name__ == '__main__':
    main()

