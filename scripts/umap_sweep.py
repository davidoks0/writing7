"""
Sweep UMAP hyperparameters for book embeddings and score layouts.

Metrics:
- Trustworthiness (sklearn) at multiple neighborhood sizes
- Neighborhood hit rate: overlap of k-NN graphs between original space and 2D

Outputs a CSV of results and optionally saves the best layout (CSV/PNG/HTML).

Example:
  python scripts/umap_sweep.py \
    --model-dir /vol/models/book_matcher_contrastive/final \
    --books-dir ./training \
    --grid-n-neighbors 15,50,100 \
    --grid-min-dist 0.01,0.05,0.1 \
    --remove-top-pcs 1 \
    --out-csv outputs/umap_sweep/results.csv \
    --save-best-prefix outputs/umap_sweep/best \
    --plotly-html outputs/umap_sweep/best.html
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

from inference_contrastive import ContrastiveBookMatcherInference
from style_map import (
    _prepare_book_chunks,
    embed_books_batched,
    _remove_top_pcs,
    reduce_to_nd,
    plot_map,
)


def _parse_list(s: str, cast):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def _read_text(path: Path, max_chars: int | None = None) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        if max_chars and len(txt) > max_chars:
            return txt[:max_chars]
        return txt
    except Exception:
        return ""


def _nhit_rate(X: np.ndarray, Y: np.ndarray, k: int = 15) -> float:
    k = max(1, min(k, X.shape[0] - 1))
    nnX = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X)
    nnY = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(Y)
    _, idxX = nnX.kneighbors(X)
    _, idxY = nnY.kneighbors(Y)
    idxX = idxX[:, 1:]
    idxY = idxY[:, 1:]
    hits = []
    for i in range(X.shape[0]):
        sX = set(idxX[i])
        sY = set(idxY[i])
        hits.append(len(sX & sY) / float(k))
    return float(np.mean(hits))


def main():
    ap = argparse.ArgumentParser(description="Sweep UMAP params for book embeddings and score layouts")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--books-dir", required=True)
    ap.add_argument("--max-books", type=int, default=400)
    ap.add_argument("--num-chunks", default="auto")
    ap.add_argument("--chunk-size", type=int, default=14)
    ap.add_argument("--overlap", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=None)
    ap.add_argument("--embed-batch-size", type=int, default=128)
    ap.add_argument("--use-projection", type=lambda x: str(x).lower() not in {"0","false","no"}, default=False)

    ap.add_argument("--remove-top-pcs", type=int, default=0)
    ap.add_argument("--n-components", type=int, choices=[2, 3], default=2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--grid-n-neighbors", type=str, default="15,50,100")
    ap.add_argument("--grid-min-dist", type=str, default="0.01,0.05,0.1")
    ap.add_argument("--densmap", action="store_true")
    ap.add_argument("--metric", choices=["cosine","euclidean"], default="cosine")
    ap.add_argument("--pca-dim", type=int, default=50)

    ap.add_argument("--eval-k", type=str, default="5,15")
    ap.add_argument("--score-by", choices=["k5","k15","avg"], default="k15")

    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--save-best-prefix", type=str, default=None)
    ap.add_argument("--plotly-html", type=str, default=None)

    args = ap.parse_args()

    # Parse lists
    try:
        num_chunks = int(args.num_chunks)  # type: ignore[arg-type]
    except Exception:
        num_chunks = "auto"
    grid_n = _parse_list(args.grid_n_neighbors, int)
    grid_d = _parse_list(args.grid_min_dist, float)
    eval_k = _parse_list(args.eval_k, int)

    # Load model
    model = ContrastiveBookMatcherInference(args.model_dir)

    # Collect books
    bdir = Path(args.books_dir)
    files = sorted([p for p in bdir.glob("**/*.txt") if p.is_file()])
    if args.max_books and len(files) > args.max_books:
        files = files[: args.max_books]
    if not files:
        raise SystemExit(f"No .txt files found under {bdir}")

    books: List[Tuple[str, List[str]]] = []
    for path in files:
        text = _read_text(path, max_chars=args.max_chars)
        chunks = _prepare_book_chunks(text, num_chunks=num_chunks, chunk_size=args.chunk_size, overlap=args.overlap)
        books.append((path.stem, chunks))

    labels, X = embed_books_batched(
        model,
        books,
        max_length=512,
        embed_batch_size=args.embed_batch_size,
        use_projection=args.use_projection,
        book_pool="trimmed_mean",
        trim_frac=0.2,
        book_topk=5,
    )
    if X.size == 0:
        raise SystemExit("No embeddings computed — check inputs")
    if args.remove_top_pcs and args.remove_top_pcs > 0:
        X = _remove_top_pcs(X, int(args.remove_top_pcs))

    # Sweep grid
    rows: List[dict] = []
    best_idx = None
    best_score = -1.0
    coords_best: np.ndarray | None = None
    cfg_best: dict | None = None
    for nn in grid_n:
        for md in grid_d:
            coords = reduce_to_nd(
                X,
                method="umap",
                n_components=args.n_components,
                pca_dim=args.pca_dim,
                perplexity=30.0,
                n_neighbors=int(nn),
                min_dist=float(md),
                metric=args.metric,
                densmap=bool(args.densmap),
                random_state=int(args.seed),
            )
            # Metrics
            metrics = {}
            for k in eval_k:
                tw = float(trustworthiness(X, coords, n_neighbors=int(k)))
                nhit = float(_nhit_rate(X, coords, k=int(k)))
                metrics[f"trust_k{k}"] = tw
                metrics[f"nhit_k{k}"] = nhit
            # Score aggregation
            if args.score_by == "avg":
                score = np.mean([metrics[f"trust_k{k}"] for k in eval_k])
            elif args.score_by == "k5":
                score = metrics.get("trust_k5", 0.0)
            else:
                score = metrics.get("trust_k15", 0.0)

            row = {
                "n_neighbors": int(nn),
                "min_dist": float(md),
                "densmap": int(bool(args.densmap)),
                **metrics,
            }
            rows.append(row)
            if score > best_score:
                best_score = score
                best_idx = len(rows) - 1
                coords_best = coords
                cfg_best = row

    # Write results CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r.keys()}, key=lambda x: (x not in {"n_neighbors","min_dist","densmap"}, x))
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print({"results": str(out_csv), "evaluated": len(rows), "best": cfg_best})

    # Save best layout
    if args.save-best-prefix and coords_best is not None:
        pref = Path(args.save-best-prefix)
        pref.parent.mkdir(parents=True, exist_ok=True)
        best_csv = pref.with_suffix(".csv")
        with best_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if args.n_components == 3:
                w.writerow(["label","x","y","z"])
                for i, lab in enumerate(labels):
                    x, y, z = coords_best[i,0], coords_best[i,1], coords_best[i,2]
                    w.writerow([lab, float(x), float(y), float(z)])
            else:
                w.writerow(["label","x","y"])
                for i, lab in enumerate(labels):
                    x, y = coords_best[i,0], coords_best[i,1]
                    w.writerow([lab, float(x), float(y)])
        # Optional PNG
        if args.n_components == 2:
            plot_map(coords_best, labels, pref.with_suffix(".png"), clusters=None)
        # Optional Plotly HTML
        if args.plotly_html:
            try:
                import plotly.express as px  # type: ignore
                if args.n_components == 3:
                    fig = px.scatter_3d(x=coords_best[:,0], y=coords_best[:,1], z=coords_best[:,2], hover_name=labels, height=720, width=960)
                else:
                    fig = px.scatter(x=coords_best[:,0], y=coords_best[:,1], hover_name=labels, height=720, width=960)
                fig.update_traces(marker=dict(size=3))
                fig.write_html(args.plotly_html, include_plotlyjs='cdn', full_html=True)
            except Exception as e:
                print(f"Plotly not available or failed to save HTML: {e}")


if __name__ == "__main__":
    main()

