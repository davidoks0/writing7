"""
Create a 2D map of book styles using the trained contrastive encoder.

For each book (.txt file), we:
- Split into sentence chunks (configurable size/overlap)
- Embed each chunk with the contrastive encoder
- Average chunk embeddings to get a per-book vector (then L2-normalize)
- Reduce embeddings to 2D via PCA or t-SNE
- Save coordinates to CSV and a PNG scatter plot

Usage example:
  python style_map.py \
    --model-dir /vol/models/book_matcher_contrastive/final \
    --books-dir ./training \
    --max-books 300 \
    --method tsne --perplexity 30 \
    --out-prefix outputs/style_map

Notes:
- The script calls the inference helper from `inference_contrastive.py` and uses the same encoder/pooling/projection stack
- If your model directory includes `style_calibration.json`, it will be ignored here (we use raw embeddings)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from inference_contrastive import ContrastiveBookMatcherInference


def _read_text(path: Path, max_chars: int | None = None) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        if max_chars and len(txt) > max_chars:
            return txt[:max_chars]
        return txt
    except Exception:
        return ""


def _split_sentences_simple(text: str) -> List[str]:
    # Keep consistent with inference helper
    import re
    sentences: List[str] = []
    pattern = r'[.!?]+["\'\)]*\s+(?=[A-Z])'
    parts = re.split(pattern, text)
    for part in parts:
        part = part.strip()
        if len(part) > 20:
            sentences.append(part)
    if not sentences and len(text.strip()) > 20:
        sentences.append(text.strip())
    return sentences


def _make_chunks_from_sentences(sentences: List[str], chunk_size: int = 14, overlap: int = 4, min_chars: int = 200) -> List[str]:
    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(sentences), step):
        chunk = ' '.join(sentences[i:i + chunk_size]).strip()
        if len(chunk) >= int(min_chars):
            chunks.append(chunk)
    if not chunks and sentences:
        chunk = ' '.join(sentences[:chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _prepare_book_chunks(
    text: str,
    *,
    num_chunks: int | str = "auto",
    chunk_size: int = 14,
    overlap: int = 4,
) -> List[str]:
    """Prepare chunk texts for a single book according to settings."""
    text = (text or "").strip()
    if not text:
        return []
    sents = _split_sentences_simple(text)
    chunks = _make_chunks_from_sentences(sents, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        chunks = [text]
    if isinstance(num_chunks, int) and num_chunks > 0:
        chunks = chunks[:num_chunks]
    else:
        chunks = chunks[:8]  # heuristic cap
    return chunks


def embed_books_batched(
    model: ContrastiveBookMatcherInference,
    books: List[Tuple[str, List[str]]],
    *,
    max_length: int = 512,
    embed_batch_size: int = 64,
    auto_tune_embed_batch: bool = False,
    use_projection: bool = False,
    book_pool: str = "trimmed_mean",  # 'mean', 'trimmed_mean', 'median', 'topk_mean'
    trim_frac: float = 0.2,
    book_topk: int = 5,
) -> Tuple[List[str], np.ndarray]:
    """Embed many books by batching chunks across books with robust per-book pooling.

    Args:
        model: inference model
        books: list of (label, chunks) where chunks is a list of chunk texts
        max_length: tokenizer max_length per chunk
        embed_batch_size: number of chunk texts per forward pass
        use_projection: whether to apply the model's projection head during embedding
        book_pool: pooling strategy across chunk embeddings per book
        trim_frac: fraction to trim at both tails for trimmed_mean (0.2 keeps middle 60%)
        book_topk: k for topk_mean (select top-k chunks closest to provisional mean)

    Returns:
        labels, book_embeddings (N x D, L2-normalized)
    """
    # Build flat list of all chunk texts and an owner index per chunk
    flat_texts: List[str] = []
    owners: List[int] = []
    labels: List[str] = []
    for bi, (lab, chs) in enumerate(books):
        if not chs:
            continue
        labels.append(lab)
        for c in chs:
            flat_texts.append(c)
            owners.append(bi)
    if not flat_texts:
        return [], np.zeros((0, 1), dtype=np.float32)

    n_books = len(books)
    per_book: List[List[np.ndarray]] = [list() for _ in range(n_books)]

    # Process in micro-batches with optional adaptive batch sizing
    i = 0
    bs = int(embed_batch_size)
    warned = False
    while i < len(flat_texts):
        j_end = min(i + bs, len(flat_texts))
        batch = flat_texts[i:j_end]
        try:
            Z = model._embed_texts(batch, max_length=max_length, use_projection=use_projection)  # (B, D)
        except RuntimeError as e:
            msg = str(e).lower()
            if auto_tune_embed_batch and ("out of memory" in msg or "cuda" in msg):
                # Back off batch size
                if not warned:
                    print("Auto-tuning embed batch size due to OOM...")
                    warned = True
                new_bs = max(1, bs // 2)
                if new_bs == bs:
                    raise
                bs = new_bs
                try:
                    import torch  # type: ignore
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                continue  # retry with smaller bs
            else:
                raise
        if Z.size == 0:
            i = j_end
            continue
        for j in range(Z.shape[0]):
            owner = owners[i + j]
            if owner < n_books:
                per_book[owner].append(Z[j])
        i = j_end

    def _pool(chunks: List[np.ndarray]) -> np.ndarray | None:
        if not chunks:
            return None
        A = np.stack(chunks, axis=0)
        if A.shape[0] == 1:
            v = A[0]
            return v / (np.linalg.norm(v) + 1e-9)
        if book_pool == "mean":
            v = A.mean(axis=0)
        elif book_pool == "median":
            v = np.median(A, axis=0)
        elif book_pool == "topk_mean":
            mu = A.mean(axis=0)
            mu = mu / (np.linalg.norm(mu) + 1e-9)
            sims = (A @ mu)  # cosine since rows are L2-normalized
            k = max(1, min(book_topk, A.shape[0]))
            idx = np.argpartition(-sims, k - 1)[:k]
            v = A[idx].mean(axis=0)
        else:  # trimmed_mean
            mu = A.mean(axis=0)
            mu = mu / (np.linalg.norm(mu) + 1e-9)
            sims = (A @ mu)
            # Keep middle fraction around the center (highest sims)
            keep = max(1, int(round((1.0 - float(trim_frac)) * A.shape[0])))
            idx = np.argpartition(-sims, keep - 1)[:keep]
            v = A[idx].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-9)
        return v.astype(np.float32)

    pooled: List[np.ndarray] = []
    keep_labels: List[str] = []
    for bi, (lab, _) in enumerate(books):
        v = _pool(per_book[bi])
        if v is not None:
            pooled.append(v)
            keep_labels.append(lab)
    if not pooled:
        return [], np.zeros((0, 1), dtype=np.float32)
    X = np.stack(pooled, axis=0)
    return keep_labels, X


def _remove_top_pcs(X: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or X.shape[0] < 2:
        return X
    Xc = X - X.mean(axis=0, keepdims=True)
    kk = min(k, min(Xc.shape) - 1)
    if kk <= 0:
        return Xc
    pca = PCA(n_components=kk, random_state=42)
    pca.fit(Xc)
    V = pca.components_  # (k, d)
    # subtract projection onto top-k components
    proj = (Xc @ V.T) @ V
    return Xc - proj


def _knn_outlier_scores(X: np.ndarray, k: int = 10, metric: str = "cosine") -> np.ndarray:
    """Compute average distance to k nearest neighbors as an outlier score.

    Higher scores indicate more isolated points.
    """
    k = max(1, min(k, max(1, X.shape[0] - 1)))
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    # skip self (first column)
    if dists.shape[1] > 1:
        d = dists[:, 1:].mean(axis=1)
    else:
        d = dists[:, 0]
    return d


def cluster_and_prune(
    X: np.ndarray,
    *,
    cluster_method: str = "none",
    drop_outliers: bool = False,
    hdbscan_min_cluster_size: int = 10,
    hdbscan_min_samples: int | None = None,
    prune_top_pct: float = 0.0,
    prune_knn_k: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Optionally cluster in embedding space and prune outliers before projection.

    Returns (X_filtered, clusters_filtered_or_None, keep_indices)
    where keep_indices are the indices into the original X that remain.
    """
    n = X.shape[0]
    keep = np.arange(n)
    clusters: np.ndarray | None = None

    # Optional HDBSCAN clustering
    labels_hdb: np.ndarray | None = None
    if cluster_method == "hdbscan" or drop_outliers:
        try:
            import hdbscan  # type: ignore
        except Exception as e:
            print(f"HDBSCAN not available ({e}); skipping HDBSCAN clustering/outlier drop.")
            labels_hdb = None
        else:
            min_samples = hdbscan_min_samples if hdbscan_min_samples is not None else None
            # Use euclidean on L2-normalized vectors; approximates angular distance
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(2, int(hdbscan_min_cluster_size)),
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_epsilon=0.0,
                allow_single_cluster=False,
                core_dist_n_jobs=1,
            )
            labels_hdb = clusterer.fit_predict(X)
            if drop_outliers and labels_hdb is not None:
                mask = labels_hdb != -1
                if mask.any():
                    keep = keep[mask]
                    X = X[mask]
                    labels_hdb = labels_hdb[mask]
                else:
                    print("All points flagged as outliers by HDBSCAN; skipping drop.")
            if cluster_method == "hdbscan":
                clusters = labels_hdb

    # Optional kNN-distance pruning (independent of HDBSCAN)
    if prune_top_pct and prune_top_pct > 0.0 and X.shape[0] > 2:
        scores = _knn_outlier_scores(X, k=int(prune_knn_k), metric="cosine")
        q = np.quantile(scores, 1.0 - min(0.95, max(0.0, float(prune_top_pct))))
        mask = scores <= q
        if mask.any() and mask.sum() >= 2:
            keep = keep[mask]
            X = X[mask]
            if clusters is not None:
                clusters = clusters[mask]
        else:
            print("kNN pruning would remove all/most points; skipping.")

    return X, clusters, keep


def reduce_to_nd(
    X: np.ndarray,
    method: str = "pca",
    *,
    n_components: int = 2,
    pca_dim: int = 50,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    densmap: bool = False,
    random_state: int = 42,
) -> np.ndarray:
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("Need at least 2 samples to project")
    if method == "pca":
        pca = PCA(n_components=n_components, random_state=random_state)
        return pca.fit_transform(X)
    elif method == "tsne":
        # Optional pre-PCA to speed up and denoise
        n = X.shape[0]
        d = X.shape[1]
        k = min(pca_dim, d, n - 1) if n > 2 else min(n_components, d)
        if k >= 2 and k < d:
            Xp = PCA(n_components=k, random_state=random_state).fit_transform(X)
        else:
            Xp = X
        tsne = TSNE(n_components=n_components, perplexity=min(perplexity, max(5, (n - 1) // 3)), init="pca", learning_rate="auto", random_state=random_state)
        return tsne.fit_transform(Xp)
    elif method == "umap":
        # For cosine metric, skip pre-PCA; otherwise apply a light pre-PCA
        n = X.shape[0]
        d = X.shape[1]
        if metric == "cosine":
            Xp = X
        else:
            k = min(pca_dim, d, n - 1) if n > 2 else min(n_components, d)
            Xp = PCA(n_components=k, random_state=random_state).fit_transform(X) if (k >= 2 and k < d) else X
        try:
            from umap import UMAP  # type: ignore
        except Exception:
            import umap as _umap  # type: ignore
            UMAP = _umap.UMAP
        umap = UMAP(
            n_components=n_components,
            n_neighbors=max(2, int(n_neighbors)),
            min_dist=float(min_dist),
            metric=metric,
            densmap=bool(densmap),
            init="spectral",
            random_state=random_state,
        )
        return umap.fit_transform(Xp)
    else:
        raise ValueError(f"Unknown method: {method}")


def reduce_to_2d(**kwargs) -> np.ndarray:
    # Backwards-compatible wrapper
    return reduce_to_nd(n_components=2, **kwargs)


def plot_map(coords: np.ndarray, labels: List[str], out_png: Path, clusters: np.ndarray | None = None) -> None:
    try:
        import matplotlib.pyplot as plt  # lazy import so CSV-only runs work without matplotlib
    except Exception:
        print("matplotlib not installed; skipping PNG plot. CSV will still be saved.")
        return
    plt.figure(figsize=(10, 8))
    x, y = coords[:, 0], coords[:, 1]
    if clusters is None:
        plt.scatter(x, y, s=14, c="#4C78A8", alpha=0.75, edgecolors="none")
    else:
        import matplotlib.cm as cm
        uniq = np.unique(clusters)
        colors = cm.get_cmap('tab20', len(uniq))
        for idx, k in enumerate(uniq):
            m = clusters == k
            plt.scatter(x[m], y[m], s=14, color=colors(idx), alpha=0.75, edgecolors="none", label=str(k))
        if len(uniq) <= 20:
            plt.legend(title="Cluster", fontsize=8, markerscale=1.5, frameon=False)
    # Annotate a subset to avoid clutter (top 60 by spread)
    if len(labels) <= 60:
        to_annotate = range(len(labels))
    else:
        center = coords.mean(axis=0, keepdims=True)
        d2 = np.sum((coords - center) ** 2, axis=1)
        to_annotate = np.argsort(-d2)[:60]
    for i in to_annotate:
        plt.annotate(labels[i], (x[i], y[i]), fontsize=8, alpha=0.9)
    plt.title("Book Style Map")
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Create a 2D style map of books using the trained contrastive encoder")
    ap.add_argument("--model-dir", required=True, help="Path to contrastive model directory (the 'final' folder)")
    ap.add_argument("--books-dir", required=True, help="Directory with .txt books (e.g., ./training)")
    ap.add_argument("--max-books", type=int, default=200, help="Max number of books to include (to keep compute reasonable)")
    ap.add_argument("--num-chunks", default="auto", help="Chunks per book: integer or 'auto' (cap ~8)")
    ap.add_argument("--chunk-size", type=int, default=14, help="Sentences per chunk")
    ap.add_argument("--overlap", type=int, default=4, help="Sentence overlap between chunks")
    ap.add_argument("--max-chars", type=int, default=None, help="Optional cap on chars per book for speed (e.g., 300000)")
    ap.add_argument("--method", choices=["pca", "tsne", "umap"], default="umap", help="Dimensionality reduction method")
    ap.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity (ignored for PCA)")
    ap.add_argument("--pca-dim", type=int, default=50, help="Pre-PCA dim for t-SNE/UMAP (used unless metric=cosine)")
    ap.add_argument("--n-neighbors", type=int, default=15, help="UMAP neighbors (local vs global structure)")
    ap.add_argument("--min-dist", type=float, default=0.1, help="UMAP cluster tightness (lower = tighter)")
    ap.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine", help="UMAP distance metric")
    ap.add_argument("--densmap", action="store_true", help="Enable UMAP densMAP (density-preserving)")
    ap.add_argument("--remove-top-pcs", type=int, default=0, help="Remove top-K principal directions before projection")
    ap.add_argument("--book-pool", choices=["mean", "trimmed_mean", "median", "topk_mean"], default="trimmed_mean", help="Pooling across chunk embeddings per book")
    ap.add_argument("--trim-frac", type=float, default=0.2, help="Trim fraction for trimmed_mean (0.2 keeps middle 80%)")
    ap.add_argument("--book-topk", type=int, default=5, help="k for topk_mean pooling")
    ap.add_argument("--use-projection", type=lambda x: str(x).lower() not in {"0","false","no"}, default=False, help="Apply model projection head when embedding chunks")
    ap.add_argument("--cluster-method", choices=["none", "kmeans", "hdbscan"], default="none", help="Optional cluster coloring (computed in embedding space)")
    # HDBSCAN-specific options and pruning
    ap.add_argument("--hdbscan-min-cluster-size", type=int, default=10, help="HDBSCAN min cluster size (embedding space)")
    ap.add_argument("--hdbscan-min-samples", type=int, default=None, help="HDBSCAN min samples; defaults to min_cluster_size if unset")
    ap.add_argument("--drop-outliers", action="store_true", help="If using HDBSCAN, drop label=-1 points before UMAP")
    ap.add_argument("--prune-top-pct", type=float, default=0.0, help="Drop top-pct outliers by avg kNN distance in embedding space (0.0 disables)")
    ap.add_argument("--prune-knn-k", type=int, default=10, help="k for kNN-based pruning score")
    ap.add_argument("--n-clusters", type=int, default=12, help="Number of clusters for kmeans coloring")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for projection")
    ap.add_argument("--n-components", type=int, choices=[2, 3], default=2, help="Output dimensionality (2 or 3)")
    ap.add_argument("--plotly-html", type=str, default=None, help="Optional path to save an interactive Plotly HTML plot")
    ap.add_argument("--out-prefix", default="outputs/style_map", help="Output prefix for CSV/PNG files")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto", help="Compute device override")
    ap.add_argument("--embed-batch-size", type=int, default=256, help="Chunk texts per forward pass (increase on GPU)")
    ap.add_argument("--auto-tune-embed-batch", action="store_true", help="Adaptively reduce batch size on OOM")
    ap.add_argument("--max-length", type=int, default=384, help="Tokenizer max_length per chunk (e.g., 384 vs 512)")
    args = ap.parse_args()

    # Parse num_chunks possibly as int
    try:
        num_chunks = int(args.num_chunks)  # type: ignore[arg-type]
    except Exception:
        num_chunks = "auto"

    # Load model
    model = ContrastiveBookMatcherInference(args.model_dir)
    # Optional device override (including MPS)
    if args.device != "auto":
        import torch
        dev = torch.device(args.device)
        model.model.to(dev)
        model.device = dev

    # Gather books
    bdir = Path(args.books_dir)
    files = sorted([p for p in bdir.glob("**/*.txt") if p.is_file()])
    if not files:
        raise SystemExit(f"No .txt files found under {bdir}")
    if args.max_books and len(files) > args.max_books:
        files = files[: args.max_books]

    # Prepare chunks for all books first
    books: List[Tuple[str, List[str]]] = []
    for path in tqdm(files, desc="Preparing chunks", leave=False):
        text = _read_text(path, max_chars=args.max_chars)
        chunks = _prepare_book_chunks(text, num_chunks=num_chunks, chunk_size=args.chunk_size, overlap=args.overlap)
        books.append((path.stem, chunks))

    labels, X = embed_books_batched(
        model,
        books,
        max_length=int(args.max_length),
        embed_batch_size=int(args.embed_batch_size),
        auto_tune_embed_batch=bool(args.auto_tune_embed_batch),
        use_projection=args.use_projection,
        book_pool=args.book_pool,
        trim_frac=args.trim_frac,
        book_topk=args.book_topk,
    )
    if X.size == 0:
        raise SystemExit("No embeddings computed — check inputs")
    # Optional dominant component removal
    if args.remove_top_pcs and args.remove_top_pcs > 0:
        X = _remove_top_pcs(X, int(args.remove_top_pcs))

    # Optional clustering and pruning before projection
    Xf, clusters, keep_idx = cluster_and_prune(
        X,
        cluster_method=args.cluster_method,
        drop_outliers=bool(args.drop_outliers),
        hdbscan_min_cluster_size=int(args.hdbscan_min_cluster_size),
        hdbscan_min_samples=(None if args.hdbscan_min_samples is None else int(args.hdbscan_min_samples)),
        prune_top_pct=float(args.prune_top_pct),
        prune_knn_k=int(args.prune_knn_k),
        seed=int(args.seed),
    )
    if Xf.shape[0] != X.shape[0]:
        labels = [labels[i] for i in keep_idx.tolist()]
        X = Xf

    coords = reduce_to_nd(
        X,
        method=args.method,
        n_components=args.n_components,
        pca_dim=args.pca_dim,
        perplexity=args.perplexity,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        densmap=args.densmap,
        random_state=args.seed,
    )

    # Optional clustering in embedding space for coloring
    if clusters is None and args.cluster_method == "kmeans":
        try:
            from sklearn.cluster import KMeans
            k = max(2, int(args.n_clusters))
            clusters = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit_predict(X)
        except Exception as e:
            print(f"Clustering failed: {e}")

    # Save CSV
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_csv = out_prefix.with_suffix(".csv")
    import csv
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if args.n_components == 3:
            cols = ["label", "x", "y", "z"]
            if clusters is not None:
                cols.append("cluster")
            w.writerow(cols)
            for i, lab in enumerate(labels):
                x, y, z = coords[i, 0], coords[i, 1], coords[i, 2]
                row = [lab, float(x), float(y), float(z)]
                if clusters is not None:
                    row.append(int(clusters[i]))
                w.writerow(row)
        else:
            cols = ["label", "x", "y"]
            if clusters is not None:
                cols.append("cluster")
            w.writerow(cols)
            for i, lab in enumerate(labels):
                x, y = coords[i, 0], coords[i, 1]
                row = [lab, float(x), float(y)]
                if clusters is not None:
                    row.append(int(clusters[i]))
                w.writerow(row)

    # Save plot
    out_png = out_prefix.with_suffix(".png")
    # Only plot PNG when 2D
    if args.n_components == 2:
        plot_map(coords, labels, out_png, clusters=clusters)
    # Optional interactive Plotly HTML
    if args.plotly_html:
        try:
            import plotly.express as px  # type: ignore
            if args.n_components == 3:
                fig = px.scatter_3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], hover_name=labels, height=720, width=960)
            else:
                fig = px.scatter(x=coords[:,0], y=coords[:,1], hover_name=labels, height=720, width=960)
            fig.update_traces(marker=dict(size=3))
            fig.write_html(args.plotly_html, include_plotlyjs='cdn', full_html=True)
        except Exception as e:
            print(f"Plotly not available or failed to save HTML: {e}")

    if args.n_components == 2:
        print(f"Saved style map: {out_png}")
    print(f"Saved coordinates: {out_csv}")
    if args.plotly_html:
        print(f"Saved interactive HTML: {args.plotly_html}")


if __name__ == "__main__":
    main()
