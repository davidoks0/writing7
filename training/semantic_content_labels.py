from __future__ import annotations

import hashlib
import math
from typing import Any


try:  # pragma: no cover - optional heavyweight dependency path
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector[:]
    return [value / norm for value in vector]


def hashed_semantic_embedding(text: str, dimension: int = 256) -> list[float]:
    vector = [0.0] * dimension
    tokens = text.lower().split()
    if not tokens:
        return vector
    for token in tokens:
        bucket = int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16) % dimension
        vector[bucket] += 1.0
    return _normalize(vector)


def _cosine(first: list[float], second: list[float]) -> float:
    return sum(left * right for left, right in zip(first, second))


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    pooled = [0.0] * width
    for vector in vectors:
        for index, value in enumerate(vector):
            pooled[index] += value
    return _normalize([value / len(vectors) for value in pooled])


def infer_content_embeddings(
    passages: list[dict[str, Any]],
    *,
    model_id: str | None = None,
    embedding_dimension: int = 256,
) -> tuple[list[list[float]], dict[str, Any]]:
    texts = [str(row.get("text", "")).strip() for row in passages]
    if model_id and SentenceTransformer is not None:
        try:  # pragma: no cover - model bootstrap depends on local cache
            encoder = SentenceTransformer(model_id)
            embeddings = encoder.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=False,
            )
            normalized = []
            for row in embeddings:
                if hasattr(row, "tolist"):
                    normalized.append(_normalize([float(value) for value in row.tolist()]))
                else:
                    normalized.append(_normalize([float(value) for value in row]))
            return normalized, {
                "backend": "sentence_transformers",
                "model_id": model_id,
                "degraded": False,
            }
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
        else:  # pragma: no cover
            fallback_reason = None
    else:
        fallback_reason = "sentence-transformers unavailable" if model_id else "no semantic teacher configured"

    return [hashed_semantic_embedding(text, dimension=embedding_dimension) for text in texts], {
        "backend": "hashed_bow_semantic_proxy",
        "model_id": model_id,
        "degraded": True,
        "fallback_reason": fallback_reason,
    }


def _initial_centroids(embeddings: list[list[float]], cluster_count: int, seed: int) -> list[list[float]]:
    ordered_indices = sorted(
        range(len(embeddings)),
        key=lambda index: (
            int(hashlib.sha1(f"{seed}|{index}".encode("utf-8")).hexdigest()[:16], 16),
            index,
        ),
    )
    return [embeddings[index][:] for index in ordered_indices[:cluster_count]]


def assign_content_clusters(
    embeddings: list[list[float]],
    *,
    cluster_count: int,
    seed: int = 42,
    max_iterations: int = 10,
) -> list[int]:
    if not embeddings:
        return []
    effective_cluster_count = max(1, min(cluster_count, len(embeddings)))
    if effective_cluster_count == 1:
        return [0] * len(embeddings)

    centroids = _initial_centroids(embeddings, effective_cluster_count, seed)
    labels = [0] * len(embeddings)
    for _ in range(max_iterations):
        updated_labels = []
        for embedding in embeddings:
            scores = [_cosine(embedding, centroid) for centroid in centroids]
            best_index = max(range(len(scores)), key=lambda index: (scores[index], -index))
            updated_labels.append(best_index)
        if updated_labels == labels:
            break
        labels = updated_labels
        new_centroids: list[list[float]] = []
        for cluster_index in range(effective_cluster_count):
            members = [embedding for embedding, label in zip(embeddings, labels) if label == cluster_index]
            if members:
                new_centroids.append(_mean_vector(members))
            else:
                replacement_index = sorted(
                    range(len(embeddings)),
                    key=lambda index: (
                        int(hashlib.sha1(f"{seed}|empty|{cluster_index}|{index}".encode("utf-8")).hexdigest()[:16], 16),
                        index,
                    ),
                )[0]
                new_centroids.append(embeddings[replacement_index][:])
        centroids = new_centroids
    return labels


def resolve_content_cluster_count(passage_count: int, requested: int | None = None) -> int:
    if requested is not None:
        return max(1, min(requested, max(1, passage_count)))
    heuristic = int(round(math.sqrt(max(1, passage_count)))) * 2
    return max(4, min(32, min(heuristic, max(1, passage_count))))
