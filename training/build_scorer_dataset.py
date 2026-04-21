from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.benchmark_io import load_json, load_jsonl, write_json, write_jsonl
from eval.benchmark_schema import PassageRecord
from eval.passage_sampling import sentence_windows_overlap
from training.semantic_content_labels import (
    assign_content_clusters,
    infer_content_embeddings,
    resolve_content_cluster_count,
)
from training.style_text import STYLE_MASKED_TEXT_VIEW, apply_text_view


def _load_config(path_or_payload: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_payload, dict):
        return dict(path_or_payload)
    return load_json(path_or_payload)


def _rank_passages(passages: list[dict[str, Any]], builder_seed: int) -> list[dict[str, Any]]:
    return sorted(
        passages,
        key=lambda row: (
            int(hashlib.sha1(f"{builder_seed}|{row['passage_id']}".encode("utf-8")).hexdigest()[:16], 16),
            row["passage_id"],
        ),
    )


def _hashed_embedding(text: str, dimension: int = 256) -> list[float]:
    vector = [0.0] * dimension
    for token in text.lower().split():
        bucket = int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16) % dimension
        vector[bucket] += 1.0
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _cosine(first: list[float], second: list[float]) -> float:
    numerator = sum(left * right for left, right in zip(first, second))
    norm_first = sum(value * value for value in first) ** 0.5
    norm_second = sum(value * value for value in second) ** 0.5
    if norm_first == 0.0 or norm_second == 0.0:
        return 0.0
    return numerator / (norm_first * norm_second)


def build_scorer_dataset(config_path_or_payload: str | Path | dict[str, Any]) -> dict[str, str]:
    config = _load_config(config_path_or_payload)
    corpus_root = Path(config["corpus_root"])
    builder_seed = int(config.get("builder_seed", 42))
    output_root = Path(config.get("artifacts_root", "build/artifacts")) / "scorer" / "datasets"
    output_root.mkdir(parents=True, exist_ok=True)

    split = load_json(config.get("split_path") or corpus_root / "splits" / "scorer_train_v1.json")
    books = {row["book_id"]: row for row in load_jsonl(corpus_root / "meta" / "books_manifest_v1.jsonl")}
    passage_pools: dict[str, list[dict[str, Any]]] = {}
    passages_by_author: dict[str, list[dict[str, Any]]] = defaultdict(list)
    book_centroids: dict[str, list[float]] = {}
    all_passages: list[dict[str, Any]] = []

    for book_id in split["book_ids"]:
        book = books[book_id]
        pool_path = corpus_root / "meta" / "passage_pools_v1" / book["author_slug"] / f"{book_id.replace(':', '_')}.jsonl"
        passages = _rank_passages(load_jsonl(pool_path), builder_seed)
        if not passages:
            continue
        passage_pools[book_id] = passages
        passages_by_author[book["author_id"]].extend(passages)
        book_centroids[book_id] = _hashed_embedding(" ".join(passage["text"] for passage in passages[:10]))
        all_passages.extend(passages)

    content_embeddings, content_embedding_meta = infer_content_embeddings(
        all_passages,
        model_id=config.get("semantic_adversary_model"),
        embedding_dimension=int(config.get("semantic_embedding_dim", 256)),
    )
    content_cluster_count = resolve_content_cluster_count(
        len(all_passages),
        requested=config.get("content_cluster_count"),
    )
    content_cluster_labels = assign_content_clusters(
        content_embeddings,
        cluster_count=content_cluster_count,
        seed=builder_seed,
    )
    passage_content_labels = {
        passage["passage_id"]: int(label)
        for passage, label in zip(all_passages, content_cluster_labels)
    }

    same_period_books: dict[str, list[str]] = defaultdict(list)
    for book_id in split["book_ids"]:
        period = books[book_id].get("period_bucket", "unknown") or "unknown"
        same_period_books[period].append(book_id)

    raw_rows: list[dict[str, Any]] = []
    raw_index = 0
    for book_id in split["book_ids"]:
        if book_id not in passage_pools:
            continue
        book = books[book_id]
        author_id = book["author_id"]
        author_books = [candidate for candidate in split["book_ids"] if books[candidate]["author_id"] == author_id and candidate in passage_pools]
        different_author_books = [candidate for candidate in split["book_ids"] if books[candidate]["author_id"] != author_id and candidate in passage_pools]
        anchors = passage_pools[book_id][:20]
        for anchor in anchors:
            anchor_record = PassageRecord.from_dict(anchor)
            positive = None
            for candidate_book_id in author_books:
                if candidate_book_id == book_id:
                    continue
                candidate_pool = passage_pools[candidate_book_id]
                if candidate_pool:
                    positive = candidate_pool[0]
                    break
            if positive is None:
                for candidate in passage_pools[book_id]:
                    if candidate["passage_id"] == anchor["passage_id"]:
                        continue
                    if not sentence_windows_overlap(anchor_record, PassageRecord.from_dict(candidate)):
                        positive = candidate
                        break
            if positive is None:
                continue

            period = book.get("period_bucket", "unknown")
            matched_candidates = [
                candidate
                for candidate in same_period_books.get(period, [])
                if candidate != book_id and books[candidate]["author_id"] != author_id and candidate in passage_pools
            ]
            if not matched_candidates:
                matched_candidates = different_author_books
            negative_matched_book = matched_candidates[0] if matched_candidates else None
            negative_matched = passage_pools[negative_matched_book][0] if negative_matched_book else None

            anchor_centroid = book_centroids[book_id]
            hard_candidates = []
            for candidate_book_id in different_author_books:
                similarity = _cosine(anchor_centroid, book_centroids[candidate_book_id])
                hard_candidates.append((similarity, candidate_book_id))
            hard_candidates.sort(key=lambda item: (-item[0], item[1]))
            negative_hard_book = hard_candidates[0][1] if hard_candidates else negative_matched_book
            negative_hard = passage_pools[negative_hard_book][1 if len(passage_pools[negative_hard_book]) > 1 else 0] if negative_hard_book else None

            candidate_rows = [
                (positive, 1, "positive_same_author_cross_book" if positive["book_id"] != book_id else "positive_same_author_same_book", None),
                (negative_matched, 0, "negative_different_author", "negative_matched"),
                (negative_hard, 0, "negative_different_author", "negative_hard" if negative_hard_book else "negative_random"),
            ]
            for passage, label, pair_role, neg_type in candidate_rows:
                if passage is None:
                    continue
                raw_rows.append(
                    {
                        "_raw_index": raw_index,
                        "passage1_id": anchor["passage_id"],
                        "passage2_id": passage["passage_id"],
                        "text1": anchor["text"],
                        "text2": passage["text"],
                        "style_text1": apply_text_view(anchor["text"], STYLE_MASKED_TEXT_VIEW),
                        "style_text2": apply_text_view(passage["text"], STYLE_MASKED_TEXT_VIEW),
                        "label": label,
                        "pair_role": pair_role,
                        "neg_type": neg_type,
                        "book1": anchor["book_id"],
                        "book2": passage["book_id"],
                        "author1": anchor["author_id"],
                        "author2": passage["author_id"],
                        "content_cluster1": passage_content_labels.get(anchor["passage_id"]),
                        "content_cluster2": passage_content_labels.get(passage["passage_id"]),
                        "same_author": anchor["author_id"] == passage["author_id"],
                        "same_book": anchor["book_id"] == passage["book_id"],
                        "same_content_cluster": passage_content_labels.get(anchor["passage_id"]) == passage_content_labels.get(passage["passage_id"]),
                    }
                )
                raw_index += 1

    ordered = sorted(
        raw_rows,
        key=lambda row: (
            int(hashlib.sha1(f"{builder_seed}|{row['_raw_index']}".encode("utf-8")).hexdigest()[:16], 16),
            row["_raw_index"],
        ),
    )

    total = len(ordered)
    train_cutoff = int(total * 0.85)
    validation_cutoff = int(total * 0.95)
    split_rows = {"train": [], "validation": [], "test": []}
    for index, row in enumerate(ordered):
        if index < train_cutoff:
            split_name = "train"
        elif index < validation_cutoff:
            split_name = "validation"
        else:
            split_name = "test"
        pair_index = len(split_rows[split_name]) + 1
        split_rows[split_name].append(
            {
                "pair_id": f"scorerpair:{split_name}:{pair_index:08d}",
                "split": split_name,
                **{key: value for key, value in row.items() if not key.startswith("_")},
            }
        )

    write_jsonl(output_root / "train_pairs_v1.jsonl", split_rows["train"])
    write_jsonl(output_root / "validation_pairs_v1.jsonl", split_rows["validation"])
    write_jsonl(output_root / "test_pairs_v1.jsonl", split_rows["test"])
    meta = {
        "artifact_type": "scorer_dataset",
        "artifact_version": "scorer_dataset_v1",
        "builder_seed": builder_seed,
        "counts": {split_name: len(rows) for split_name, rows in split_rows.items()},
        "content_labels": {
            "cluster_count": content_cluster_count,
            **content_embedding_meta,
        },
    }
    write_json(output_root / "scorer_dataset_meta_v1.json", meta)
    return {
        "train_pairs": (output_root / "train_pairs_v1.jsonl").as_posix(),
        "validation_pairs": (output_root / "validation_pairs_v1.jsonl").as_posix(),
        "test_pairs": (output_root / "test_pairs_v1.jsonl").as_posix(),
        "meta": (output_root / "scorer_dataset_meta_v1.json").as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    build_scorer_dataset(args.config)


if __name__ == "__main__":
    main()
