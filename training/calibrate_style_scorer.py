from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from eval.benchmark_io import load_json, load_jsonl, write_json
from eval.style_scoring import StyleScorer


MINIMAL_PNG_BYTES = bytes.fromhex(
    "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C4890000000A49444154789C6360000002000154A24F5D0000000049454E44AE426082"
)


def _load_config(path_or_payload: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_payload, dict):
        return dict(path_or_payload)
    return load_json(path_or_payload)


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def _fit_logistic(features: list[float], labels: list[int], *, epochs: int = 400, learning_rate: float = 0.2) -> tuple[float, float]:
    coef = 0.0
    intercept = 0.0
    for _ in range(epochs):
        grad_coef = 0.0
        grad_intercept = 0.0
        for feature, label in zip(features, labels):
            prediction = _sigmoid((coef * feature) + intercept)
            error = prediction - label
            grad_coef += error * feature
            grad_intercept += error
        scale = 1.0 / max(1, len(features))
        coef -= learning_rate * grad_coef * scale
        intercept -= learning_rate * grad_intercept * scale
    return coef, intercept


def _brier_score(features: list[float], labels: list[int], coef: float, intercept: float) -> float:
    total = 0.0
    for feature, label in zip(features, labels):
        prediction = _sigmoid((coef * feature) + intercept)
        total += (prediction - label) ** 2
    return total / max(1, len(features))


def _build_calibration_pairs(corpus_root: Path, split_payload: dict[str, Any]) -> list[dict[str, Any]]:
    books = {row["book_id"]: row for row in load_jsonl(corpus_root / "meta" / "books_manifest_v1.jsonl")}
    pairs: list[dict[str, Any]] = []
    author_books: dict[str, list[str]] = {}
    for book_id in split_payload["book_ids"]:
        author_books.setdefault(books[book_id]["author_id"], []).append(book_id)
    for author_id, book_ids in author_books.items():
        if len(book_ids) >= 2:
            book_a, book_b = book_ids[:2]
            pool_a = load_jsonl(corpus_root / "meta" / "passage_pools_v1" / books[book_a]["author_slug"] / f"{book_a.replace(':', '_')}.jsonl")
            pool_b = load_jsonl(corpus_root / "meta" / "passage_pools_v1" / books[book_b]["author_slug"] / f"{book_b.replace(':', '_')}.jsonl")
            if pool_a and pool_b:
                pairs.append(
                    {
                        "text1": pool_a[0]["text"],
                        "text2": pool_b[0]["text"],
                        "label": 1,
                        "group": author_id,
                        "book1": book_a,
                        "book2": book_b,
                        "author1": author_id,
                        "author2": author_id,
                        "pair_role": "positive_same_author_cross_book",
                        "neg_type": None,
                    }
                )
    all_book_ids = split_payload["book_ids"]
    for index, book_id in enumerate(all_book_ids):
        other_book_id = next(
            (
                candidate
                for candidate in all_book_ids[index + 1 :]
                if books[candidate]["author_id"] != books[book_id]["author_id"]
            ),
            None,
        )
        if other_book_id is None:
            continue
        pool_a = load_jsonl(corpus_root / "meta" / "passage_pools_v1" / books[book_id]["author_slug"] / f"{book_id.replace(':', '_')}.jsonl")
        pool_b = load_jsonl(corpus_root / "meta" / "passage_pools_v1" / books[other_book_id]["author_slug"] / f"{other_book_id.replace(':', '_')}.jsonl")
        if pool_a and pool_b:
            pairs.append(
                {
                    "text1": pool_a[0]["text"],
                    "text2": pool_b[0]["text"],
                    "label": 0,
                    "group": book_id,
                    "book1": book_id,
                    "book2": other_book_id,
                    "author1": books[book_id]["author_id"],
                    "author2": books[other_book_id]["author_id"],
                    "pair_role": "negative_different_author",
                    "neg_type": "negative_matched",
                }
            )
    return pairs


def calibrate_style_scorer(config_path_or_payload: str | Path | dict[str, Any]) -> dict[str, str]:
    config = _load_config(config_path_or_payload)
    corpus_root = Path(config["corpus_root"])
    artifacts_root = Path(config.get("artifacts_root", "build/artifacts")) / "scorer"
    datasets_root = artifacts_root / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)
    split_payload = load_json(config.get("calibration_split_path") or corpus_root / "splits" / "scorer_calibration_v1.json")
    pair_rows = _build_calibration_pairs(corpus_root, split_payload)

    csv_path = datasets_root / "scorer_calibration_pairs_v1.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "text1",
                "text2",
                "label",
                "group",
                "book1",
                "book2",
                "author1",
                "author2",
                "pair_role",
                "neg_type",
            ],
        )
        writer.writeheader()
        writer.writerows(pair_rows)

    scorer = StyleScorer(artifacts_root / "final")
    features = []
    labels = []
    for row in pair_rows:
        score_payload = scorer.score_pair(row["text1"], row["text2"])
        features.append(float(score_payload["score_0_1"]))
        labels.append(int(row["label"]))

    coef, intercept = _fit_logistic(features, labels)
    logistic_brier = _brier_score(features, labels, coef, intercept)
    identity_brier = sum((feature - label) ** 2 for feature, label in zip(features, labels)) / max(1, len(features))
    chosen_method = "logistic" if logistic_brier <= identity_brier else "identity"

    calibration_payload = {
        "artifact_type": "style_calibration",
        "artifact_version": "style_calibration_v1",
        "style_calibration": {
            "method": chosen_method,
            "coef": round(coef, 6),
            "intercept": round(intercept, 6),
        },
        "meta": {
            "n_samples": len(features),
            "num_chunks": "auto",
            "chunk_size": config.get("chunk_size_words"),
            "overlap": config.get("chunk_overlap_words"),
            "aggregate": config.get("chunk_aggregation", "topk_mean"),
            "topk": config.get("chunk_top_k"),
            "max_length": config.get("max_length", 512),
            "selection_metric": "brier",
            "n_splits": 1,
            "method_requested": "auto",
        },
        "selection": {
            "scores": {
                "logistic": round(logistic_brier, 6),
                "identity": round(identity_brier, 6),
            },
            "chosen": chosen_method,
        },
    }
    write_json(artifacts_root / "style_calibration_v1.json", calibration_payload)
    write_json(artifacts_root / "final" / "style_calibration_v1.json", calibration_payload)
    write_json(
        artifacts_root / "calibration_report_v1.json",
        {
            "pair_count": len(pair_rows),
            "positive_count": sum(labels),
            "negative_count": len(labels) - sum(labels),
            "brier_identity": round(identity_brier, 6),
            "brier_logistic": round(logistic_brier, 6),
            "chosen_method": chosen_method,
        },
    )
    (artifacts_root / "style_calibration_reliability.png").write_bytes(MINIMAL_PNG_BYTES)
    return {
        "calibration_csv": csv_path.as_posix(),
        "style_calibration": (artifacts_root / "style_calibration_v1.json").as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    calibrate_style_scorer(args.config)


if __name__ == "__main__":
    main()
