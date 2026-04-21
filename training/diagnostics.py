from __future__ import annotations

from pathlib import Path
from typing import Any

from eval.benchmark_io import load_jsonl, write_json
from eval.style_scoring import StyleScorer


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _row_score(scorer: StyleScorer, row: dict[str, Any]) -> dict[str, Any]:
    payload = scorer.score_pair(row["text1"], row["text2"])
    return {
        "label": int(row["label"]),
        "score": float(payload.get("calibrated", payload["score_0_1"])),
        "raw_similarity": float(payload.get("raw_similarity", payload["score_0_1"])),
        "masked_similarity": float(payload.get("masked_similarity", payload["score_0_1"])),
    }


def _subset_report(scorer: StyleScorer, rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "count": 0,
            "score_mean": 0.0,
            "accuracy_at_0_5": 0.0,
            "raw_similarity_mean": 0.0,
            "masked_similarity_mean": 0.0,
        }
    scored = [_row_score(scorer, row) for row in rows]
    return {
        "count": len(scored),
        "score_mean": round(mean([row["score"] for row in scored]), 6),
        "accuracy_at_0_5": round(
            mean([1.0 if ((row["score"] >= 0.5) == bool(row["label"])) else 0.0 for row in scored]),
            6,
        ),
        "raw_similarity_mean": round(mean([row["raw_similarity"] for row in scored]), 6),
        "masked_similarity_mean": round(mean([row["masked_similarity"] for row in scored]), 6),
    }


def build_style_diagnostics(
    *,
    model_dir: str | Path,
    dataset_root: str | Path,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    rows = load_jsonl(dataset_root / "test_pairs_v1.jsonl") or load_jsonl(dataset_root / "validation_pairs_v1.jsonl")
    scorer = StyleScorer(model_dir)
    same_author = [row for row in rows if bool(row.get("same_author"))]
    same_book = [row for row in rows if bool(row.get("same_book"))]
    cross_book_positive = [row for row in rows if int(row["label"]) == 1 and not bool(row.get("same_book"))]
    topic_confusable_negative = [
        row
        for row in rows
        if int(row["label"]) == 0 and bool(row.get("same_content_cluster"))
    ]
    nonconfusable_negative = [
        row
        for row in rows
        if int(row["label"]) == 0 and not bool(row.get("same_content_cluster"))
    ]
    diagnostics = {
        "artifact_type": "style_scorer_diagnostics",
        "artifact_version": "style_scorer_diagnostics_v1",
        "row_count": len(rows),
        "same_author": _subset_report(scorer, same_author),
        "same_book": _subset_report(scorer, same_book),
        "cross_book_positive": _subset_report(scorer, cross_book_positive),
        "topic_confusable_negative": _subset_report(scorer, topic_confusable_negative),
        "nonconfusable_negative": _subset_report(scorer, nonconfusable_negative),
    }
    diagnostics["topic_leakage_delta"] = round(
        diagnostics["topic_confusable_negative"]["score_mean"] - diagnostics["nonconfusable_negative"]["score_mean"],
        6,
    )
    diagnostics["masked_minus_raw_on_topic_confusable_negative"] = round(
        diagnostics["topic_confusable_negative"]["masked_similarity_mean"]
        - diagnostics["topic_confusable_negative"]["raw_similarity_mean"],
        6,
    )
    if out_path:
        write_json(out_path, diagnostics)
    return diagnostics
