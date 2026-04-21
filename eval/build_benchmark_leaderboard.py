from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
from typing import Any

from eval.aggregate_benchmark_results import (
    aggregate_benchmark_results,
    build_leaderboard_row,
    render_leaderboard_markdown,
)
from eval.benchmark_io import load_json, write_json


def _load_summary(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.suffix == ".jsonl":
        return aggregate_benchmark_results(source)
    payload = load_json(source)
    if "metrics_all" in payload and "metrics_valid" in payload:
        return payload
    raise ValueError(f"Unsupported leaderboard input: {source}")


def _sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, str]:
    return (
        -float(row.get("style_mimicry_score", 0.0) or 0.0),
        -float(row.get("style_win_rate_valid", 0.0) or 0.0),
        -float(row.get("valid_rate", 0.0) or 0.0),
        -float(row.get("style_margin_valid", 0.0) or 0.0),
        str(row.get("model_name") or ""),
    )


def _render_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    ordered_fields = [
        "benchmark_version",
        "provider",
        "model_name",
        "track",
        "split",
        "sample_count",
        "valid_sample_count",
        "style_mimicry_score",
        "style_win_rate_valid",
        "style_margin_valid",
        "top1_target_accuracy_valid",
        "style_percentile_valid",
        "valid_rate",
        "originality_pass_rate",
        "conditioning_copy_free_rate",
        "target_evaluation_copy_free_rate",
        "full_target_book_copy_free_rate",
        "fluency_pass_rate",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=ordered_fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in ordered_fields})
    return buffer.getvalue()


def build_benchmark_leaderboard(
    inputs: list[str | Path],
    out_path: str | Path | None = None,
    *,
    group_by: str | None = None,
) -> dict[str, Any]:
    rows = []
    summaries = []
    for source in inputs:
        summary = _load_summary(source)
        summaries.append(summary)
        rows.append(summary.get("leaderboard_row") or build_leaderboard_row(summary))
    rows = sorted(rows, key=_sort_key)
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    if group_by:
        for row in rows:
            group_value = str(row.get(group_by) or "unknown")
            grouped_rows.setdefault(group_value, []).append(row)
    payload = {
        "task": "style_mimicry",
        "row_count": len(rows),
        "rows": rows,
        "markdown": render_leaderboard_markdown(rows),
        "csv": _render_csv(rows),
        "sources": [Path(source).as_posix() for source in inputs],
        "tracks": sorted({row.get("track") for row in rows if row.get("track")}),
        "splits": sorted({row.get("split") for row in rows if row.get("split")}),
        "models": [row.get("model_name") for row in rows],
        "group_by": group_by,
        "grouped_rows": grouped_rows,
    }
    if out_path:
        out_path = Path(out_path)
        write_json(out_path, payload)
        (out_path.with_suffix(".md")).write_text(payload["markdown"] + "\n", encoding="utf-8")
        (out_path.with_suffix(".csv")).write_text(payload["csv"], encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--group-by")
    args = parser.parse_args()
    build_benchmark_leaderboard(args.input, args.out, group_by=args.group_by)


if __name__ == "__main__":
    main()
