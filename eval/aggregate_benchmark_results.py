from __future__ import annotations

import argparse
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.benchmark_io import load_json, load_jsonl, write_json


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _flag_value(row: dict[str, Any], flag_name: str, *, default: bool | None = None) -> bool | None:
    if flag_name in row.get("valid_flags", {}):
        return bool(row["valid_flags"][flag_name])
    return default


def _row_is_valid(row: dict[str, Any]) -> bool:
    originality_pass = bool(_flag_value(row, "originality_pass", default=True))
    fluency_pass = bool(_flag_value(row, "fluency_pass", default=True))
    return originality_pass and fluency_pass


def _reference_group_copy_free_value(row: dict[str, Any], group_name: str) -> float | None:
    group_metrics = row.get("originality_metrics", {}).get("reference_group_metrics", {})
    if group_name not in group_metrics:
        return None
    return 0.0 if group_metrics[group_name].get("copy_flag") else 1.0


def _collect_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if _row_is_valid(row)]
    originality_values = [1.0 if bool(_flag_value(row, "originality_pass", default=True)) else 0.0 for row in rows]
    fluency_flags = [_flag_value(row, "fluency_pass", default=None) for row in rows]
    entity_transplant_values = [
        0.0 if bool(row.get("originality_metrics", {}).get("entity_transplant_flag")) else 1.0
        for row in rows
    ]
    conditioning_copy_free_values = [
        value for value in (_reference_group_copy_free_value(row, "conditioning") for row in rows) if value is not None
    ]
    target_evaluation_copy_free_values = [
        value for value in (_reference_group_copy_free_value(row, "target_evaluation") for row in rows) if value is not None
    ]
    full_target_book_copy_free_values = [
        value for value in (_reference_group_copy_free_value(row, "full_target_book") for row in rows) if value is not None
    ]
    metrics_all = {
        "style_win_rate_mean": round(_mean([row["style_metrics"]["style_win_rate_case"] for row in rows]), 6),
        "style_margin_mean": round(_mean([row["style_metrics"]["style_margin_case"] for row in rows]), 6),
        "top1_target_accuracy_mean": round(_mean([row["style_metrics"]["top1_target_case"] for row in rows]), 6),
        "mrr_mean": round(_mean([row["style_metrics"]["mrr_case"] for row in rows]), 6),
        "originality_pass_rate": round(_mean(originality_values), 6),
        "entity_transplant_free_rate": round(_mean(entity_transplant_values), 6),
        "valid_rate": round(_mean([1.0 if _row_is_valid(row) else 0.0 for row in rows]), 6),
    }
    if any(flag is not None for flag in fluency_flags):
        metrics_all["fluency_pass_rate"] = round(
            _mean([1.0 if bool(flag) else 0.0 for flag in fluency_flags if flag is not None]),
            6,
        )
    if conditioning_copy_free_values:
        metrics_all["conditioning_copy_free_rate"] = round(_mean(conditioning_copy_free_values), 6)
    if target_evaluation_copy_free_values:
        metrics_all["target_evaluation_copy_free_rate"] = round(_mean(target_evaluation_copy_free_values), 6)
    if full_target_book_copy_free_values:
        metrics_all["full_target_book_copy_free_rate"] = round(_mean(full_target_book_copy_free_values), 6)
    metrics_valid = {
        "style_win_rate_mean": round(_mean([row["style_metrics"]["style_win_rate_case"] for row in valid_rows]), 6) if valid_rows else 0.0,
        "style_margin_mean": round(_mean([row["style_metrics"]["style_margin_case"] for row in valid_rows]), 6) if valid_rows else 0.0,
        "top1_target_accuracy_mean": round(_mean([row["style_metrics"]["top1_target_case"] for row in valid_rows]), 6) if valid_rows else 0.0,
        "mrr_mean": round(_mean([row["style_metrics"]["mrr_case"] for row in valid_rows]), 6) if valid_rows else 0.0,
        "style_percentile_valid_mean": round(_mean([row["style_metrics"]["style_percentile_case"] for row in valid_rows]), 6) if valid_rows else 0.0,
    }
    return {
        "sample_count": len(rows),
        "valid_sample_count": len(valid_rows),
        "metrics_all": metrics_all,
        "metrics_valid": metrics_valid,
    }


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = pct * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_case_metrics(rows: list[dict[str, Any]], *, resamples: int, seed: int = 42) -> dict[str, list[float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)
    case_ids = sorted(grouped)
    if not case_ids:
        return {"style_win_rate_valid": [0.0, 0.0], "style_margin_valid": [0.0, 0.0]}
    rng = random.Random(seed)
    win_rates = []
    margins = []
    for _ in range(resamples):
        sampled_rows: list[dict[str, Any]] = []
        for _case_id in case_ids:
            sampled_rows.extend(grouped[rng.choice(case_ids)])
        valid_rows = [row for row in sampled_rows if _row_is_valid(row)]
        win_rates.append(_mean([row["style_metrics"]["style_win_rate_case"] for row in valid_rows]) if valid_rows else 0.0)
        margins.append(_mean([row["style_metrics"]["style_margin_case"] for row in valid_rows]) if valid_rows else 0.0)
    return {
        "style_win_rate_valid": [round(_percentile(win_rates, 0.025), 6), round(_percentile(win_rates, 0.975), 6)],
        "style_margin_valid": [round(_percentile(margins, 0.025), 6), round(_percentile(margins, 0.975), 6)],
    }


def _group_summary(rows: list[dict[str, Any]], key_fn) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    summary = {}
    for key, group_rows in sorted(grouped.items()):
        valid_rows = [row for row in group_rows if _row_is_valid(row)]
        summary[key] = {
            "style_win_rate_valid": round(_mean([row["style_metrics"]["style_win_rate_case"] for row in valid_rows]), 6) if valid_rows else 0.0,
            "valid_rate": round(_mean([1.0 if _row_is_valid(row) else 0.0 for row in group_rows]), 6),
        }
    return summary


def build_leaderboard_row(summary: dict[str, Any]) -> dict[str, Any]:
    metrics_all = summary.get("metrics_all", {})
    metrics_valid = summary.get("metrics_valid", {})
    style_mimicry_score = round(
        float(metrics_valid.get("style_win_rate_mean", 0.0)) * float(metrics_all.get("valid_rate", 0.0)),
        6,
    )
    return {
        "benchmark_version": summary.get("benchmark_version"),
        "provider": summary.get("generator", {}).get("provider"),
        "model_name": summary.get("generator", {}).get("model_name"),
        "track": summary.get("track"),
        "split": summary.get("split"),
        "sample_count": summary.get("sample_count"),
        "valid_sample_count": summary.get("valid_sample_count"),
        "style_mimicry_score": style_mimicry_score,
        "style_win_rate_valid": round(float(metrics_valid.get("style_win_rate_mean", 0.0)), 6),
        "style_margin_valid": round(float(metrics_valid.get("style_margin_mean", 0.0)), 6),
        "top1_target_accuracy_valid": round(float(metrics_valid.get("top1_target_accuracy_mean", 0.0)), 6),
        "style_percentile_valid": round(float(metrics_valid.get("style_percentile_valid_mean", 0.0)), 6),
        "valid_rate": round(float(metrics_all.get("valid_rate", 0.0)), 6),
        "originality_pass_rate": round(float(metrics_all.get("originality_pass_rate", 0.0)), 6),
        "fluency_pass_rate": (
            round(float(metrics_all["fluency_pass_rate"]), 6) if "fluency_pass_rate" in metrics_all else None
        ),
        "conditioning_copy_free_rate": (
            round(float(metrics_all["conditioning_copy_free_rate"]), 6)
            if "conditioning_copy_free_rate" in metrics_all
            else None
        ),
        "target_evaluation_copy_free_rate": (
            round(float(metrics_all["target_evaluation_copy_free_rate"]), 6)
            if "target_evaluation_copy_free_rate" in metrics_all
            else None
        ),
        "full_target_book_copy_free_rate": (
            round(float(metrics_all["full_target_book_copy_free_rate"]), 6)
            if "full_target_book_copy_free_rate" in metrics_all
            else None
        ),
    }


def render_leaderboard_markdown(rows: list[dict[str, Any]]) -> str:
    columns = [
        ("rank", "rank"),
        ("model", "model_name"),
        ("provider", "provider"),
        ("track", "track"),
        ("split", "split"),
        ("style_mimicry", "style_mimicry_score"),
        ("style_win_valid", "style_win_rate_valid"),
        ("valid_rate", "valid_rate"),
        ("orig_pass", "originality_pass_rate"),
        ("fluency_pass", "fluency_pass_rate"),
    ]

    def _format_value(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    header = "| " + " | ".join(name for name, _ in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for index, row in enumerate(rows, start=1):
        ranked_row = dict(row)
        ranked_row["rank"] = index
        body.append("| " + " | ".join(_format_value(ranked_row.get(key)) for _, key in columns) + " |")
    return "\n".join([header, separator, *body]) if body else "\n".join([header, separator])


def aggregate_benchmark_results(input_path: str | Path, out_path: str | Path | None = None) -> dict[str, Any]:
    rows = load_jsonl(input_path)
    if not rows:
        raise ValueError("benchmark result file is empty")
    input_path = Path(input_path)
    config_candidates = [
        input_path.parent.parent / "benchmark_data" / "config_v1.json",
        input_path.parents[2] / "benchmark" / "manifests" / "config_v1.json" if len(input_path.parents) >= 3 else None,
        input_path.parents[2] / "benchmark" / "runtime_benchmark_config.json" if len(input_path.parents) >= 3 else None,
        Path("eval/benchmark_data/config_v1.json"),
    ]
    config = {"bootstrap_resamples": 1000}
    for candidate in config_candidates:
        if candidate is not None and candidate.exists():
            config = load_json(candidate)
            break
    summary = {
        "benchmark_version": rows[0]["benchmark_version"],
        "track": rows[0]["track"],
        "split": rows[0]["split"],
        "generator": {
            "provider": rows[0]["generator"]["provider"],
            "model_name": rows[0]["generator"]["model_name"],
        },
        **_collect_metrics(rows),
        "bootstrap_ci_95": _bootstrap_case_metrics(rows, resamples=int(config.get("bootstrap_resamples", 1000))),
        "by_prompt_family": _group_summary(rows, lambda row: row["prompt_id"].split(":")[1]),
        "by_target": _group_summary(rows, lambda row: row["target_id"]),
    }
    summary["leaderboard_row"] = build_leaderboard_row(summary)
    summary["leaderboard_markdown"] = render_leaderboard_markdown([summary["leaderboard_row"]])
    if out_path:
        write_json(out_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    aggregate_benchmark_results(args.input, args.out)


if __name__ == "__main__":
    main()
