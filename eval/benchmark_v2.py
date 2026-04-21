from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.benchmark_io import append_jsonl, load_json, load_jsonl
from eval.benchmark_schema import BenchmarkCase, PassageRecord, PromptRecord
from eval.fluency import compute_fluency_metrics
from eval.llm_clients import GenerationRequest, generate_text
from eval.originality import compute_originality_metrics
from eval.style_scoring import StyleScorer, compute_style_metrics


SYSTEM_PROMPT = (
    "You are a careful creative writer. Imitate stylistic features of the reference passages, "
    "including voice, rhythm, syntax, diction, and paragraph movement, while writing entirely new content. "
    "Do not reuse names, places, quotations, or distinctive phrases from the references. "
    "Do not mention the references or the benchmark. Output only the new prose."
)

USER_PROMPT_TEMPLATE = """You will receive style references and a content prompt.

Write an original passage in prose.

Requirements:
- Follow the content prompt.
- Match the style of the references.
- Do not copy phrases, names, places, or plot specifics from the references.
- Keep the response between 500 and 800 words.

STYLE REFERENCES
[Reference 1]
{conditioning_text_1}

[Reference 2]
{conditioning_text_2}

[Reference 3]
{conditioning_text_3}

CONTENT PROMPT
{prompt_text}
"""


def _utc_run_id(model: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    model_slug = model.replace(":", "_").replace("/", "_").replace(".", "_")
    return f"run:{timestamp}:{model_slug}"


def _load_passages(path: str | Path) -> dict[str, dict[str, Any]]:
    return {row["passage_id"]: row for row in load_jsonl(path)}


def _render_user_prompt(conditioning_texts: list[str], prompt_text: str) -> str:
    padded = conditioning_texts + ["", "", ""]
    return USER_PROMPT_TEMPLATE.format(
        conditioning_text_1=padded[0],
        conditioning_text_2=padded[1],
        conditioning_text_3=padded[2],
        prompt_text=prompt_text,
    ).strip()


def _load_target_reference_groups(cases_file: Path, *, track: str, split: str) -> dict[str, dict[str, list[str]]]:
    reference_groups_path = cases_file.parent / f"reference_groups_{track}_{split}_v1.json"
    if not reference_groups_path.exists():
        return {}
    payload = load_json(reference_groups_path)
    return payload.get("target_reference_groups", {})


def run_benchmark(
    *,
    track: str,
    split: str,
    cases_path: str | Path,
    prompts_path: str | Path,
    model: str,
    model_dir: str | Path,
    out_path: str | Path,
    config_path: str | Path | None = None,
    passages_path: str | Path | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    limit: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
    stream_print: bool = False,
    fail_fast: bool = False,
) -> str:
    cases_file = Path(cases_path)
    prompts_file = Path(prompts_path)
    out_file = Path(out_path)
    if overwrite and out_file.exists():
        out_file.unlink()

    config = load_json(config_path or (prompts_file.parent / "config_v1.json"))
    passages_file = Path(passages_path or (cases_file.parent / f"passages_{track}_v1.jsonl"))
    passages = _load_passages(passages_file)
    prompts = {row["prompt_id"]: PromptRecord.from_dict(row) for row in load_json(prompts_file)}
    cases = [BenchmarkCase.from_dict(row) for row in load_jsonl(cases_file)]
    if limit is not None:
        cases = cases[:limit]
    scorer = StyleScorer(model_dir)
    reference_distribution_path = cases_file.parent / f"benchmark_reference_distributions_{track}_v1.json"
    reference_distribution = load_json(reference_distribution_path) if reference_distribution_path.exists() else None
    target_reference_groups = _load_target_reference_groups(cases_file, track=track, split=split)
    generation_profile = config["generation_profiles"].get("leaderboard_v1", {})
    request_temperature = temperature if temperature is not None else generation_profile.get("temperature", 0.8)
    request_top_p = top_p if top_p is not None else generation_profile.get("top_p", 0.95)
    request_max_tokens = max_tokens if max_tokens is not None else generation_profile.get("max_tokens", 900)

    existing_pairs = set()
    if resume and out_file.exists():
        for row in load_jsonl(out_file):
            existing_pairs.add((row["case_id"], row["sample_index"]))

    run_id = _utc_run_id(model)
    written_rows = []
    for case in cases:
        prompt = prompts[case.prompt_id]
        conditioning_texts = [passages[passage_id]["text"] for passage_id in case.conditioning_passage_ids]
        evaluation_texts = [passages[passage_id]["text"] for passage_id in case.evaluation_passage_ids]
        distractor_texts = {
            target_id: [passages[passage_id]["text"] for passage_id in passage_ids]
            for target_id, passage_ids in case.distractor_passage_ids_by_target.items()
        }
        if not distractor_texts:
            raise ValueError(f"benchmark case {case.case_id} has no distractor targets")
        user_prompt = _render_user_prompt(conditioning_texts, prompt.text)
        for sample_index, sample_seed in enumerate(case.sample_seeds):
            if (case.case_id, sample_index) in existing_pairs:
                continue
            response = generate_text(
                GenerationRequest(
                    model=model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=request_temperature,
                    top_p=request_top_p,
                    max_tokens=request_max_tokens,
                    seed=sample_seed,
                )
            )
            if not response.ok and fail_fast:
                raise RuntimeError(response.error_message or "generation failed")
            output_text = response.output_text
            style_metrics = compute_style_metrics(scorer, output_text, evaluation_texts, distractor_texts, reference_distribution)
            originality_metrics = compute_originality_metrics(
                output_text,
                conditioning_texts,
                reference_groups={
                    "target_evaluation": evaluation_texts,
                    **target_reference_groups.get(case.target_id, {}),
                },
                char_threshold=config["originality_thresholds"]["char_8gram_overlap_max"],
                lcs_threshold=config["originality_thresholds"]["token_lcs_ratio_max"],
                joint_char_threshold=config["originality_thresholds"]["joint_char_overlap_threshold"],
                joint_lcs_threshold=config["originality_thresholds"]["joint_lcs_threshold"],
                entity_sequence_threshold=int(config["originality_thresholds"].get("entity_sequence_threshold", 1)),
                rare_capitalized_threshold=int(config["originality_thresholds"].get("rare_capitalized_threshold", 2)),
            )
            fluency_metrics = compute_fluency_metrics(
                output_text,
                min_words_valid=config["fluency_thresholds"]["min_words_valid"],
                max_words_valid=config["fluency_thresholds"]["max_words_valid"],
                max_repetition_rate_6gram=config["fluency_thresholds"]["max_repetition_rate_6gram"],
            )
            valid_flags = {
                "originality_pass": bool(originality_metrics["originality_pass"]),
                "fluency_pass": bool(fluency_metrics["fluency_pass"]),
            }
            valid_flags["valid"] = all(valid_flags.values())

            row = {
                "run_id": run_id,
                "sample_id": f"{run_id}:{case.case_id}:{sample_index}",
                "benchmark_version": case.benchmark_version,
                "track": case.track,
                "split": case.split,
                "case_id": case.case_id,
                "target_id": case.target_id,
                "prompt_id": case.prompt_id,
                "sample_index": sample_index,
                "sample_seed": sample_seed,
                "generator": {
                    "provider": response.provider,
                    "model_name": response.model_name,
                    "model_version": response.model_version,
                    "temperature": request_temperature,
                    "top_p": request_top_p,
                    "max_tokens": request_max_tokens,
                    "seed_supported": response.seed_supported,
                    "finish_reason": response.finish_reason,
                    "latency_ms": round(response.latency_ms or 0.0, 3),
                    "error_type": response.error_type,
                    "error_message": response.error_message,
                },
                "scorer": {
                    "name": scorer.manifest.get("model_name", "style_scorer"),
                    "model_dir": Path(model_dir).as_posix(),
                    "reference_distribution_version": (reference_distribution or {}).get("artifact_version"),
                },
                "prompt_text": prompt.text,
                "conditioning_texts": conditioning_texts,
                "output_text": output_text,
                "style_metrics": style_metrics,
                "originality_metrics": originality_metrics,
                "fluency_metrics": fluency_metrics,
                "valid_flags": valid_flags,
            }
            written_rows.append(row)
            if stream_print:
                print(f"{case.case_id} sample {sample_index}: valid={valid_flags['valid']} style={style_metrics['style_margin_case']}")

    append_jsonl(out_file, written_rows)
    return out_file.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config")
    parser.add_argument("--passages")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stream-print", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    run_benchmark(
        track=args.track,
        split=args.split,
        cases_path=args.cases,
        prompts_path=args.prompts,
        model=args.model,
        model_dir=args.model_dir,
        out_path=args.out,
        config_path=args.config,
        passages_path=args.passages,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        limit=args.limit,
        resume=args.resume,
        overwrite=args.overwrite,
        stream_print=args.stream_print,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    main()
