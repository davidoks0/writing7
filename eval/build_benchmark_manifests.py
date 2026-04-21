from __future__ import annotations

import argparse
import hashlib
import math
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from corpus.layout import resolve_with_root
from corpus.manifests import build_corpus_manifests, freeze_corpus_splits, load_corpus_config
from corpus.passage_pools import build_passage_pools
from eval.benchmark_io import load_json, load_jsonl, write_json, write_jsonl
from eval.benchmark_schema import BenchmarkCase, BenchmarkTarget, PassageRecord, PromptRecord
from eval.distractors import select_distractor_targets
from eval.passage_sampling import sentence_windows_overlap
from eval.style_scoring import StyleScorer, compute_style_metrics


DEFAULT_CONFIG_PATH = Path("eval/benchmark_data/config_v1.json")
DEFAULT_PROMPTS_PATH = Path("eval/benchmark_data/prompts_v1.json")


def _load_config(
    benchmark_config: str | Path | None,
    *,
    benchmark_version: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    payload = load_json(benchmark_config or DEFAULT_CONFIG_PATH)
    if benchmark_version:
        payload["benchmark_version"] = benchmark_version
    if seed is not None:
        payload["builder_seed"] = seed
    return payload


def _derived_seed(benchmark_version: str, track: str, target_id: str, prompt_id: str, builder_seed: int) -> int:
    digest = hashlib.sha256(
        f"{benchmark_version}|{track}|{target_id}|{prompt_id}|{builder_seed}".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16)


def _resolve_distractor_policy(config: dict[str, Any]) -> dict[str, Any]:
    payload = config.get("distractor_policy", {})
    target_count = int(payload.get("target_count", 5))
    min_count = int(payload.get("min_count", target_count))
    book_track_same_author_policy = payload.get("book_track_same_author_policy", "prefer_other_author")
    if target_count < 1:
        raise ValueError("distractor_policy.target_count must be at least 1")
    if min_count < 1 or min_count > target_count:
        raise ValueError("distractor_policy.min_count must be between 1 and target_count")
    if book_track_same_author_policy not in {"allow", "prefer_other_author", "exclude"}:
        raise ValueError(
            "distractor_policy.book_track_same_author_policy must be 'allow', 'prefer_other_author', or 'exclude'"
        )
    return {
        "target_count": target_count,
        "min_count": min_count,
        "book_track_same_author_policy": book_track_same_author_policy,
    }


def _rank_passages(passages: list[dict[str, Any]], case_seed: int, *, preferred_words: int = 220, target_words: float | None = None) -> list[dict[str, Any]]:
    if target_words is None:
        target_words = preferred_words
    return sorted(
        passages,
        key=lambda passage: (
            abs(passage["word_count"] - target_words),
            int(hashlib.sha1(f"{case_seed}|{passage['passage_id']}".encode("utf-8")).hexdigest()[:16], 16),
            passage["passage_id"],
        ),
    )


def _passes_non_overlap(candidate: dict[str, Any], chosen: list[dict[str, Any]], *, min_gap: int = 3) -> bool:
    candidate_record = PassageRecord.from_dict(candidate)
    for row in chosen:
        if row["book_id"] != candidate["book_id"]:
            continue
        if sentence_windows_overlap(candidate_record, PassageRecord.from_dict(row), min_gap=min_gap):
            return False
    return True


def _select_passages(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    existing: list[dict[str, Any]] | None = None,
    min_gap: int = 3,
    prefer_distinct_buckets: bool = True,
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    existing = existing or []
    used_buckets = {row["region_bucket"] for row in existing}
    candidate_order = {row["passage_id"]: index for index, row in enumerate(candidates)}

    def append_candidates(require_new_bucket: bool) -> None:
        for candidate in candidates:
            if candidate in chosen or candidate in existing:
                continue
            if require_new_bucket and candidate["region_bucket"] in used_buckets:
                continue
            if not _passes_non_overlap(candidate, existing + chosen, min_gap=min_gap):
                continue
            chosen.append(candidate)
            used_buckets.add(candidate["region_bucket"])
            if len(chosen) >= count:
                return

    if prefer_distinct_buckets:
        append_candidates(True)
    if len(chosen) < count:
        append_candidates(False)
    if len(chosen) >= count:
        return chosen[:count]

    search_candidates = candidates[: min(len(candidates), max(24, count * 8))]
    best: list[dict[str, Any]] = chosen[:]
    best_score = (
        len(best),
        len({row["region_bucket"] for row in best}),
        -sum(candidate_order[row["passage_id"]] for row in best),
    )

    def consider(selection: list[dict[str, Any]]) -> None:
        nonlocal best, best_score
        score = (
            len(selection),
            len({row["region_bucket"] for row in selection}) if prefer_distinct_buckets else 0,
            -sum(candidate_order[row["passage_id"]] for row in selection),
        )
        if score > best_score:
            best = selection[:]
            best_score = score

    def search(start: int, selection: list[dict[str, Any]]) -> None:
        consider(selection)
        if len(selection) >= count:
            return
        remaining_slots = count - len(selection)
        for index in range(start, len(search_candidates)):
            if len(search_candidates) - index < remaining_slots:
                return
            candidate = search_candidates[index]
            if candidate in selection or candidate in existing:
                continue
            if not _passes_non_overlap(candidate, existing + selection, min_gap=min_gap):
                continue
            selection.append(candidate)
            search(index + 1, selection)
            selection.pop()

    search(0, [])
    return best[:count]


def _bundle_target_passages(
    target: BenchmarkTarget,
    *,
    case_seed: int,
    passages_by_book: dict[str, list[dict[str, Any]]],
    target_eval_word_goal: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if target.track == "author":
        conditioning_book_ids = target.conditioning_book_ids or []
        eval_book_id = target.evaluation_book_id
        conditioning: list[dict[str, Any]] = []
        for book_id in conditioning_book_ids[:2]:
            conditioning.extend(
                _select_passages(_rank_passages(passages_by_book[book_id], case_seed), count=1, existing=conditioning)
            )
        combined_remaining = []
        for book_id in conditioning_book_ids[:2]:
            combined_remaining.extend(passages_by_book[book_id])
        combined_ranked = _rank_passages(combined_remaining, case_seed)
        extra = _select_passages(combined_ranked, count=1, existing=conditioning)
        conditioning.extend(extra)
        evaluation = _select_passages(
            _rank_passages(passages_by_book[eval_book_id], case_seed, target_words=target_eval_word_goal),
            count=4,
            existing=[],
        )
        return {"conditioning": conditioning[:3], "evaluation": evaluation[:4]}

    book_id = target.book_id
    conditioning = _select_passages(_rank_passages(passages_by_book[book_id], case_seed), count=3)
    evaluation = _select_passages(
        _rank_passages(passages_by_book[book_id], case_seed, target_words=target_eval_word_goal),
        count=4,
        existing=conditioning,
    )
    return {"conditioning": conditioning[:3], "evaluation": evaluation[:4]}


def _build_author_targets(
    split_payload: dict[str, Any],
    *,
    benchmark_version: str,
    builder_seed: int,
    author_books: dict[str, list[str]],
) -> list[BenchmarkTarget]:
    targets: list[BenchmarkTarget] = []
    for author_id in sorted(split_payload["author_ids"]):
        candidate_books = author_books.get(author_id, [])
        if len(candidate_books) < 3:
            continue
        ranked = sorted(
            candidate_books,
            key=lambda book_id: (
                hashlib.sha1(f"{benchmark_version}|author|{author_id}|{book_id}|{builder_seed}".encode("utf-8")).hexdigest(),
                book_id,
            ),
        )
        targets.append(
            BenchmarkTarget(
                target_id=author_id,
                track="author",
                author_id=author_id,
                conditioning_book_ids=ranked[:2],
                evaluation_book_id=ranked[2],
            ).validate()
        )
    return targets


def _build_book_targets(split_payload: dict[str, Any], *, books: dict[str, dict[str, Any]]) -> list[BenchmarkTarget]:
    targets: list[BenchmarkTarget] = []
    for book_id in sorted(split_payload["book_track_book_ids"]):
        book = books[book_id]
        if not book.get("eligible_book_track"):
            continue
        targets.append(
            BenchmarkTarget(
                target_id=book_id,
                track="book",
                author_id=book["author_id"],
                book_id=book_id,
            ).validate()
        )
    return targets


def _target_book_meta(target: BenchmarkTarget, books: dict[str, dict[str, Any]]) -> dict[str, Any]:
    book_id = target.evaluation_book_id if target.track == "author" else target.book_id
    return books[book_id]


def _case_payload(
    target: BenchmarkTarget,
    prompt: PromptRecord,
    *,
    split: str,
    benchmark_version: str,
    builder_seed: int,
    generation_profile_id: str,
    sample_seeds: list[int],
    passages_by_book: dict[str, list[dict[str, Any]]],
    candidate_targets: list[BenchmarkTarget],
    books: dict[str, dict[str, Any]],
    scorer: StyleScorer | None,
    distractor_policy: dict[str, Any],
) -> tuple[BenchmarkCase, dict[str, dict[str, list[dict[str, Any]]]]]:
    case_seed = _derived_seed(benchmark_version, target.track, target.target_id, prompt.prompt_id, builder_seed)
    target_bundle = _bundle_target_passages(target, case_seed=case_seed, passages_by_book=passages_by_book)
    if len(target_bundle["conditioning"]) != 3 or len(target_bundle["evaluation"]) != 4:
        raise ValueError(
            f"target {target.target_id} does not have the required 3 conditioning and 4 evaluation passages"
        )
    all_bundles: dict[str, dict[str, list[dict[str, Any]]]] = {
        target.target_id: target_bundle,
    }
    target_meta = _target_book_meta(target, books)
    target_eval_word_goal = statistics.median(passage["word_count"] for passage in target_bundle["evaluation"]) if target_bundle["evaluation"] else 220
    for candidate in candidate_targets:
        if candidate.target_id == target.target_id:
            continue
        all_bundles[candidate.target_id] = _bundle_target_passages(
            candidate,
            case_seed=case_seed,
            passages_by_book=passages_by_book,
            target_eval_word_goal=target_eval_word_goal,
        )

    distractor_targets = select_distractor_targets(
        case_seed=case_seed,
        track=target.track,
        target=target,
        candidate_targets=candidate_targets,
        target_book_meta=target_meta,
        target_eval_passages=[PassageRecord.from_dict(row) for row in target_bundle["evaluation"]],
        passages_by_target={
            target_id: {
                "conditioning": [PassageRecord.from_dict(row) for row in bundle["conditioning"]],
                "evaluation": [PassageRecord.from_dict(row) for row in bundle["evaluation"]],
            }
            for target_id, bundle in all_bundles.items()
        },
        books_by_id=books,
        scorer=scorer,
        target_count=distractor_policy["target_count"],
        book_track_same_author_policy=distractor_policy["book_track_same_author_policy"],
    )
    if len(distractor_targets) < distractor_policy["min_count"]:
        raise ValueError(
            f"target {target.target_id} produced only {len(distractor_targets)} distractors, below the required minimum "
            f"{distractor_policy['min_count']}"
        )
    distractor_ids = [candidate.target_id for candidate in distractor_targets]
    for candidate in distractor_targets:
        if len(all_bundles[candidate.target_id]["evaluation"]) != 4:
            raise ValueError(f"distractor target {candidate.target_id} does not have 4 evaluation passages")
    distractor_passage_ids = {
        candidate.target_id: [row["passage_id"] for row in all_bundles[candidate.target_id]["evaluation"]]
        for candidate in distractor_targets
    }
    prompt_slug = prompt.prompt_id.split(":")[-2] + "_" + prompt.prompt_id.split(":")[-1]
    target_slug = target.target_id.split(":")[-1]
    case = BenchmarkCase(
        case_id=f"case:{target.track}:{split}:{target_slug}:{prompt_slug}",
        benchmark_version=benchmark_version,
        track=target.track,
        split=split,
        target_id=target.target_id,
        prompt_id=prompt.prompt_id,
        conditioning_passage_ids=[row["passage_id"] for row in target_bundle["conditioning"]],
        evaluation_passage_ids=[row["passage_id"] for row in target_bundle["evaluation"]],
        distractor_target_ids=distractor_ids,
        distractor_passage_ids_by_target=distractor_passage_ids,
        generation_profile_id=generation_profile_id,
        sample_seeds=sample_seeds,
    ).validate()
    return case, all_bundles


def _summary_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "raw_values": []}
    ordered = sorted(values)
    def percentile(pct: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = pct * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight
    return {
        "p1": round(percentile(0.01), 6),
        "p5": round(percentile(0.05), 6),
        "p10": round(percentile(0.10), 6),
        "p25": round(percentile(0.25), 6),
        "p50": round(percentile(0.50), 6),
        "p75": round(percentile(0.75), 6),
        "p90": round(percentile(0.90), 6),
        "p95": round(percentile(0.95), 6),
        "p99": round(percentile(0.99), 6),
        "mean": round(sum(ordered) / len(ordered), 6),
        "median": round(statistics.median(ordered), 6),
        "count": len(ordered),
        "raw_values": [round(value, 6) for value in ordered],
    }


def _build_reference_distribution(
    *,
    track: str,
    benchmark_version: str,
    targets: list[BenchmarkTarget],
    prompts: list[PromptRecord],
    books: dict[str, dict[str, Any]],
    passages_by_book: dict[str, list[dict[str, Any]]],
    scorer: StyleScorer | None,
    builder_seed: int,
    distractor_policy: dict[str, Any],
) -> dict[str, Any]:
    if scorer is None or not targets or not prompts:
        return {
            "artifact_type": "benchmark_reference_distributions",
            "artifact_version": f"benchmark_reference_distributions_{track}_v1",
            "benchmark_version": benchmark_version,
            "track": track,
            "score_basis": "calibrated_or_score_0_1",
            "source_split": "dev",
            "prompt_count": len(prompts),
            "global": {
                "target_similarity": _summary_stats([]),
                "distractor_similarity": _summary_stats([]),
                "style_margin": _summary_stats([]),
            },
        }

    target_scores: list[float] = []
    distractor_scores: list[float] = []
    style_margins: list[float] = []
    for prompt in prompts:
        for target in targets:
            try:
                case, bundles = _case_payload(
                    target,
                    prompt,
                    split="dev",
                    benchmark_version=benchmark_version,
                    builder_seed=builder_seed,
                    generation_profile_id="leaderboard_v1",
                    sample_seeds=[11],
                    passages_by_book=passages_by_book,
                    candidate_targets=targets,
                    books=books,
                    scorer=scorer,
                    distractor_policy=distractor_policy,
                )
            except ValueError:
                continue
            target_eval_texts = [
                next(row["text"] for row in bundles[target.target_id]["evaluation"] if row["passage_id"] == passage_id)
                for passage_id in case.evaluation_passage_ids
            ]
            distractor_map = {
                distractor_id: [row["text"] for row in bundles[distractor_id]["evaluation"]]
                for distractor_id in case.distractor_target_ids
            }
            for index, hypothesis in enumerate(target_eval_texts):
                references = [text for offset, text in enumerate(target_eval_texts) if offset != index]
                metrics = compute_style_metrics(scorer, hypothesis, references, distractor_map, reference_distribution=None)
                target_scores.append(metrics["target_similarity_mean"])
                distractor_scores.extend(metrics["distractor_similarity_means"].values())
                style_margins.append(metrics["style_margin_case"])
    return {
        "artifact_type": "benchmark_reference_distributions",
        "artifact_version": f"benchmark_reference_distributions_{track}_v1",
        "benchmark_version": benchmark_version,
        "track": track,
        "score_basis": "calibrated_or_score_0_1",
        "source_split": "dev",
        "prompt_count": len(prompts),
        "global": {
            "target_similarity": _summary_stats(target_scores),
            "distractor_similarity": _summary_stats(distractor_scores),
            "style_margin": _summary_stats(style_margins),
        },
    }


def _load_book_text(book: dict[str, Any], *, corpus_root: Path, cache: dict[str, str]) -> str:
    book_id = book["book_id"]
    if book_id in cache:
        return cache[book_id]
    candidate_path = None
    if book.get("clean_path"):
        candidate_path = resolve_with_root(corpus_root, book["clean_path"])
    elif book.get("source_path"):
        candidate_path = resolve_with_root(corpus_root, book["source_path"])
    text = candidate_path.read_text(encoding="utf-8") if candidate_path is not None and candidate_path.exists() else ""
    cache[book_id] = text
    return text


def _build_target_reference_groups(
    targets: list[BenchmarkTarget],
    *,
    books: dict[str, dict[str, Any]],
    corpus_root: Path,
) -> dict[str, dict[str, list[str]]]:
    cache: dict[str, str] = {}
    payload: dict[str, dict[str, list[str]]] = {}
    for target in targets:
        if target.track == "author":
            evaluation_book = books[target.evaluation_book_id]
            conditioning_books = [books[book_id] for book_id in (target.conditioning_book_ids or [])]
            payload[target.target_id] = {
                "full_target_book": [_load_book_text(evaluation_book, corpus_root=corpus_root, cache=cache)],
                "conditioning_books": [
                    _load_book_text(book, corpus_root=corpus_root, cache=cache)
                    for book in conditioning_books
                ],
            }
        else:
            book = books[target.book_id]
            payload[target.target_id] = {
                "full_target_book": [_load_book_text(book, corpus_root=corpus_root, cache=cache)],
            }
    return payload


def build_benchmark_manifests(
    *,
    books_manifest: str | Path,
    out_dir: str | Path,
    benchmark_config: str | Path | None = None,
    benchmark_version: str | None = None,
    seed: int | None = None,
    corpus_config: str | Path | None = None,
    split_root: str | Path | None = None,
    scorer_dir: str | Path | None = None,
) -> dict[str, str]:
    config = _load_config(benchmark_config, benchmark_version=benchmark_version, seed=seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    books_input_rows = load_jsonl(books_manifest)
    if not books_input_rows:
        raise ValueError("books manifest is empty")

    first_row = books_input_rows[0]
    manifest_path = Path(books_manifest)
    manifest_corpus_root = manifest_path.parent.parent if manifest_path.name == "books_manifest_v1.jsonl" else None
    first_clean_path = (
        resolve_with_root(manifest_corpus_root, first_row["clean_path"])
        if manifest_corpus_root is not None and first_row.get("clean_path")
        else Path(first_row["clean_path"])
        if first_row.get("clean_path")
        else None
    )
    if "clean_path" not in first_row or first_clean_path is None or not first_clean_path.exists():
        corpus_output_root = Path(config.get("corpus_output_root", out_dir.parent / "corpus"))
        corpus_payload = load_corpus_config(corpus_config or {})
        corpus_payload["input_books_manifest"] = Path(books_manifest).as_posix()
        corpus_payload["output_root"] = corpus_output_root.as_posix()
        if seed is not None:
            corpus_payload["builder_seed"] = seed
        build_corpus_manifests(corpus_payload)
        freeze_corpus_splits(corpus_payload)
        build_passage_pools(corpus_payload)
        corpus_root = corpus_output_root
        books_manifest_path = corpus_root / "meta" / "books_manifest_v1.jsonl"
        split_root_path = corpus_root / "splits"
    else:
        books_manifest_path = Path(books_manifest)
        corpus_root = books_manifest_path.parent.parent
        split_root_path = Path(split_root) if split_root else corpus_root / "splits"

    books = {row["book_id"]: row for row in load_jsonl(books_manifest_path)}
    authors = {row["author_id"]: row for row in load_jsonl(corpus_root / "meta" / "authors_manifest_v1.jsonl")}
    prompts = [PromptRecord.from_dict(row) for row in load_json(config.get("prompt_bank_path", DEFAULT_PROMPTS_PATH))]
    prompts = sorted(prompts, key=lambda row: row.prompt_id)
    builder_seed = int(config.get("builder_seed", 42))
    distractor_policy = _resolve_distractor_policy(config)

    author_books = {
        author_id: [book_id for book_id in row["eligible_book_ids"] if books[book_id]["eligible_author_track"]]
        for author_id, row in authors.items()
    }
    passages_by_book: dict[str, list[dict[str, Any]]] = {}
    for book_id, book in books.items():
        pool_path = corpus_root / "meta" / "passage_pools_v1" / book["author_slug"] / f"{book_id.replace(':', '_')}.jsonl"
        if pool_path.exists():
            passages_by_book[book_id] = load_jsonl(pool_path)

    scorer = StyleScorer(scorer_dir) if scorer_dir else None
    generation_profile_id = next(iter(config["generation_profiles"].keys()))
    sample_seeds = list(config["generation_profiles"][generation_profile_id]["sample_seeds"])

    outputs: dict[str, str] = {}
    all_track_passages: dict[str, dict[str, dict[str, Any]]] = {"author": {}, "book": {}}
    target_manifests: dict[tuple[str, str], list[BenchmarkTarget]] = {}

    for split_name in ("dev", "test"):
        split_payload = load_json(split_root_path / f"benchmark_{split_name}_v1.json")
        author_targets = _build_author_targets(
            split_payload,
            benchmark_version=config["benchmark_version"],
            builder_seed=builder_seed,
            author_books=author_books,
        )
        book_targets = _build_book_targets(split_payload, books=books)
        target_manifests[(split_name, "author")] = author_targets
        target_manifests[(split_name, "book")] = book_targets

        for track, targets in (("author", author_targets), ("book", book_targets)):
            target_file = out_dir / f"benchmark_{split_name}_targets_{track}_v1.json"
            write_json(
                target_file,
                {
                    "artifact_type": "benchmark_targets",
                    "benchmark_version": config["benchmark_version"],
                    "track": track,
                    "split": split_name,
                    "targets": [asdict(target) for target in targets],
                },
            )
            outputs[f"targets_{track}_{split_name}"] = target_file.as_posix()
            reference_groups_file = out_dir / f"reference_groups_{track}_{split_name}_v1.json"
            write_json(
                reference_groups_file,
                {
                    "artifact_type": "benchmark_reference_groups",
                    "benchmark_version": config["benchmark_version"],
                    "track": track,
                    "split": split_name,
                    "target_reference_groups": _build_target_reference_groups(
                        targets,
                        books=books,
                        corpus_root=corpus_root,
                    ),
                },
            )
            outputs[f"reference_groups_{track}_{split_name}"] = reference_groups_file.as_posix()

            cases: list[dict[str, Any]] = []
            for target in targets:
                for prompt in prompts:
                    try:
                        case, bundles = _case_payload(
                            target,
                            prompt,
                            split=split_name,
                            benchmark_version=config["benchmark_version"],
                            builder_seed=builder_seed,
                            generation_profile_id=generation_profile_id,
                            sample_seeds=sample_seeds,
                            passages_by_book=passages_by_book,
                            candidate_targets=targets,
                            books=books,
                            scorer=scorer,
                            distractor_policy=distractor_policy,
                        )
                    except ValueError:
                        continue
                    cases.append(asdict(case))
                    for target_id, bundle in bundles.items():
                        for record in bundle["conditioning"] + bundle["evaluation"]:
                            all_track_passages[track][record["passage_id"]] = record
            cases.sort(key=lambda row: (row["target_id"], row["prompt_id"]))
            case_file = out_dir / f"cases_{track}_{split_name}_v1.jsonl"
            write_jsonl(case_file, cases)
            outputs[f"cases_{track}_{split_name}"] = case_file.as_posix()

    write_jsonl(out_dir / "books_manifest.jsonl", sorted(books.values(), key=lambda row: row["book_id"]))
    write_json(
        out_dir / "scorer_train_authors_v1.json",
        load_json(split_root_path / "scorer_train_v1.json"),
    )
    for track in ("author", "book"):
        write_jsonl(
            out_dir / f"passages_{track}_v1.jsonl",
            sorted(all_track_passages[track].values(), key=lambda row: row["passage_id"]),
        )
    write_json(out_dir / "config_v1.json", config)
    write_json(out_dir / "prompts_v1.json", [asdict(prompt) for prompt in prompts])
    (out_dir / "VERSION").write_text(config["benchmark_version"] + "\n", encoding="utf-8")

    for track in ("author", "book"):
        reference_distribution = _build_reference_distribution(
            track=track,
            benchmark_version=config["benchmark_version"],
            targets=target_manifests[("dev", track)],
            prompts=prompts,
            books=books,
            passages_by_book=passages_by_book,
            scorer=scorer,
            builder_seed=builder_seed,
            distractor_policy=distractor_policy,
        )
        reference_path = out_dir / f"benchmark_reference_distributions_{track}_v1.json"
        write_json(reference_path, reference_distribution)
        outputs[f"reference_distribution_{track}"] = reference_path.as_posix()

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--books-manifest")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--benchmark-version")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--benchmark-config")
    parser.add_argument("--corpus-config")
    parser.add_argument("--split-root")
    parser.add_argument("--scorer-dir")
    args = parser.parse_args()

    if not args.books_manifest:
        raise SystemExit("--books-manifest is required")
    books_manifest = args.books_manifest
    build_benchmark_manifests(
        books_manifest=books_manifest,
        out_dir=args.out_dir,
        benchmark_config=args.benchmark_config,
        benchmark_version=args.benchmark_version,
        seed=args.seed,
        corpus_config=args.corpus_config,
        split_root=args.split_root,
        scorer_dir=args.scorer_dir,
    )


if __name__ == "__main__":
    main()
