from __future__ import annotations

import hashlib
import statistics
from typing import Any

from eval.benchmark_schema import BenchmarkTarget, PassageRecord
from eval.style_scoring import StyleScorer, cosine_similarity, mean_pool


def median_word_count(passages: list[PassageRecord]) -> float:
    if not passages:
        return 0.0
    return statistics.median(passage.word_count for passage in passages)


def _normalized_labels(values: list[str] | None) -> set[str]:
    return {value.strip().lower() for value in values or [] if value and value.strip()}


def metadata_match_score(
    target_book: dict[str, Any],
    candidate_book: dict[str, Any],
    target_eval_passages: list[PassageRecord],
    candidate_eval_passages: list[PassageRecord],
) -> int:
    target_period = target_book.get("period_bucket", "unknown")
    candidate_period = candidate_book.get("period_bucket", "unknown")
    target_genre = target_book.get("genre", "unknown")
    candidate_genre = candidate_book.get("genre", "unknown")
    same_period = int(target_period != "unknown" and candidate_period != "unknown" and target_period == candidate_period)
    same_genre = int(target_genre != "unknown" and candidate_genre != "unknown" and target_genre == candidate_genre)
    same_length = int(abs(median_word_count(target_eval_passages) - median_word_count(candidate_eval_passages)) <= 40)
    target_bookshelves = _normalized_labels(target_book.get("bookshelves"))
    candidate_bookshelves = _normalized_labels(candidate_book.get("bookshelves"))
    target_subjects = _normalized_labels(target_book.get("subjects"))
    candidate_subjects = _normalized_labels(candidate_book.get("subjects"))
    bookshelf_overlap = int(bool(target_bookshelves & candidate_bookshelves))
    subject_overlap = int(bool(target_subjects & candidate_subjects))
    return (2 * same_period) + (2 * same_genre) + same_length + bookshelf_overlap + subject_overlap


def seeded_candidate_key(case_seed: int, candidate_target_id: str) -> tuple[int, str]:
    digest = hashlib.sha1(f"{case_seed}|{candidate_target_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16), candidate_target_id


def select_distractor_targets(
    *,
    case_seed: int,
    track: str,
    target: BenchmarkTarget,
    candidate_targets: list[BenchmarkTarget],
    target_book_meta: dict[str, Any],
    target_eval_passages: list[PassageRecord],
    passages_by_target: dict[str, dict[str, list[PassageRecord]]],
    books_by_id: dict[str, dict[str, Any]],
    scorer: StyleScorer | None = None,
    target_count: int = 5,
    book_track_same_author_policy: str = "prefer_other_author",
) -> list[BenchmarkTarget]:
    if target_count < 1:
        raise ValueError("target_count must be at least 1")
    if book_track_same_author_policy not in {"allow", "prefer_other_author", "exclude"}:
        raise ValueError("book_track_same_author_policy must be 'allow', 'prefer_other_author', or 'exclude'")
    eligible_candidates = []
    for candidate in candidate_targets:
        if candidate.target_id == target.target_id:
            continue
        same_author = candidate.author_id == target.author_id
        if track == "author" and same_author:
            continue
        if track == "book" and same_author and book_track_same_author_policy == "exclude":
            continue
        candidate_book_id = candidate.evaluation_book_id if candidate.track == "author" else candidate.book_id
        candidate_book_meta = books_by_id[candidate_book_id]
        candidate_eval = passages_by_target[candidate.target_id]["evaluation"]
        meta_score = metadata_match_score(target_book_meta, candidate_book_meta, target_eval_passages, candidate_eval)
        eligible_candidates.append((same_author, meta_score, seeded_candidate_key(case_seed, candidate.target_id), candidate))

    if track == "book" and book_track_same_author_policy == "prefer_other_author":
        eligible_candidates.sort(key=lambda item: (1 if item[0] else 0, -item[1], item[2]))
    else:
        eligible_candidates.sort(key=lambda item: (-item[1], item[2]))

    matched_pool = [candidate for _, _, _, candidate in eligible_candidates[:20]]
    matched_count = min(3, target_count)
    hard_count = min(2, max(0, target_count - matched_count))
    matched = matched_pool[:matched_count]
    remaining = [candidate for candidate in matched_pool if candidate not in matched]
    if hard_count == 0:
        return matched
    if not scorer or not remaining:
        return matched + remaining[:hard_count]

    target_profile = mean_pool(scorer.embed(passage.text) for passage in passages_by_target[target.target_id]["conditioning"])
    ranked_hard = []
    for candidate in remaining:
        candidate_profile = mean_pool(scorer.embed(passage.text) for passage in passages_by_target[candidate.target_id]["conditioning"])
        ranked_hard.append((cosine_similarity(target_profile, candidate_profile), seeded_candidate_key(case_seed, candidate.target_id), candidate))
    ranked_hard.sort(key=lambda item: (-item[0], item[1]))
    hard = [candidate for _, _, candidate in ranked_hard[:hard_count]]
    selected = matched + hard
    if len(selected) < min(target_count, len(matched_pool)):
        for candidate in matched_pool:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= min(target_count, len(matched_pool)):
                break
    return selected[:target_count]
