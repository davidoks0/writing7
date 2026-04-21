from __future__ import annotations

import re
import string
from collections.abc import Iterable, Mapping


TOKEN_RE = re.compile(r"\s+")
CAPITALIZED_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z'-]+\b")
ENTITY_SPAN_RE = re.compile(r"\b(?:[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+)*)\b")
COMMON_CAPITALIZED = {
    "A",
    "An",
    "And",
    "April",
    "August",
    "December",
    "February",
    "Friday",
    "He",
    "Her",
    "His",
    "I",
    "If",
    "In",
    "It",
    "January",
    "July",
    "June",
    "March",
    "May",
    "Monday",
    "Mr",
    "Mrs",
    "November",
    "October",
    "Saturday",
    "She",
    "Sunday",
    "The",
    "Their",
    "There",
    "They",
    "Thursday",
    "Tuesday",
    "Wednesday",
}


def normalize_for_char_ngrams(text: str) -> str:
    return " ".join(text.lower().split()).strip()


def normalize_for_token_overlap(text: str) -> str:
    lowered = text.lower()
    translation = str.maketrans({char: " " for char in string.punctuation})
    return TOKEN_RE.sub(" ", lowered.translate(translation)).strip()


def char_ngrams(text: str, n: int = 8) -> set[str]:
    normalized = normalize_for_char_ngrams(text)
    if len(normalized) < n:
        return {normalized} if normalized else set()
    return {normalized[index : index + n] for index in range(0, len(normalized) - n + 1)}


def char_8gram_overlap(hypothesis: str, reference: str) -> float:
    hypothesis_ngrams = char_ngrams(hypothesis, 8)
    reference_ngrams = char_ngrams(reference, 8)
    if not hypothesis_ngrams:
        return 0.0
    return len(hypothesis_ngrams & reference_ngrams) / max(1, len(hypothesis_ngrams))


def lcs_tokens(text: str) -> list[str]:
    normalized = normalize_for_token_overlap(text)
    return normalized.split() if normalized else []


def exact_lcs_length(first: list[str], second: list[str]) -> int:
    if not first or not second:
        return 0
    previous = [0] * (len(second) + 1)
    for token_first in first:
        current = [0]
        for index, token_second in enumerate(second, start=1):
            if token_first == token_second:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def token_lcs_ratio(hypothesis: str, reference: str) -> float:
    hypothesis_tokens = lcs_tokens(hypothesis)
    reference_tokens = lcs_tokens(reference)
    if not hypothesis_tokens:
        return 0.0
    return exact_lcs_length(hypothesis_tokens, reference_tokens) / max(1, len(hypothesis_tokens))


def _normalize_references(values: Iterable[str] | None) -> list[str]:
    return [value for value in (values or []) if value and value.strip()]


def extract_titlecase_entity_spans(text: str) -> set[str]:
    spans = set()
    for match in ENTITY_SPAN_RE.finditer(text):
        span = " ".join(match.group(0).split())
        words = span.split()
        if not words:
            continue
        if len(words) == 1 and words[0].rstrip(".") in COMMON_CAPITALIZED:
            continue
        if len(words) == 1 and len(words[0]) < 4:
            continue
        spans.add(span)
    return spans


def extract_rare_capitalized_tokens(text: str) -> set[str]:
    return {
        token.rstrip(".")
        for token in CAPITALIZED_TOKEN_RE.findall(text)
        if token.rstrip(".") not in COMMON_CAPITALIZED and len(token.rstrip(".")) >= 5
    }


def entity_transplant_metrics(
    hypothesis: str,
    references: Iterable[str],
    *,
    entity_sequence_threshold: int,
    rare_capitalized_threshold: int,
) -> dict[str, float | bool]:
    hypothesis_spans = extract_titlecase_entity_spans(hypothesis)
    hypothesis_rare_caps = extract_rare_capitalized_tokens(hypothesis)
    max_shared_spans = 0
    max_shared_rare_caps = 0
    for reference in references:
        max_shared_spans = max(max_shared_spans, len(hypothesis_spans & extract_titlecase_entity_spans(reference)))
        max_shared_rare_caps = max(max_shared_rare_caps, len(hypothesis_rare_caps & extract_rare_capitalized_tokens(reference)))
    flag = max_shared_spans >= entity_sequence_threshold or max_shared_rare_caps >= rare_capitalized_threshold
    return {
        "entity_sequence_overlap_max": float(max_shared_spans),
        "rare_capitalized_overlap_max": float(max_shared_rare_caps),
        "entity_transplant_flag": flag,
    }


def _group_originality_metrics(
    hypothesis: str,
    references: Iterable[str],
    *,
    char_threshold: float,
    lcs_threshold: float,
    joint_char_threshold: float,
    joint_lcs_threshold: float,
    entity_sequence_threshold: int,
    rare_capitalized_threshold: int,
) -> dict[str, float | bool]:
    normalized_references = _normalize_references(references)
    char_scores = [char_8gram_overlap(hypothesis, reference) for reference in normalized_references]
    lcs_scores = [token_lcs_ratio(hypothesis, reference) for reference in normalized_references]
    char_max = max(char_scores) if char_scores else 0.0
    lcs_max = max(lcs_scores) if lcs_scores else 0.0
    copy_score = max(char_max, lcs_max)
    copy_flag = (
        char_max >= char_threshold
        or lcs_max >= lcs_threshold
        or (char_max >= joint_char_threshold and lcs_max >= joint_lcs_threshold)
    )
    transplant_metrics = entity_transplant_metrics(
        hypothesis,
        normalized_references,
        entity_sequence_threshold=entity_sequence_threshold,
        rare_capitalized_threshold=rare_capitalized_threshold,
    )
    return {
        "reference_count": len(normalized_references),
        "char_8gram_overlap_max": round(char_max, 6),
        "token_lcs_ratio_max": round(lcs_max, 6),
        "copy_score": round(copy_score, 6),
        "copy_flag": copy_flag,
        "entity_sequence_overlap_max": round(float(transplant_metrics["entity_sequence_overlap_max"]), 6),
        "rare_capitalized_overlap_max": round(float(transplant_metrics["rare_capitalized_overlap_max"]), 6),
        "entity_transplant_flag": bool(transplant_metrics["entity_transplant_flag"]),
    }


def compute_originality_metrics(
    hypothesis: str,
    conditioning_texts: Iterable[str] | None = None,
    *,
    comparison_texts: Iterable[str] | None = None,
    reference_groups: Mapping[str, Iterable[str]] | None = None,
    char_threshold: float = 0.30,
    lcs_threshold: float = 0.20,
    joint_char_threshold: float = 0.20,
    joint_lcs_threshold: float = 0.15,
    entity_sequence_threshold: int = 1,
    rare_capitalized_threshold: int = 2,
) -> dict[str, float | bool]:
    groups: dict[str, list[str]] = {}
    base_conditioning = _normalize_references(conditioning_texts)
    if base_conditioning:
        groups["conditioning"] = base_conditioning
    extra_comparison = _normalize_references(comparison_texts)
    if extra_comparison:
        groups["comparison"] = extra_comparison
    for group_name, values in (reference_groups or {}).items():
        normalized_values = _normalize_references(values)
        if not normalized_values:
            continue
        groups.setdefault(group_name, [])
        groups[group_name].extend(normalized_values)

    reference_group_metrics = {
        group_name: _group_originality_metrics(
            hypothesis,
            references,
            char_threshold=char_threshold,
            lcs_threshold=lcs_threshold,
            joint_char_threshold=joint_char_threshold,
            joint_lcs_threshold=joint_lcs_threshold,
            entity_sequence_threshold=entity_sequence_threshold,
            rare_capitalized_threshold=rare_capitalized_threshold,
        )
        for group_name, references in groups.items()
    }
    char_max = max((float(metrics["char_8gram_overlap_max"]) for metrics in reference_group_metrics.values()), default=0.0)
    lcs_max = max((float(metrics["token_lcs_ratio_max"]) for metrics in reference_group_metrics.values()), default=0.0)
    copy_score = max(char_max, lcs_max)
    entity_sequence_overlap_max = max(
        (float(metrics["entity_sequence_overlap_max"]) for metrics in reference_group_metrics.values()),
        default=0.0,
    )
    rare_capitalized_overlap_max = max(
        (float(metrics["rare_capitalized_overlap_max"]) for metrics in reference_group_metrics.values()),
        default=0.0,
    )
    max_reference_group = None
    if reference_group_metrics:
        max_reference_group = max(
            reference_group_metrics,
            key=lambda group_name: (
                bool(reference_group_metrics[group_name]["entity_transplant_flag"]),
                float(reference_group_metrics[group_name]["copy_score"]),
                float(reference_group_metrics[group_name]["entity_sequence_overlap_max"]),
                float(reference_group_metrics[group_name]["rare_capitalized_overlap_max"]),
                group_name,
            ),
        )
    copy_flag = any(bool(metrics["copy_flag"]) for metrics in reference_group_metrics.values())
    entity_transplant_flag = any(bool(metrics["entity_transplant_flag"]) for metrics in reference_group_metrics.values())
    return {
        "char_8gram_overlap_max": round(char_max, 6),
        "token_lcs_ratio_max": round(lcs_max, 6),
        "copy_score": round(copy_score, 6),
        "copy_flag": copy_flag,
        "entity_sequence_overlap_max": round(entity_sequence_overlap_max, 6),
        "rare_capitalized_overlap_max": round(rare_capitalized_overlap_max, 6),
        "entity_transplant_flag": entity_transplant_flag,
        "originality_pass": not copy_flag and not entity_transplant_flag,
        "max_reference_group": max_reference_group,
        "reference_group_metrics": reference_group_metrics,
    }
