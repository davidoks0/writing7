from __future__ import annotations

import re
from typing import Iterable


RAW_TEXT_VIEW = "raw"
STYLE_MASKED_TEXT_VIEW = "style_masked_v1"
SUPPORTED_TEXT_VIEWS = {RAW_TEXT_VIEW, STYLE_MASKED_TEXT_VIEW}

HONORIFICS = {
    "mr",
    "mrs",
    "ms",
    "miss",
    "dr",
    "prof",
    "sir",
    "lady",
    "lord",
    "capt",
    "captain",
    "colonel",
    "general",
}

TOKEN_RE = re.compile(r"\s+|[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:[.,:/-]\d+)*|[^\w\s]", re.UNICODE)


def normalize_text_view(text_view: str | None) -> str:
    if not text_view:
        return RAW_TEXT_VIEW
    if text_view not in SUPPORTED_TEXT_VIEWS:
        raise ValueError(f"Unsupported text_view: {text_view}")
    return text_view


def normalize_score_text_views(text_views: Iterable[str] | None, *, default: str | None = None) -> list[str]:
    candidate_views = list(text_views or [])
    if not candidate_views:
        candidate_views = [default or RAW_TEXT_VIEW]
    normalized: list[str] = []
    for value in candidate_views:
        normalized_value = normalize_text_view(value)
        if normalized_value not in normalized:
            normalized.append(normalized_value)
    return normalized


def normalize_blend_weights(
    blend_weights: dict[str, float] | None,
    *,
    score_text_views: Iterable[str],
) -> dict[str, float]:
    normalized_views = normalize_score_text_views(score_text_views)
    if not normalized_views:
        return {RAW_TEXT_VIEW: 1.0}
    raw_weights = {
        normalize_text_view(key): max(0.0, float(value))
        for key, value in (blend_weights or {}).items()
    }
    normalized = {view: raw_weights.get(view, 0.0) for view in normalized_views}
    if not any(weight > 0.0 for weight in normalized.values()):
        fallback_weight = 1.0 / len(normalized_views)
        return {view: fallback_weight for view in normalized_views}
    total = sum(normalized.values())
    return {view: weight / total for view, weight in normalized.items()}


def _is_word(token: str) -> bool:
    return bool(token) and any(char.isalpha() for char in token)


def _is_numeric(token: str) -> bool:
    return bool(token) and any(char.isdigit() for char in token)


def _is_titlecase_like(token: str) -> bool:
    if not _is_word(token):
        return False
    letters = [char for char in token if char.isalpha()]
    if not letters:
        return False
    return letters[0].isupper() and any(char.islower() for char in letters[1:])


def _is_all_caps(token: str) -> bool:
    letters = [char for char in token if char.isalpha()]
    return len(letters) >= 2 and all(char.isupper() for char in letters)


def _next_lexical_token(tokens: list[str], start_index: int) -> str | None:
    for index in range(start_index + 1, len(tokens)):
        candidate = tokens[index]
        if candidate.isspace():
            continue
        if _is_word(candidate) or _is_numeric(candidate):
            return candidate
        if candidate in {'"', "'", "(", "[", "{", "`"}:
            continue
        break
    return None


def style_focus_text(text: str) -> str:
    if not text.strip():
        return text
    tokens = TOKEN_RE.findall(text)
    rendered: list[str] = []
    sentence_start = True
    previous_word_lower: str | None = None
    for index, token in enumerate(tokens):
        if token.isspace():
            rendered.append(token)
            if "\n" in token:
                sentence_start = True
                previous_word_lower = None
            continue
        if _is_numeric(token):
            rendered.append("<NUM>")
            previous_word_lower = "<num>"
            sentence_start = False
            continue
        if _is_word(token):
            lower = token.lower()
            if lower in HONORIFICS:
                rendered.append(token)
                previous_word_lower = lower
                sentence_start = False
                continue
            next_token = _next_lexical_token(tokens, index)
            next_named_like = (
                _is_numeric(next_token or "")
                or _is_titlecase_like(next_token or "")
                or _is_all_caps(next_token or "")
            )
            named_like = False
            if _is_all_caps(token):
                named_like = True
            elif _is_titlecase_like(token):
                named_like = (
                    previous_word_lower in HONORIFICS
                    or not sentence_start
                    or next_named_like
                )
            if named_like:
                rendered.append("<ENT>")
                previous_word_lower = "<ent>"
            else:
                rendered.append(token)
                previous_word_lower = lower
            sentence_start = False
            continue
        rendered.append(token)
        if token in ".!?":
            sentence_start = previous_word_lower not in HONORIFICS
            if sentence_start:
                previous_word_lower = None
    return "".join(rendered)


def apply_text_view(text: str, text_view: str | None) -> str:
    normalized = normalize_text_view(text_view)
    if normalized == RAW_TEXT_VIEW:
        return text
    return style_focus_text(text)
