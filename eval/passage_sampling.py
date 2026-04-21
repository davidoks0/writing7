from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from eval.benchmark_schema import PassageRecord


ABBREVIATIONS = (
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "st.",
    "prof.",
    "jr.",
    "sr.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
)
SENTENCE_SPLIT_RE = re.compile(r"""[.!?]+["')\]]*\s+(?=[A-Z])""")
WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)


@dataclass
class SentenceSpan:
    text: str
    start: int
    end: int


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _protect_abbreviations(text: str) -> tuple[str, list[int]]:
    lower = text.lower()
    chars: list[str] = []
    index_map: list[int] = []
    cursor = 0
    while cursor < len(text):
        matched = False
        for abbr in ABBREVIATIONS:
            if lower.startswith(abbr, cursor):
                for offset, char in enumerate(abbr):
                    original_index = cursor + offset
                    if offset == len(abbr) - 1:
                        chars.extend(list("<DOT>"))
                        index_map.extend([original_index] * len("<DOT>"))
                    else:
                        chars.append(text[original_index])
                        index_map.append(original_index)
                cursor += len(abbr)
                matched = True
                break
        if matched:
            continue
        chars.append(text[cursor])
        index_map.append(cursor)
        cursor += 1
    return "".join(chars), index_map


def _restore_abbreviations(text: str) -> str:
    return text.replace("<DOT>", ".")


def _trim_segment(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def split_sentences(text: str) -> list[SentenceSpan]:
    normalized = normalize_newlines(text)
    if not normalized.strip():
        return []
    protected, index_map = _protect_abbreviations(normalized)
    boundaries: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_SPLIT_RE.finditer(protected):
        end = match.end()
        boundaries.append((start, end))
        start = end
    boundaries.append((start, len(protected)))

    spans: list[SentenceSpan] = []
    for protected_start, protected_end in boundaries:
        if protected_start >= protected_end:
            continue
        original_start = index_map[protected_start]
        original_end = index_map[protected_end - 1] + 1
        original_start, original_end = _trim_segment(normalized, original_start, original_end)
        if original_start >= original_end:
            continue
        spans.append(
            SentenceSpan(
                text=_restore_abbreviations(normalized[original_start:original_end].strip()),
                start=original_start,
                end=original_end,
            )
        )

    merged: list[SentenceSpan] = []
    for span in spans:
        if len(span.text) < 30:
            if merged:
                previous = merged[-1]
                combined_text = normalized[previous.start:span.end].strip()
                merged[-1] = SentenceSpan(text=combined_text, start=previous.start, end=span.end)
            else:
                merged.append(span)
            continue
        merged.append(span)

    if merged and len(merged[0].text) < 30 and len(merged) > 1:
        first = merged.pop(0)
        nxt = merged[0]
        merged[0] = SentenceSpan(text=normalized[first.start:nxt.end].strip(), start=first.start, end=nxt.end)

    return merged


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _alpha_ratio(text: str) -> float:
    non_ws = [char for char in text if not char.isspace()]
    if not non_ws:
        return 0.0
    alpha = sum(1 for char in non_ws if char.isalpha())
    return alpha / len(non_ws)


def extract_passage_candidates(
    text: str,
    *,
    book_id: str,
    author_id: str,
    min_words: int = 150,
    max_words: int = 300,
    min_sentences: int = 6,
    max_sentences: int = 18,
    region_buckets: int = 5,
) -> list[PassageRecord]:
    sentences = split_sentences(text)
    passages: list[PassageRecord] = []
    total_sentences = len(sentences)
    for start_index in range(total_sentences):
        combined_words = 0
        end_index = start_index
        while end_index < total_sentences:
            candidate_sentences = sentences[start_index : end_index + 1]
            candidate_text = " ".join(sentence.text for sentence in candidate_sentences).strip()
            combined_words = word_count(candidate_text)
            sentence_count = end_index - start_index + 1
            if (combined_words >= min_words and sentence_count >= min_sentences) or combined_words > max_words or sentence_count > max_sentences:
                break
            end_index += 1

        if end_index >= total_sentences:
            continue
        candidate_sentences = sentences[start_index : end_index + 1]
        candidate_text = " ".join(sentence.text for sentence in candidate_sentences).strip()
        combined_words = word_count(candidate_text)
        sentence_count = end_index - start_index + 1
        if not (min_words <= combined_words <= max_words):
            continue
        if not (min_sentences <= sentence_count <= max_sentences):
            continue
        if _alpha_ratio(candidate_text) < 0.60:
            continue
        start_char = candidate_sentences[0].start
        end_char = candidate_sentences[-1].end
        region_bucket = min(region_buckets - 1, math.floor(start_index * region_buckets / max(1, total_sentences)))
        passage_id = f"passage:{book_id.replace(':', '_')}:{start_index}:{end_index + 1}"
        passages.append(
            PassageRecord(
                passage_id=passage_id,
                book_id=book_id,
                author_id=author_id,
                text=candidate_text,
                start_sentence=start_index,
                end_sentence=end_index + 1,
                start_char=start_char,
                end_char=end_char,
                word_count=combined_words,
                char_count=len(candidate_text),
                region_bucket=region_bucket,
                text_sha1=hashlib.sha1(candidate_text.encode("utf-8")).hexdigest(),
            ).validate()
        )
    return passages


def sentence_windows_overlap(first: PassageRecord, second: PassageRecord, min_gap: int = 0) -> bool:
    return not (
        first.end_sentence + min_gap <= second.start_sentence
        or second.end_sentence + min_gap <= first.start_sentence
    )


def validate_non_overlapping(passages: list[PassageRecord], *, min_gap: int = 0) -> None:
    ordered = sorted(passages, key=lambda row: (row.start_sentence, row.end_sentence, row.passage_id))
    for index, current in enumerate(ordered):
        for other in ordered[index + 1 :]:
            if sentence_windows_overlap(current, other, min_gap=min_gap):
                raise ValueError(f"overlapping passage windows: {current.passage_id} vs {other.passage_id}")
