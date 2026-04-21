from __future__ import annotations

import re
from pathlib import Path

from eval.passage_sampling import split_sentences, word_count


HEADER_RE = re.compile(r"\*\*\*\s*start of (?:this|the) project gutenberg ebook.*?\*\*\*", re.IGNORECASE | re.DOTALL)
FOOTER_RE = re.compile(r"\*\*\*\s*end of (?:this|the) project gutenberg ebook.*", re.IGNORECASE | re.DOTALL)
MULTI_BLANK_RE = re.compile(r"\n{3,}")
WORD_SPLIT_RE = re.compile(r"\S+")
SPEAKER_LINE_RE = re.compile(r"^[A-Z][A-Z .'\-]{1,30}[.:]$")


def strip_gutenberg_boilerplate(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    start_match = HEADER_RE.search(normalized)
    if start_match:
        normalized = normalized[start_match.end() :]
    footer_match = FOOTER_RE.search(normalized)
    if footer_match:
        normalized = normalized[: footer_match.start()]
    return normalized


def normalize_body_text(text: str) -> str:
    body = strip_gutenberg_boilerplate(text)
    body = body.replace("\u2018", "'").replace("\u2019", "'")
    body = body.replace("\u201c", '"').replace("\u201d", '"')
    body = body.replace("\u2014", "--").replace("\u2013", "-")
    body = MULTI_BLANK_RE.sub("\n\n", body)
    return body.strip() + "\n"


def clean_book_text(path: str | Path) -> str:
    return normalize_body_text(Path(path).read_text(encoding="utf-8"))


def alpha_char_ratio(text: str) -> float:
    non_ws = [char for char in text if not char.isspace()]
    if not non_ws:
        return 0.0
    alpha = sum(1 for char in non_ws if char.isalpha())
    return alpha / len(non_ws)


def prose_heuristics(text: str) -> dict[str, float | bool]:
    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not non_empty_lines:
        return {
            "short_line_rate": 1.0,
            "speaker_line_rate": 1.0,
            "passes_prose_heuristic": False,
        }
    short_lines = 0
    speaker_lines = 0
    for line in non_empty_lines:
        if len(WORD_SPLIT_RE.findall(line)) <= 8:
            short_lines += 1
        if SPEAKER_LINE_RE.fullmatch(line):
            speaker_lines += 1
    short_rate = short_lines / len(non_empty_lines)
    speaker_rate = speaker_lines / len(non_empty_lines)
    return {
        "short_line_rate": short_rate,
        "speaker_line_rate": speaker_rate,
        "passes_prose_heuristic": short_rate <= 0.45 and speaker_rate <= 0.20,
    }


def text_stats(text: str) -> dict[str, int | float]:
    sentences = split_sentences(text)
    return {
        "clean_word_count": word_count(text),
        "clean_char_count": len(text),
        "clean_sentence_count": len(sentences),
        "alpha_char_ratio": alpha_char_ratio(text),
    }

