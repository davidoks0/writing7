from __future__ import annotations

import re


WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def repetition_rate_6gram(text: str) -> float:
    tokens = [token.lower() for token in WORD_RE.findall(text)]
    if len(tokens) < 6:
        return 0.0
    seen: dict[tuple[str, ...], int] = {}
    repeated = 0
    total = 0
    for index in range(0, len(tokens) - 5):
        sixgram = tuple(tokens[index : index + 6])
        total += 1
        seen[sixgram] = seen.get(sixgram, 0) + 1
        if seen[sixgram] > 1:
            repeated += 1
    return repeated / max(1, total)


def malformed_flag(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if "[llm error]" in lowered or "api key" in lowered or "rate limit" in lowered:
        return True
    non_ws = [char for char in stripped if not char.isspace()]
    if not non_ws:
        return True
    alpha = sum(1 for char in non_ws if char.isalpha())
    return (alpha / len(non_ws)) < 0.40


def compute_fluency_metrics(
    text: str,
    *,
    min_words_valid: int = 350,
    max_words_valid: int = 1000,
    max_repetition_rate_6gram: float = 0.20,
) -> dict[str, int | float | bool]:
    count = word_count(text)
    repetition = repetition_rate_6gram(text)
    malformed = malformed_flag(text)
    min_length_pass = min_words_valid <= count <= max_words_valid
    fluency_pass = min_length_pass and not malformed and repetition < max_repetition_rate_6gram
    return {
        "word_count": count,
        "repetition_rate_6gram": round(repetition, 6),
        "malformed_flag": malformed,
        "min_length_pass": min_length_pass,
        "fluency_pass": fluency_pass,
    }
