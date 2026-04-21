from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from typing import Iterable


TRAILING_TITLE_MARKERS = {
    "illustrated",
    "complete",
    "abridged",
    "unabridged",
    "volume",
    "vol",
    "part",
}
ROMAN_NUMERAL_RE = re.compile(r"^(?=[ivxlcdm]+$)m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")
CENTURY_RE = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\s+century\b", re.IGNORECASE)

GENRE_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("science_fiction", ("science fiction", "speculative fiction")),
    ("mystery", ("mystery", "detective", "crime fiction", "crime stories")),
    ("adventure", ("adventure", "sea stories", "pirates", "western stories")),
    ("historical_fiction", ("historical fiction",)),
    ("romance", ("romance", "love stories", "courtship")),
    ("children", ("children", "juvenile fiction", "juvenile literature")),
    ("travel", ("travel", "voyages and travels")),
    ("memoir", ("autobiography", "memoir", "personal narratives", "biography")),
    ("essays", ("essays",)),
    ("short_stories", ("short stories",)),
    ("drama", ("plays", "drama", "tragedies", "comedies")),
    ("poetry", ("poetry", "poems")),
    ("novel", ("novel", "fiction")),
]


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_only = ascii_only.lower()
    ascii_only = ascii_only.replace("'", " ").replace("-", " ")
    ascii_only = re.sub(r"[^a-z0-9 ]+", " ", ascii_only)
    ascii_only = re.sub(r"\s+", " ", ascii_only).strip()
    slug = ascii_only.replace(" ", "_")
    return slug or "untitled"


def safe_clip_slug(slug: str, *, max_len: int, salt: str) -> str:
    if len(slug) <= max_len:
        return slug
    digest = hashlib.sha1(f"{slug}{salt}".encode("utf-8")).hexdigest()[:8]
    clipped = slug[: max_len - 10].rstrip("_")
    return f"{clipped}__{digest}"


def canonical_ids(author: str, title: str) -> dict[str, str]:
    author_slug = safe_clip_slug(slugify(author), max_len=80, salt="author")
    title_slug = safe_clip_slug(slugify(title), max_len=120, salt="title")
    title_core = title_core_slug(title_slug)
    return {
        "author_slug": author_slug,
        "title_slug": title_slug,
        "title_core_slug": title_core,
        "author_id": f"author:{author_slug}",
        "book_id": f"book:{author_slug}:{title_slug}",
        "work_id": f"work:{author_slug}:{title_core}",
    }


def title_core_slug(title_slug: str) -> str:
    tokens = title_slug.split("_")
    while tokens:
        tail = tokens[-1]
        if tail in TRAILING_TITLE_MARKERS or tail.isdigit() or ROMAN_NUMERAL_RE.fullmatch(tail):
            tokens.pop()
            continue
        break
    return "_".join(tokens) or title_slug or "untitled"


def choose_display_name(source_names: Iterable[str]) -> str:
    candidates = [name.strip() for name in source_names if name and name.strip()]
    if not candidates:
        return "Unknown"
    counts = Counter(candidates)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    return ranked[0][0]


def source_suffix_score(path: str) -> int:
    suffixes = {
        "-0.txt": 5,
        ".txt.utf-8": 4,
        "-utf8.txt": 4,
        "-8.txt": 3,
        ".txt": 2,
    }
    for suffix, score in suffixes.items():
        if path.endswith(suffix):
            return score
    return 1


def _metadata_strings(*groups: Iterable[str | None]) -> list[str]:
    values: list[str] = []
    for group in groups:
        for value in group:
            if value and value.strip():
                values.append(value.strip())
    return values


def _extract_first_year(values: Iterable[str]) -> int | None:
    for value in values:
        match = YEAR_RE.search(value)
        if match:
            return int(match.group(1))
    return None


def _extract_century(values: Iterable[str]) -> int | None:
    for value in values:
        match = CENTURY_RE.search(value)
        if match:
            return int(match.group(1))
    return None


def bucket_publication_year(year: int | None) -> str:
    if year is None:
        return "unknown"
    bucket_start = (year // 50) * 50
    bucket_end = bucket_start + 49
    return f"{bucket_start:04d}_{bucket_end:04d}"


def infer_publication_year(
    *,
    publication_year: int | None = None,
    subjects: Iterable[str] = (),
    bookshelves: Iterable[str] = (),
) -> int | None:
    if publication_year is not None:
        return publication_year
    metadata_values = _metadata_strings(subjects, bookshelves)
    return _extract_first_year(metadata_values)


def infer_period_bucket(
    *,
    publication_year: int | None = None,
    subjects: Iterable[str] = (),
    bookshelves: Iterable[str] = (),
) -> str:
    resolved_year = infer_publication_year(
        publication_year=publication_year,
        subjects=subjects,
        bookshelves=bookshelves,
    )
    if resolved_year is not None:
        return bucket_publication_year(resolved_year)
    metadata_values = _metadata_strings(subjects, bookshelves)
    century = _extract_century(metadata_values)
    if century is None or century < 1:
        return "unknown"
    start_year = (century - 1) * 100
    end_year = start_year + 99
    return f"{start_year:04d}_{end_year:04d}"


def infer_genre(
    *,
    genre: str | None = None,
    subjects: Iterable[str] = (),
    bookshelves: Iterable[str] = (),
    source_type: str | None = None,
) -> str:
    if genre and genre.strip() and genre.strip().lower() != "unknown":
        return genre.strip().lower().replace(" ", "_")
    metadata_blob = " | ".join(
        value.lower()
        for value in _metadata_strings(subjects, bookshelves, [source_type])
    )
    if not metadata_blob:
        return "unknown"
    for normalized_genre, patterns in GENRE_HINTS:
        if any(pattern in metadata_blob for pattern in patterns):
            return normalized_genre
    return "unknown"
