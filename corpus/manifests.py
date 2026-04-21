from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from corpus.clean_books import clean_book_text, prose_heuristics, text_stats
from corpus.gutenberg_catalog import load_gutenberg_catalog_index
from corpus.layout import is_within_root, relativize_to_root, resolve_with_root
from corpus.metadata import (
    canonical_ids,
    choose_display_name,
    infer_genre,
    infer_period_bucket,
    infer_publication_year,
    safe_clip_slug,
    source_suffix_score,
    title_core_slug,
)
from eval.benchmark_io import load_json, load_jsonl, write_json, write_jsonl
from eval.benchmark_schema import InputBookRecord


TITLE_RE = re.compile(r"^title:\s*(.+)$", re.IGNORECASE)
AUTHOR_RE = re.compile(r"^author:\s*(.+)$", re.IGNORECASE)
LANGUAGE_RE = re.compile(r"^language:\s*(.+)$", re.IGNORECASE)
EBOOK_RE = re.compile(r"ebook\s+#?(\d+)", re.IGNORECASE)


DEFAULT_CORPUS_CONFIG = {
    "builder_seed": 42,
    "min_clean_words": 40000,
    "min_clean_sentences": 1500,
    "min_author_books": 3,
    "benchmark_split_requires_author_track": True,
    "alpha_char_ratio_min": 0.70,
    "raw_scan_roots": [
        "raw/gutenberg/http",
        "raw/gutenberg/rsync",
    ],
    "passage_policy": {
        "min_words": 150,
        "max_words": 300,
        "min_sentences": 6,
        "max_sentences": 18,
        "region_buckets": 5,
    },
}


def load_corpus_config(path_or_payload: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_payload, dict):
        config = dict(DEFAULT_CORPUS_CONFIG)
        config.update(path_or_payload)
        return config
    config = dict(DEFAULT_CORPUS_CONFIG)
    if path_or_payload:
        config.update(load_json(path_or_payload))
    return config


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha1_path(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _clean_relpath(author_slug: str, title_slug: str) -> str:
    return f"clean/gutenberg/{author_slug}/{title_slug}.txt"


def _local_raw_relpath(author_slug: str, source_stem: str, suffix: str) -> str:
    suffix = suffix if suffix.startswith(".") else f".{suffix}" if suffix else ".txt"
    if suffix == ".":
        suffix = ".txt"
    return f"raw/local_manifest/{author_slug}/{source_stem}{suffix}"


def _language_is_english(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.lower().strip()
    return normalized in {"en", "english"}


def _is_stable_author(author: str) -> bool:
    lowered = author.lower().strip()
    return bool(lowered) and lowered not in {"anonymous", "unknown", "various"}


def _infer_acquisition_mode(path: Path, output_root: Path) -> tuple[str, str]:
    try:
        relpath = path.resolve().relative_to(output_root.resolve()).as_posix()
    except ValueError:
        return "local_manifest", "local_manifest"
    if relpath.startswith("raw/gutenberg/http/"):
        return "gutenberg_http", "http"
    if relpath.startswith("raw/gutenberg/rsync/"):
        return "gutenberg_rsync", "rsync"
    return "local_manifest", "local_manifest"


def _parse_gutenberg_header(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()[:400]
    title = None
    author = None
    languages: list[str] = []
    gutenberg_id = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        title_match = TITLE_RE.match(line)
        if title is None and title_match:
            title = title_match.group(1).strip()
            continue
        author_match = AUTHOR_RE.match(line)
        if author is None and author_match:
            author = author_match.group(1).strip()
            continue
        language_match = LANGUAGE_RE.match(line)
        if language_match:
            languages.append(language_match.group(1).strip())
        if gutenberg_id is None:
            ebook_match = EBOOK_RE.search(line)
            if ebook_match:
                gutenberg_id = ebook_match.group(1)
    if gutenberg_id is None:
        filename_digits = re.findall(r"\d+", path.stem)
        gutenberg_id = filename_digits[0] if filename_digits else None
    return {
        "title": title or path.stem.replace("_", " "),
        "author": author or "Unknown",
        "languages": languages,
        "gutenberg_id": gutenberg_id,
    }


def _discover_input_rows(config: dict[str, Any], output_root: Path) -> list[InputBookRecord]:
    input_manifest = config.get("input_books_manifest")
    if input_manifest and Path(input_manifest).exists():
        return [InputBookRecord.from_dict(row) for row in load_jsonl(input_manifest)]

    rows: list[InputBookRecord] = []
    catalog_by_id = load_gutenberg_catalog_index(output_root)
    raw_roots = [output_root / relpath for relpath in config.get("raw_scan_roots", [])]
    for raw_root in raw_roots:
        if not raw_root.exists():
            continue
        for path in sorted(raw_root.rglob("*")):
            if not path.is_file():
                continue
            if not (path.name.endswith(".txt") or ".txt." in path.name):
                continue
            header = _parse_gutenberg_header(path)
            catalog_row = catalog_by_id.get(str(header["gutenberg_id"])) if header.get("gutenberg_id") else None
            title = catalog_row.get("title") if catalog_row and catalog_row.get("title") else header["title"]
            author = catalog_row.get("author") if catalog_row and catalog_row.get("author") else header["author"]
            ids = canonical_ids(author, title)
            subjects = list(catalog_row.get("subjects", [])) if catalog_row else []
            bookshelves = list(catalog_row.get("bookshelves", [])) if catalog_row else []
            publication_year = infer_publication_year(subjects=subjects, bookshelves=bookshelves)
            period_bucket = infer_period_bucket(publication_year=publication_year, subjects=subjects, bookshelves=bookshelves)
            genre = infer_genre(subjects=subjects, bookshelves=bookshelves, source_type=(catalog_row or {}).get("type"))
            language = (
                catalog_row.get("language")
                if catalog_row and catalog_row.get("language")
                else header["languages"][0]
                if header["languages"]
                else None
            )
            rows.append(
                InputBookRecord(
                    book_id=ids["book_id"],
                    author_id=ids["author_id"],
                    title=title,
                    author=author,
                    source_path=path.as_posix(),
                    gutenberg_id=header["gutenberg_id"],
                    language=language,
                    publication_year=publication_year,
                    period_bucket=period_bucket,
                    genre=genre,
                    is_translation=False,
                    subjects=subjects,
                    bookshelves=bookshelves,
                    source_type=(catalog_row or {}).get("type"),
                ).validate()
            )
    return rows


def _materialize_source(record: InputBookRecord, output_root: Path, *, author_slug: str) -> tuple[Path, str]:
    source_path = Path(record.source_path)
    if is_within_root(source_path, output_root):
        resolved = resolve_with_root(output_root, relativize_to_root(source_path, output_root))
        return resolved, relativize_to_root(resolved, output_root)
    source_stem = safe_clip_slug(source_path.stem or "source", max_len=100, salt="raw_source")
    raw_relpath = _local_raw_relpath(author_slug, source_stem, source_path.suffix or ".txt")
    destination = output_root / raw_relpath
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = source_path.read_bytes()
    if not destination.exists() or destination.read_bytes() != payload:
        destination.write_bytes(payload)
    return destination, raw_relpath


def _source_id_for_record(record: InputBookRecord, source_path: Path, output_root: Path) -> tuple[str, str, str]:
    source_name, acquisition_mode = _infer_acquisition_mode(source_path, output_root)
    source_key = record.gutenberg_id or safe_clip_slug(source_path.stem, max_len=120, salt="source")
    return f"source:{acquisition_mode}:{source_key}", source_name, acquisition_mode


def _sorted_author_rows(rows: list[dict[str, Any]], builder_seed: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(hashlib.sha1(f"{builder_seed}|{row['author_id']}".encode("utf-8")).hexdigest()[:16], 16),
            row["author_id"],
        ),
    )


def _split_counts(total: int) -> tuple[int, int, int, int]:
    scorer_train_count = int(total * 0.70)
    scorer_calibration_count = int(total * 0.10)
    benchmark_dev_count = int(total * 0.10)
    benchmark_test_count = total - scorer_train_count - scorer_calibration_count - benchmark_dev_count

    if total >= 4:
        scorer_calibration_count = max(1, scorer_calibration_count)
        benchmark_dev_count = max(1, benchmark_dev_count)
        benchmark_test_count = max(1, benchmark_test_count)
        scorer_train_count = max(0, total - scorer_calibration_count - benchmark_dev_count - benchmark_test_count)
    return scorer_train_count, scorer_calibration_count, benchmark_dev_count, benchmark_test_count


def build_corpus_manifests(config_path_or_payload: str | Path | dict[str, Any]) -> dict[str, Any]:
    config = load_corpus_config(config_path_or_payload)
    output_root = Path(config.get("output_root", "build/corpus"))
    rows = _discover_input_rows(config, output_root)
    if not rows:
        raise ValueError("no input books found; provide input_books_manifest or populate raw/gutenberg/*")

    source_rows: list[dict[str, Any]] = []
    book_rows: list[dict[str, Any]] = []
    by_author_slug: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_work: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for record in rows:
        ids = canonical_ids(record.author, record.title)
        source_path, raw_relpath = _materialize_source(record, output_root, author_slug=ids["author_slug"])
        cleaned_text = clean_book_text(source_path)
        source_id, source_name, acquisition_mode = _source_id_for_record(record, source_path, output_root)
        clean_relpath = _clean_relpath(ids["author_slug"], ids["title_slug"])
        clean_path = output_root / clean_relpath
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        clean_path.write_text(cleaned_text, encoding="utf-8")

        stats = text_stats(cleaned_text)
        prose = prose_heuristics(cleaned_text)
        languages_header = [record.language] if record.language else []
        source_rows.append(
            {
                "source_id": source_id,
                "source": source_name,
                "acquisition_mode": acquisition_mode,
                "gutenberg_id": record.gutenberg_id,
                "url": None,
                "raw_relpath": raw_relpath,
                "clean_relpath": clean_relpath,
                "header_title": record.title,
                "header_author": record.author,
                "languages": languages_header,
                "subjects": list(record.subjects),
                "bookshelves": list(record.bookshelves),
                "catalog_type": record.source_type,
                "raw_sha1": _sha1_path(source_path),
                "clean_sha1": _sha1_path(clean_path),
                "raw_bytes": source_path.stat().st_size,
                "clean_bytes": clean_path.stat().st_size,
                "fetched_at_utc": _timestamp_utc(),
            }
        )

        book_row = {
            "book_id": ids["book_id"],
            "work_id": ids["work_id"],
            "author_id": ids["author_id"],
            "author_slug": ids["author_slug"],
            "title_slug": ids["title_slug"],
            "title": record.title,
            "author": record.author,
            "source_path": raw_relpath,
            "source_ids": [source_id],
            "primary_source_id": source_id,
            "clean_path": clean_relpath,
            "clean_relpath": clean_relpath,
            "language": record.language,
            "languages_header": languages_header,
            "publication_year": record.publication_year,
            "period_bucket": (
                record.period_bucket
                if record.period_bucket and record.period_bucket != "unknown"
                else infer_period_bucket(
                    publication_year=record.publication_year,
                    subjects=record.subjects,
                    bookshelves=record.bookshelves,
                )
            ),
            "genre": infer_genre(
                genre=record.genre,
                subjects=record.subjects,
                bookshelves=record.bookshelves,
                source_type=record.source_type,
            ),
            "is_translation": bool(record.is_translation),
            "subjects": list(record.subjects),
            "bookshelves": list(record.bookshelves),
            "catalog_type": record.source_type,
            "duplicate_group_id": f"work:{ids['author_slug']}:{title_core_slug(ids['title_slug'])}",
            "duplicate_rank": 0,
            **stats,
            **prose,
        }
        book_rows.append(book_row)
        by_author_slug[ids["author_slug"]].append(book_row)
        by_work[(ids["author_slug"], title_core_slug(ids["title_slug"]))].append(book_row)

    duplicate_lookup: dict[str, dict[str, Any]] = {}
    for group_rows in by_work.values():
        ranked = sorted(
            group_rows,
            key=lambda row: (
                0 if "english" in [lang.lower() for lang in row.get("languages_header", [])] else 1,
                -source_suffix_score(row["source_path"]),
                -row["clean_word_count"],
                -row["clean_char_count"],
                row["clean_relpath"],
            ),
        )
        for rank, row in enumerate(ranked):
            row["duplicate_rank"] = rank
            duplicate_lookup[row["book_id"]] = {
                "passes_duplicate_filter": rank == 0,
                "duplicate_of_book_id": None if rank == 0 else ranked[0]["book_id"],
            }

    eligibility_rows: list[dict[str, Any]] = []
    for book_row in book_rows:
        duplicate_info = duplicate_lookup[book_row["book_id"]]
        passes_language = _language_is_english(book_row.get("language"))
        passes_cleaning = bool(resolve_with_root(output_root, book_row["clean_path"]).read_text(encoding="utf-8").strip())
        passes_length = (
            book_row["clean_word_count"] >= config["min_clean_words"]
            and book_row["clean_sentence_count"] >= config["min_clean_sentences"]
            and book_row["alpha_char_ratio"] >= config["alpha_char_ratio_min"]
        )
        passes_prose_heuristic = bool(book_row["passes_prose_heuristic"])
        passes_author_stability = _is_stable_author(book_row["author"])
        exclusion_reasons = []
        if not passes_language:
            exclusion_reasons.append("language_not_english")
        if not passes_cleaning:
            exclusion_reasons.append("cleaning_failed")
        if not passes_length:
            exclusion_reasons.append("below_length_threshold")
        if not passes_prose_heuristic:
            exclusion_reasons.append("prose_heuristic_failed")
        if not duplicate_info["passes_duplicate_filter"]:
            exclusion_reasons.append("duplicate_edition")
        if not passes_author_stability:
            exclusion_reasons.append("author_not_stable")
        eligible_corpus_all = all(
            [
                passes_language,
                passes_cleaning,
                passes_length,
                passes_prose_heuristic,
                duplicate_info["passes_duplicate_filter"],
                passes_author_stability,
            ]
        )
        eligibility_rows.append(
            {
                "book_id": book_row["book_id"],
                "author_id": book_row["author_id"],
                "passes_language": passes_language,
                "passes_cleaning": passes_cleaning,
                "passes_length": passes_length,
                "passes_prose_heuristic": passes_prose_heuristic,
                "passes_duplicate_filter": duplicate_info["passes_duplicate_filter"],
                "passes_author_stability": passes_author_stability,
                "eligible_corpus_all": eligible_corpus_all,
                "eligible_author_track": False,
                "eligible_book_track": eligible_corpus_all,
                "duplicate_of_book_id": duplicate_info["duplicate_of_book_id"],
                "exclusion_reasons": exclusion_reasons,
            }
        )

    eligibility_by_book = {row["book_id"]: row for row in eligibility_rows}
    author_rows: list[dict[str, Any]] = []
    for author_slug, author_books in sorted(by_author_slug.items()):
        eligible_books = [
            row["book_id"]
            for row in author_books
            if eligibility_by_book[row["book_id"]]["eligible_corpus_all"]
        ]
        author_id = author_books[0]["author_id"]
        display_name = choose_display_name(book["author"] for book in author_books)
        eligible_author_track = len(eligible_books) >= config["min_author_books"]
        for book_id in eligible_books:
            eligibility_by_book[book_id]["eligible_author_track"] = eligible_author_track
        author_rows.append(
            {
                "author_id": author_id,
                "author_slug": author_slug,
                "display_name": display_name,
                "source_names": sorted({book["author"] for book in author_books}),
                "candidate_book_ids": sorted(book["book_id"] for book in author_books),
                "eligible_book_ids": sorted(eligible_books),
                "n_candidate_books": len(author_books),
                "n_eligible_books": len(eligible_books),
                "total_clean_words": sum(book["clean_word_count"] for book in author_books),
                "eligible_author_track": eligible_author_track,
                "eligible_book_track": bool(eligible_books),
                "exclusion_reasons": [] if eligible_books else ["no_eligible_books"],
            }
        )

    author_track_lookup = {row["author_id"]: row["eligible_author_track"] for row in author_rows}
    for book_row in book_rows:
        eligibility = eligibility_by_book[book_row["book_id"]]
        book_row["eligible_author_track"] = bool(author_track_lookup.get(book_row["author_id"], False) and eligibility["eligible_corpus_all"])
        book_row["eligible_book_track"] = bool(eligibility["eligible_book_track"])
        book_row["exclusion_reasons"] = eligibility["exclusion_reasons"]

    meta_root = output_root / "meta"
    write_jsonl(meta_root / "source_index.jsonl", source_rows)
    write_jsonl(meta_root / "books_manifest_v1.jsonl", sorted(book_rows, key=lambda row: row["book_id"]))
    write_jsonl(meta_root / "authors_manifest_v1.jsonl", sorted(author_rows, key=lambda row: row["author_id"]))
    write_jsonl(meta_root / "eligibility_v1.jsonl", sorted(eligibility_rows, key=lambda row: row["book_id"]))

    return {
        "output_root": output_root.as_posix(),
        "source_index": (meta_root / "source_index.jsonl").as_posix(),
        "books_manifest": (meta_root / "books_manifest_v1.jsonl").as_posix(),
        "authors_manifest": (meta_root / "authors_manifest_v1.jsonl").as_posix(),
        "eligibility_manifest": (meta_root / "eligibility_v1.jsonl").as_posix(),
    }


def freeze_corpus_splits(config_path_or_payload: str | Path | dict[str, Any]) -> dict[str, str]:
    config = load_corpus_config(config_path_or_payload)
    output_root = Path(config.get("output_root", "build/corpus"))
    authors_manifest = load_jsonl(output_root / "meta" / "authors_manifest_v1.jsonl")
    books_manifest = {row["book_id"]: row for row in load_jsonl(output_root / "meta" / "books_manifest_v1.jsonl")}
    eligible_authors = [row for row in authors_manifest if row["n_eligible_books"] >= 1]
    builder_seed = config["builder_seed"]
    eligible_authors = _sorted_author_rows(eligible_authors, builder_seed)
    benchmark_requires_author_track = bool(config.get("benchmark_split_requires_author_track", True))
    benchmark_candidate_authors = [
        row for row in eligible_authors if row["eligible_author_track"] or not benchmark_requires_author_track
    ]
    benchmark_candidate_ids = {row["author_id"] for row in benchmark_candidate_authors}
    auxiliary_train_only_authors = [
        row for row in eligible_authors if row["author_id"] not in benchmark_candidate_ids
    ]
    total = len(benchmark_candidate_authors)
    scorer_train_count, scorer_calibration_count, benchmark_dev_count, benchmark_test_count = _split_counts(total)
    split_slices = {
        "scorer_train": benchmark_candidate_authors[:scorer_train_count] + auxiliary_train_only_authors,
        "scorer_calibration": benchmark_candidate_authors[
            scorer_train_count : scorer_train_count + scorer_calibration_count
        ],
        "benchmark_dev": benchmark_candidate_authors[
            scorer_train_count + scorer_calibration_count : scorer_train_count + scorer_calibration_count + benchmark_dev_count
        ],
        "benchmark_test": benchmark_candidate_authors[
            scorer_train_count + scorer_calibration_count + benchmark_dev_count :
        ],
    }

    split_root = output_root / "splits"
    outputs: dict[str, str] = {}
    for split_name, split_authors in split_slices.items():
        split_authors = _sorted_author_rows(split_authors, builder_seed)
        author_ids = [row["author_id"] for row in split_authors]
        book_ids = sorted(book_id for author_row in split_authors for book_id in author_row["eligible_book_ids"])
        author_track_book_ids = sorted(book_id for book_id in book_ids if books_manifest[book_id]["eligible_author_track"])
        book_track_book_ids = sorted(book_id for book_id in book_ids if books_manifest[book_id]["eligible_book_track"])
        payload = {
            "artifact_type": "corpus_split",
            "split_version": "splits_v1",
            "split_name": split_name,
            "builder_seed": builder_seed,
            "author_ids": author_ids,
            "book_ids": book_ids,
            "author_track_book_ids": author_track_book_ids,
            "book_track_book_ids": book_track_book_ids,
            "counts": {"authors": len(author_ids), "books": len(book_ids)},
        }
        destination = split_root / f"{split_name}_v1.json"
        write_json(destination, payload)
        outputs[split_name] = destination.as_posix()
    return outputs
