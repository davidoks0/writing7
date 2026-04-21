from __future__ import annotations

import csv
import gzip
import io
import urllib.request
from pathlib import Path
from typing import Any

from eval.benchmark_io import load_jsonl, write_jsonl


DEFAULT_CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
DEFAULT_USER_AGENT = "stylebench-catalog-builder/1.0 (research use; contact required by deployer)"


def _normalize_key(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum())


def _pick(row: dict[str, str], *names: str) -> str | None:
    for name in names:
        normalized = _normalize_key(name)
        if normalized in row and row[normalized].strip():
            return row[normalized].strip()
    return None


def _split_multi(value: str | None, *, separators: tuple[str, ...] = (";",)) -> list[str]:
    if not value:
        return []
    tokens = [value]
    for separator in separators:
        expanded: list[str] = []
        for token in tokens:
            expanded.extend(token.split(separator))
        tokens = expanded
    return [token.strip() for token in tokens if token.strip()]


def _parse_catalog_csv_bytes(payload: bytes, source_name: str) -> list[dict[str, Any]]:
    if source_name.endswith(".gz"):
        payload = gzip.decompress(payload)
    text = payload.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, Any]] = []
    for raw_row in reader:
        row = {_normalize_key(key): (value or "") for key, value in raw_row.items() if key}
        gutenberg_id = _pick(row, "Text#", "ID", "EBook-No.", "ebook_no")
        if not gutenberg_id:
            continue
        authors = _split_multi(_pick(row, "Authors", "Author"))
        languages = _split_multi(_pick(row, "Language", "Languages"))
        issued = _pick(row, "Issued", "Release Date", "Released")
        rows.append(
            {
                "gutenberg_id": gutenberg_id,
                "title": _pick(row, "Title") or f"Project Gutenberg {gutenberg_id}",
                "author": authors[0] if authors else "Unknown",
                "authors": authors,
                "language": languages[0] if languages else None,
                "languages": languages,
                "issued": issued,
                "subjects": _split_multi(_pick(row, "Subjects", "Subject")),
                "bookshelves": _split_multi(_pick(row, "Bookshelves", "Bookshelf")),
                "type": _pick(row, "Type"),
            }
        )
    return rows


def fetch_gutenberg_catalog(
    *,
    output_root: str | Path,
    url: str = DEFAULT_CATALOG_URL,
    timeout: float = 90.0,
    user_agent: str = DEFAULT_USER_AGENT,
) -> dict[str, Any]:
    output_root = Path(output_root)
    destination_root = output_root / "raw" / "gutenberg" / "catalog"
    destination_root.mkdir(parents=True, exist_ok=True)
    filename = "pg_catalog.csv.gz" if url.endswith(".gz") else "pg_catalog.csv"
    destination = destination_root / filename
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    destination.write_bytes(payload)
    return {
        "ok": True,
        "url": url,
        "bytes": len(payload),
        "catalog_path": destination.as_posix(),
    }


def build_gutenberg_catalog_index(
    *,
    output_root: str | Path,
    catalog_path: str | Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root)
    source = Path(catalog_path) if catalog_path else None
    if source is None:
        csv_path = output_root / "raw" / "gutenberg" / "catalog" / "pg_catalog.csv"
        gz_path = output_root / "raw" / "gutenberg" / "catalog" / "pg_catalog.csv.gz"
        source = csv_path if csv_path.exists() else gz_path
    if source is None or not source.exists():
        raise ValueError("catalog_path does not exist and no default catalog file was found")

    rows = _parse_catalog_csv_bytes(source.read_bytes(), source.name)
    destination = output_root / "meta" / "gutenberg_catalog_v1.jsonl"
    write_jsonl(destination, sorted(rows, key=lambda row: int(row["gutenberg_id"]) if str(row["gutenberg_id"]).isdigit() else row["gutenberg_id"]))
    return {
        "catalog_rows": len(rows),
        "catalog_index": destination.as_posix(),
    }


def load_gutenberg_catalog_index(output_root: str | Path) -> dict[str, dict[str, Any]]:
    output_root = Path(output_root)
    index_path = output_root / "meta" / "gutenberg_catalog_v1.jsonl"
    if not index_path.exists():
        return {}
    return {str(row["gutenberg_id"]): row for row in load_jsonl(index_path)}
