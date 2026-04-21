from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


DEFAULT_USER_AGENT = "stylebench-corpus-builder/1.0 (research use; contact required by deployer)"
HTTP_URL_PATTERNS = (
    "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
    "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt.utf-8",
    "https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
    "https://www.gutenberg.org/files/{gid}/{gid}.txt",
)
DEFAULT_RSYNC_SOURCE = "aleph.gutenberg.org::gutenberg"


def candidate_text_urls(gutenberg_id: int) -> list[str]:
    return [pattern.format(gid=gutenberg_id) for pattern in HTTP_URL_PATTERNS]


def _request_bytes(url: str, *, timeout: float, user_agent: str, retries: int, backoff_seconds: float) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": user_agent})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 404:
                break
        except urllib.error.URLError as exc:
            last_error = exc
        if attempt < retries - 1:
            time.sleep(backoff_seconds * (attempt + 1))
    raise last_error or RuntimeError(f"failed to fetch {url}")


def fetch_gutenberg_http(
    gutenberg_id: int,
    *,
    output_root: str | Path,
    timeout: float = 30.0,
    user_agent: str = DEFAULT_USER_AGENT,
    retries: int = 3,
    backoff_seconds: float = 1.0,
) -> dict[str, Any]:
    output_root = Path(output_root)
    destination_root = output_root / "raw" / "gutenberg" / "http"
    destination_root.mkdir(parents=True, exist_ok=True)
    attempted: list[str] = []
    for url in candidate_text_urls(gutenberg_id):
        attempted.append(url)
        try:
            payload = _request_bytes(
                url,
                timeout=timeout,
                user_agent=user_agent,
                retries=retries,
                backoff_seconds=backoff_seconds,
            )
        except Exception as exc:
            last_error = str(exc)
            continue
        suffix = ".txt.utf-8" if url.endswith(".txt.utf-8") else ".txt"
        destination = destination_root / f"pg{gutenberg_id}{suffix}"
        destination.write_bytes(payload)
        return {
            "gutenberg_id": gutenberg_id,
            "ok": True,
            "url": url,
            "attempted_urls": attempted,
            "raw_relpath": destination.relative_to(output_root).as_posix(),
            "bytes": len(payload),
        }
    return {
        "gutenberg_id": gutenberg_id,
        "ok": False,
        "attempted_urls": attempted,
        "error": last_error if attempted else "no_candidate_urls",
    }


def fetch_gutenberg_http_range(
    start_id: int,
    end_id: int,
    *,
    output_root: str | Path,
    index_output_path: str | Path | None = None,
    max_workers: int = 8,
    timeout: float = 30.0,
    user_agent: str = DEFAULT_USER_AGENT,
    retries: int = 3,
) -> list[dict[str, Any]]:
    ids = list(range(start_id, end_id + 1))

    def _fetch(book_id: int) -> dict[str, Any]:
        return fetch_gutenberg_http(
            book_id,
            output_root=output_root,
            timeout=timeout,
            user_agent=user_agent,
            retries=retries,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_fetch, ids))
    index_path = Path(index_output_path) if index_output_path is not None else (Path(output_root) / "meta" / "http_fetch_index.json")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def _parse_rsync_listing(stdout: str) -> list[str]:
    relative_files: list[str] = []
    for line in stdout.splitlines():
        parts = line.split(maxsplit=4)
        if len(parts) < 5:
            continue
        relpath = parts[4].strip()
        if not relpath or relpath.endswith("/"):
            continue
        if not _is_text_relpath(relpath):
            continue
        relative_files.append(relpath)
    return relative_files


def _is_text_relpath(relpath: str) -> bool:
    return relpath.endswith(".txt") or ".txt." in relpath


def _extract_gutenberg_id(path_value: str | Path) -> int | None:
    name = Path(path_value).name
    match = re.search(r"(?:^|[^0-9])(?:pg)?(\d+)(?:[^0-9]|$)", name)
    if match is None:
        return None
    return int(match.group(1))


def _selection_inventory(relative_files: list[str], destination_root: Path) -> dict[str, Any]:
    selected_book_ids: set[int] = set()
    present_book_ids: set[int] = set()
    present_file_count = 0
    present_bytes = 0
    for relpath in relative_files:
        gutenberg_id = _extract_gutenberg_id(relpath)
        if gutenberg_id is not None:
            selected_book_ids.add(gutenberg_id)
        destination = destination_root / relpath
        if not destination.exists():
            continue
        present_file_count += 1
        present_bytes += destination.stat().st_size
        if gutenberg_id is not None:
            present_book_ids.add(gutenberg_id)
    return {
        "selected_file_count": len(relative_files),
        "selected_unique_book_count": len(selected_book_ids),
        "present_file_count": present_file_count,
        "present_unique_book_count": len(present_book_ids),
        "present_bytes": present_bytes,
    }


def inspect_gutenberg_rsync_inventory(
    *,
    output_root: str | Path,
    catalog_path: str | Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root)
    destination_root = output_root / "raw" / "gutenberg" / "rsync"
    file_count = 0
    total_bytes = 0
    unique_book_ids: set[int] = set()

    if destination_root.exists():
        for path in destination_root.rglob("*"):
            if not path.is_file():
                continue
            relpath = path.relative_to(destination_root).as_posix()
            if not _is_text_relpath(relpath):
                continue
            file_count += 1
            total_bytes += path.stat().st_size
            gutenberg_id = _extract_gutenberg_id(relpath)
            if gutenberg_id is not None:
                unique_book_ids.add(gutenberg_id)

    if catalog_path is None:
        default_catalog = output_root / "meta" / "gutenberg_catalog_v1.jsonl"
        catalog_path = default_catalog if default_catalog.exists() else None

    catalog_rows = None
    if catalog_path is not None and Path(catalog_path).exists():
        with Path(catalog_path).open("r", encoding="utf-8") as handle:
            catalog_rows = sum(1 for line in handle if line.strip())

    coverage_percent = None
    if catalog_rows:
        coverage_percent = round((len(unique_book_ids) / catalog_rows) * 100.0, 2)

    return {
        "destination": destination_root.as_posix(),
        "file_count": file_count,
        "unique_book_count": len(unique_book_ids),
        "total_bytes": total_bytes,
        "catalog_rows": catalog_rows,
        "catalog_coverage_percent": coverage_percent,
    }


def list_gutenberg_rsync_text_files(
    *,
    rsync_source: str = DEFAULT_RSYNC_SOURCE,
) -> dict[str, Any]:
    list_command = [
        "rsync",
        "-r",
        "--list-only",
        rsync_source,
    ]
    listed = subprocess.run(list_command, capture_output=True, text=True, check=False)
    return {
        "ok": listed.returncode == 0,
        "command": list_command,
        "stdout": listed.stdout,
        "stderr": listed.stderr,
        "relative_files": _parse_rsync_listing(listed.stdout) if listed.returncode == 0 else [],
    }


def select_rsync_text_files(
    relative_files: list[str],
    *,
    max_files: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> list[str]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    selected = relative_files[: max_files or None]
    if shard_count == 1:
        return selected
    return [relpath for index, relpath in enumerate(selected) if index % shard_count == shard_index]


def ingest_gutenberg_rsync(
    *,
    output_root: str | Path,
    rsync_source: str = DEFAULT_RSYNC_SOURCE,
    max_files: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    rsync_timeout_seconds: int = 600,
) -> dict[str, Any]:
    output_root = Path(output_root)
    destination_root = output_root / "raw" / "gutenberg" / "rsync"
    destination_root.mkdir(parents=True, exist_ok=True)

    if max_files is None and shard_count == 1:
        before_inventory = inspect_gutenberg_rsync_inventory(output_root=output_root)
        command = [
            "rsync",
            "-av",
            "--timeout",
            str(rsync_timeout_seconds),
            "--exclude",
            "cache/",
            rsync_source,
            destination_root.as_posix(),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        after_inventory = inspect_gutenberg_rsync_inventory(output_root=output_root)
        return {
            "ok": completed.returncode == 0,
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "destination": destination_root.as_posix(),
            "inventory_before": before_inventory,
            "inventory_after": after_inventory,
            "new_file_count": max(0, int(after_inventory["file_count"]) - int(before_inventory["file_count"])),
            "new_unique_book_count": max(0, int(after_inventory["unique_book_count"]) - int(before_inventory["unique_book_count"])),
        }

    listed = list_gutenberg_rsync_text_files(rsync_source=rsync_source)
    if not listed["ok"]:
        return {
            "ok": False,
            "command": listed["command"],
            "stdout": listed["stdout"][-4000:],
            "stderr": listed["stderr"][-4000:],
        }

    relative_files = select_rsync_text_files(
        listed["relative_files"],
        max_files=max_files,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    selection_before = _selection_inventory(relative_files, destination_root)

    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            for relpath in relative_files:
                handle.write(relpath + "\n")
            files_from_path = handle.name

        command = [
            "rsync",
            "-av",
            "--timeout",
            str(rsync_timeout_seconds),
            "--files-from",
            files_from_path,
            rsync_source,
            destination_root.as_posix(),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        selection_after = _selection_inventory(relative_files, destination_root)
        return {
            "ok": completed.returncode == 0,
            "command": command,
            "returncode": completed.returncode,
            "available_count": len(listed["relative_files"]),
            "listed_count": len(relative_files),
            "shard_index": shard_index,
            "shard_count": shard_count,
            "selected_file_count": selection_after["selected_file_count"],
            "selected_unique_book_count": selection_after["selected_unique_book_count"],
            "preexisting_file_count": selection_before["present_file_count"],
            "preexisting_unique_book_count": selection_before["present_unique_book_count"],
            "present_after_file_count": selection_after["present_file_count"],
            "present_after_unique_book_count": selection_after["present_unique_book_count"],
            "new_file_count": max(0, selection_after["present_file_count"] - selection_before["present_file_count"]),
            "new_unique_book_count": max(0, selection_after["present_unique_book_count"] - selection_before["present_unique_book_count"]),
            "missing_after_count": max(0, selection_after["selected_file_count"] - selection_after["present_file_count"]),
            "present_after_bytes": selection_after["present_bytes"],
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "destination": destination_root.as_posix(),
        }
    finally:
        if "files_from_path" in locals():
            Path(files_from_path).unlink(missing_ok=True)
