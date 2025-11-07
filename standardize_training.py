#!/usr/bin/env python3
"""
Standardize text files in `training/` by removing Project Gutenberg headers/footers
and normalizing basic formatting.

Features:
- Strips common Project Gutenberg headers and footers (START/END markers, license lines).
- Removes typical "Produced by", "Transcriber's Note", and PGDP notices near the top.
- Optionally removes "Transcriber's Note(s)" sections elsewhere.
- Normalizes newlines to `\n`, trims trailing whitespace, removes BOM.
- Collapses multiple blank lines to a single blank line and trims leading/trailing blanks.

Usage:
  python standardize_training.py [--in-place] [--src DIR] [--dst DIR] [--limit N]

Defaults:
- Reads from `training/` and writes cleaned files to `training_clean/`.
- Use `--in-place` to overwrite the originals in `training/` (creates a `.bak` alongside).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
import unicodedata


# Regex patterns for PG markers
RE_START = re.compile(r"(?is)^.*?\*\*\*\s*START OF\s+(THIS|THE)\s+PROJECT\s+GUTENBERG\s+EBOOK.*?\n")
RE_END_A = re.compile(r"(?is)\*\*\*\s*END OF\s+(THIS|THE)\s+PROJECT\s+GUTENBERG\s+EBOOK.*$")
RE_END_B = re.compile(r"(?is)End of( the)? Project Gutenberg.*$")

# Lines that indicate PG header-ish content near the very top
HEADER_HINTS = re.compile(
    r"(?i)^(?:\s*(?:the\s+)?project\s+gutenberg|produced by|e-?text prepared|e[ -]?book|online distributed proofreading|distributed proofreading team|pgdp\.net|pglaf\.org|gutenberg\.org|license|transcriber[’']?s note|transcribed from|google print project|public domain|character set encoding|release date|language:)"
)

TRANSCRIBER_HEADING = re.compile(r"(?i)^\s*transcriber[’']?s notes?:?\s*$")


def normalize_newlines(text: str) -> str:
    # Normalize to LF and strip BOM
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.lstrip("\ufeff")
    # NFC to normalize diacritics
    return unicodedata.normalize("NFC", text)


def strip_pg_header(text: str) -> str:
    """Remove PG header using explicit START marker when present, else heuristic."""
    m = RE_START.search(text)
    if m:
        return text[m.end() :]

    # Heuristic: remove initial block of header-ish lines (up to 300 lines)
    lines = text.split("\n")
    out = []
    i = 0
    n = min(len(lines), 300)
    # Skip initial empty lines
    while i < n and not lines[i].strip():
        i += 1
    # Skip lines that look like PG header/meta
    while i < n and (not lines[i].strip() or HEADER_HINTS.match(lines[i])):
        i += 1
    # Additionally, if a bracketed transcriber's note begins right away, skip that block
    if i < len(lines) and lines[i].lstrip().startswith("[") and re.search(r"(?i)transcriber", lines[i]):
        # Skip until we hit a closing bracket on a line by itself or a blank line after a ']'
        i += 1
        while i < len(lines):
            if lines[i].strip().endswith("]"):
                i += 1
                break
            i += 1
        # consume following blank lines
        while i < len(lines) and not lines[i].strip():
            i += 1
    out.extend(lines[i:])
    return "\n".join(out)


def strip_transcriber_sections(text: str) -> str:
    """Remove 'Transcriber's Note(s)' sections.

    Handles both plain headings and bracketed blocks like
    "[Transcriber's Note: ...]" by removing until a strong break.
    """
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
            # Plain heading on its own line
        if TRANSCRIBER_HEADING.match(lines[i]):
            # Skip heading and subsequent lines until a strong break (double blank line)
            i += 1
            blank_run = 0
            while i < len(lines):
                if not lines[i].strip():
                    blank_run += 1
                else:
                    blank_run = 0
                if blank_run >= 2:
                    # keep exactly one blank line as separator
                    i += 0
                    break
                i += 1
            # consume any extra blanks
            while i < len(lines) and not lines[i].strip():
                i += 1
            # insert a single blank line to separate
            if out and out[-1].strip():
                out.append("")
            continue
        # Bracketed block starting with [Transcriber's Note: ...]
        if lines[i].lstrip().startswith("[") and re.search(r"(?i)transcriber", lines[i]):
            i += 1
            while i < len(lines):
                if lines[i].strip().endswith("]"):
                    i += 1
                    break
                i += 1
            # consume trailing blanks
            while i < len(lines) and not lines[i].strip():
                i += 1
            if out and out[-1].strip():
                out.append("")
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def strip_pg_footer(text: str) -> str:
    # Drop from END markers to end
    m = RE_END_A.search(text)
    if m:
        return text[: m.start()].rstrip()
    m = RE_END_B.search(text)
    if m:
        return text[: m.start()].rstrip()
    return text


def normalize_whitespace(text: str) -> str:
    # Trim trailing whitespace per line
    lines = [ln.rstrip() for ln in text.split("\n")]
    # Collapse multiple blank lines to a single blank
    out = []
    blank = 0
    for ln in lines:
        if ln.strip():
            out.append(ln)
            blank = 0
        else:
            if blank == 0:
                out.append("")
            blank = 1
    # Strip leading/trailing blanks
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out) + "\n"


def clean_text(text: str) -> str:
    text = normalize_newlines(text)
    text = strip_pg_header(text)
    text = strip_transcriber_sections(text)
    text = strip_pg_footer(text)
    text = normalize_whitespace(text)
    return text


def process_file(src: Path, dst: Path, in_place: bool = False) -> bool:
    try:
        raw = src.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(raw)
        if in_place:
            # Backup original
            backup = src.with_suffix(src.suffix + ".bak")
            if not backup.exists():
                shutil.copy2(src, backup)
            src.write_text(cleaned, encoding="utf-8")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(cleaned, encoding="utf-8")
        return True
    except Exception as e:
        print(f"Error processing {src}: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Standardize training text files (remove PG boilerplate)")
    ap.add_argument("--src", type=Path, default=Path("training"), help="Source directory of texts")
    ap.add_argument("--dst", type=Path, default=Path("training_clean"), help="Destination directory for cleaned texts")
    ap.add_argument("--in-place", action="store_true", help="Overwrite files in-place under --src (backs up to .bak)")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of files (for testing)")
    args = ap.parse_args()

    if args.in_place:
        args.dst = args.src

    src_dir: Path = args.src
    dst_dir: Path = args.dst

    files = sorted(src_dir.rglob("*.txt"))
    if args.limit:
        files = files[: args.limit]

    print(f"Found {len(files)} files under {src_dir}")
    ok = 0
    for f in files:
        rel = f.relative_to(src_dir)
        dst = (dst_dir / rel) if not args.in_place else f
        if process_file(f, dst, in_place=args.in_place):
            ok += 1
    print(f"Cleaned {ok}/{len(files)} files -> {dst_dir}")


if __name__ == "__main__":
    main()
