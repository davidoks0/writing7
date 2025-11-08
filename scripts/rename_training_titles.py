#!/usr/bin/env python3
import argparse
import os
import re
import sys
from pathlib import Path


EXCLUDE_WORDS = {
    "PROJECT", "GUTENBERG", "EBOOK", "EBOOKS", "ETEXT", "ETEXTS", "COPYRIGHT", "SMALL", "PRINT",
    "LICENSE", "LICENCE", "FOUNDATION", "NEWSLETTER", "HTTP", "WWW", "TRANSCRIBER",
    "PRODUCED", "PROOFREADING", "PROOFREADERS", "DISTRIBUTED", "ONLINE", "TEAM",
    "BY", "RELEASE", "UPDATED", "CATALOG", "CATALOGUE", "EDITION", "VERSION", "ELECTRONIC",
    "LIBRARY", "INC", "COLLEGE",
    # Legal/license speech common in PG headers
    "SERVICE", "DOWNLOAD", "MEMBERSHIP", "ACCESS", "REFUND", "REPLACEMENT",
    "WARRANTY", "DISCLAIMERS", "DISCLAIMER", "INDEMNITY", "DAMAGES", "DAMAGE",
    "COST", "EXPENSE", "LIABILITY", "NOTICE", "AGREEMENT", "COPYING",
    # Publisher/organization words
    "PUBLISHING", "COMPANY", "CO", "INCORPORATED", "LLC", "LTD",
    # Illustration/figure words
    "ILLUSTRATION", "ILLUSTRATIONS", "PLATE", "FIGURE", "FIG.",
    # Generic sections to avoid
    "CONTENTS", "PREFACE", "INTRODUCTION", "INDEX", "ACKNOWLEDGMENTS", "ACKNOWLEDGEMENTS",
    "FOREWORD", "AFTERWORD", "EPILOGUE", "PROLOGUE", "DEDICATION", "COPYRIGHTS",
    # Structural words
    "CHAPTER", "BOOK", "VOLUME", "PART", "CANTO", "CANTICLE",
}


def clean_title_to_filename(title: str, max_len: int = 120) -> str:
    # Normalize whitespace and punctuation
    t = title.strip().lower()
    # Replace apostrophes like alice's -> alices
    t = re.sub(r"'s\b", "s", t)
    t = re.sub(r"'", "", t)
    # Replace non-alphanumeric sequences with underscore
    t = re.sub(r"[^a-z0-9]+", "_", t)
    # Collapse multiple underscores
    t = re.sub(r"_+", "_", t).strip("_")
    # Truncate to max_len while keeping extension space
    if len(t) > max_len:
        t = t[:max_len].rstrip("_")
    if not t:
        t = "untitled"
    return f"{t}.txt"


def is_mostly_upper(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return False
    uppers = sum(1 for c in letters if c.isupper())
    return uppers / max(1, len(letters)) >= 0.7


def looks_like_title_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # Exclude very short or very long lines
    if len(s) < 6 or len(s) > 120:
        return False
    # Exclude list/TOC items starting with Roman numerals
    if re.match(r"^\s*[IVXLCDM]+[).\s]", s):
        return False
    # Exclude list/TOC items starting with letters+dot (A., B., I.)
    if re.match(r"^\s*[A-Z]\.", s):
        return False
    # Exclude common boilerplate/credit patterns
    if re.search(r"\b(produced|transcribed|scanned|prepared|distributed)\b", s, re.IGNORECASE):
        return False
    if re.search(r"https?://|www\\.", s, re.IGNORECASE):
        return False
    if s.lower().startswith(("and ", "by ", "from ", "with ")):
        return False
    if re.search(r"\b(translated\s+by|edited\s+by)\b", s, re.IGNORECASE):
        return False
    # Exclude publisher/address-like lines
    if re.search(r"\b(street|st\.?|road|rd\.?|row|square|avenue|ave\.?|press|printer|printers|publishers?|bros?\.?|&\s*co\.?|ltd\.?|inc\.?|strand|holborn|nassau)\b", s, re.IGNORECASE):
        return False
    # Exclude common ephemera/economics markers
    if re.search(r"\b(price|net|cents?)\b", s, re.IGNORECASE):
        return False
    if re.search(r"\b(vol\.?|no\.?|number)\b", s, re.IGNORECASE):
        return False
    # Exclude typical boilerplate/noise
    up = re.sub(r"[^A-Z]", " ", s.upper())
    words = {w for w in up.split() if w}
    if words & EXCLUDE_WORDS:
        return False
    # Avoid chapter headings and numeric-only
    if re.search(r"\bCHAPTER\b|\bBOOK\b|\bVOLUME\b", s, re.IGNORECASE):
        return False
    if re.match(r"^\d+[:.\s]", s):
        return False
    # Prefer all-caps or Title Case-ish lines
    if is_mostly_upper(s):
        # Avoid very short single-word shouty lines (likely ads/headers)
        if len(s.split()) == 1 and len(s) < 10:
            return False
        return True
    # Heuristic: many words starting with capitals and not shouting
    tokens = [t for t in re.split(r"\s+", s) if t]
    if 2 <= len(tokens) <= 12:
        cap_initials = sum(1 for t in tokens if re.match(r"^[A-Z][a-z'\-]+$", t))
        if cap_initials >= max(2, len(tokens) // 2):
            return True
    return False


def ascii_art_heavy(line: str) -> bool:
    if not line:
        return False
    specials = sum(1 for c in line if c in "|+-=_*#[]{}~`^\\/")
    return specials >= max(5, len(line) * 0.3)


def extract_title(text: str) -> tuple[str | None, str]:
    lines = text.splitlines()
    def merge_adjacent_title_lines(idx: int, base: str) -> str:
        s = base.strip()
        # Merge preceding short article line (e.g., THE, A, AN), skipping one blank if needed
        steps = 0
        j = idx - 1
        while j >= 0 and steps < 2:
            prev_raw = lines[j]
            prev = prev_raw.strip()
            if not prev:
                j -= 1
                steps += 1
                continue
            if is_mostly_upper(prev) and len(prev) <= 6 and not re.search(r"\d", prev):
                if prev.upper() in {"THE", "A", "AN", "LE", "LA", "EL"}:
                    s = f"{prev} {s}"
            break
        return s
    # 1) Explicit Title: field (search deeper; some files have long preambles)
    for i, line in enumerate(lines[:400]):
        m = re.match(r"^\s*Title\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
        if m:
            # Validate surrounding context to avoid false positives inside body text
            context = "\n".join(lines[max(0, i-30): i+31])
            if not re.search(r"project\s+gutenberg|author\s*[:\-]|release\s+date\s*[:\-]", context, re.IGNORECASE):
                continue
            candidate = m.group(1).strip()
            if candidate:
                return candidate, "high"
            # Sometimes next line has the title
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if nxt:
                    return nxt, "high"

    # 2) Project Gutenberg eBook header patterns (robust, case-insensitive)
    pg_patterns = [
        r"project\s+gutenberg[^\n]*?\b(e?text|ebook)\b\s+of\s+(.+?)(?:,\s+by\b|\s+by\b|$)",
        r"project\s+gutenberg'?s\s+(.+?)(?:,\s+by\b|\s+by\b|$)",
    ]
    head = "\n".join(lines[:1500])
    for pat in pg_patterns:
        m = re.search(pat, head, re.IGNORECASE)
        if m:
            # Group index depends on pattern
            cand = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
            cand = cand.strip()
            # Remove enclosing quotes if present
            cand = re.sub(r"^[\"'“”]+|[\"'“”]+$", "", cand)
            if cand:
                return cand, "high"

    # 2b) Byline proximity: line with 'by <Author>' preceded by a good title line
    for idx in range(min(600, len(lines))):
        line = lines[idx]
        if re.search(r"^\s*(by|translated\s+by|edited\s+by)\b", line.strip(), re.IGNORECASE):
            # Look back a few lines for a plausible title
            for back in range(max(0, idx - 5), idx):
                prev = lines[back].strip()
                if looks_like_title_line(prev):
                    return merge_adjacent_title_lines(back, prev), "medium"

    # 2c) Early title block: a plausible title near the top followed soon by a BY/TRANSLATED BY line
    max_top = min(80, len(lines))
    for idx in range(max_top):
        s = lines[idx]
        if not looks_like_title_line(s):
            continue
        lookahead = " ".join(lines[idx+1:idx+10])
        if re.search(r"\b(BY|TRANSLATED\s+BY|EDITED\s+BY)\b", lookahead, re.IGNORECASE):
            return merge_adjacent_title_lines(idx, s.strip()), "medium"

    # 3) First strong title-ish line in the first 200 lines (score candidates)
    candidates: list[tuple[int, int, str]] = []  # (score, index, line)
    for idx, line in enumerate(lines[:200]):
        if not looks_like_title_line(line):
            continue
        s = line.strip()
        score = 0
        # Base by length
        L = len(s)
        if 10 <= L <= 60:
            score += 3
        elif 6 <= L <= 80:
            score += 1
        # Token count preference
        tokens = [t for t in re.split(r"\s+", s) if t]
        if 2 <= len(tokens) <= 12:
            score += 2
        # Uppercase boost
        if is_mostly_upper(s):
            score += 1
        # Penalize single short possessive
        if len(tokens) == 1 and re.search(r"'[sS]$", s) and L <= 12:
            score -= 2
        # Context penalties for ascii-art borders
        prev = lines[idx - 1] if idx > 0 else ""
        nxt = lines[idx + 1] if idx + 1 < len(lines) else ""
        if ascii_art_heavy(prev) or ascii_art_heavy(nxt):
            score -= 2
        # Penalize lines containing digits (likely dates/prices), allow roman numerals in isolation
        if re.search(r"\d", s):
            score -= 3
        # Boost if a BY/TRANSLATED BY line appears shortly after (typical title placement)
        lookahead = " ".join(lines[idx+1:idx+6])
        if re.search(r"\b(BY|TRANSLATED\s+BY|EDITED\s+BY)\b", lookahead, re.IGNORECASE):
            score += 2
        # Additional early-position boost
        if idx < 15 and is_mostly_upper(s):
            score += 2
        # Penalize likely author name lines: 2-4 tokens, all capitalized words, followed by dates
        toks = s.split()
        if 2 <= len(toks) <= 4 and all(t.isalpha() and t[0].isupper() for t in toks):
            next5 = " ".join(lines[idx+1:idx+6])
            if re.search(r"\(\s*\d{3,4}\s*[-–—]\s*\d{2,4}\s*\)|\b\d{4}\b", next5):
                score -= 3
        # Penalize if within a CONTENTS/CREDITS section
        back30 = " ".join(lines[max(0, idx-30):idx])
        if re.search(r"\b(CONTENTS|CREDITS)\b", back30, re.IGNORECASE):
            score -= 5
        # Boost for periodical/magazine uppercase single/multi-word name followed by date in next lines
        if 1 <= len(tokens) <= 5 and is_mostly_upper(s) and not re.search(r"\d", s):
            window = " ".join(lines[idx+1:idx+6])
            if (
                re.search(r"\b(MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)\b", window, re.IGNORECASE)
                or re.search(r"\b(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\b", window, re.IGNORECASE)
            ):
                score += 3
        candidates.append((score, idx, s))
    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1]))
        best = candidates[0]
        if best[0] >= 1:
            return merge_adjacent_title_lines(best[1], best[2]), "low"
    
    # 4) Fallback: first non-empty, non-boilerplate line
    for line in lines[:300]:
        s = line.strip()
        if not s:
            continue
        up = re.sub(r"[^A-Z]", " ", s.upper())
        words = {w for w in up.split() if w}
        if words & EXCLUDE_WORDS:
            continue
        return s, "low"
    return None, "low"


def unique_filename(target_dir: Path, base_name: str) -> str:
    stem = base_name[:-4] if base_name.endswith(".txt") else base_name
    candidate = f"{stem}.txt"
    n = 1
    while (target_dir / candidate).exists():
        candidate = f"{stem}_{n}.txt"
        n += 1
    return candidate


def process_file(path: Path) -> tuple[str | None, str | None, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return None, f"read_error: {e}", "low"
    title, conf = extract_title(text)
    if not title:
        return None, "no_title_found", conf
    filename = clean_title_to_filename(title)
    return filename, None, conf


def main():
    ap = argparse.ArgumentParser(description="Rename training/*.txt to title-based filenames.")
    ap.add_argument("--dir", default="training", help="Directory containing .txt files")
    ap.add_argument("--apply", action="store_true", help="Actually rename files (default is dry-run)")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of files to process")
    ap.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    ap.add_argument("--min-confidence", choices=["low","medium","high"], default="medium", help="Minimum confidence to apply rename")
    ap.add_argument("--log", help="Write TSV log of old\tnew\tconfidence to this path")
    args = ap.parse_args()

    d = Path(args.dir)
    if not d.is_dir():
        print(f"error: directory not found: {d}", file=sys.stderr)
        sys.exit(1)

    txts = sorted(p for p in d.iterdir() if p.suffix == ".txt")
    if args.limit:
        txts = txts[: args.limit]

    renames = []
    errors = []
    rows = []
    for p in txts:
        new_name, err, conf = process_file(p)
        if err:
            errors.append((p.name, err))
            rows.append((p.name, "", conf, err))
            continue
        # Filter by confidence for actual rename, but always record row
        target_name = unique_filename(d, new_name)
        rows.append((p.name, target_name, conf, ""))
        # Determine if this meets min-confidence threshold
        order = {"low":0, "medium":1, "high":2}
        if order[conf] >= order[args.min_confidence]:
            renames.append((p, d / target_name, conf))

    # Report
    print(f"Found {len(renames)} rename candidates (>= {args.min_confidence}); {len(errors)} errors.")
    if args.verbose:
        for old, new, conf in renames[:50]:
            print(f"- {old.name} -> {new.name} [{conf}]")
        if len(renames) > 50:
            print(f"... and {len(renames) - 50} more")
        if errors:
            print("Errors:")
            for name, err in errors[:50]:
                print(f"- {name}: {err}")
    if args.log:
        try:
            with open(args.log, 'w', encoding='utf-8') as f:
                for old, new, conf, err in rows:
                    f.write(f"{old}\t{new}\t{conf}\t{err}\n")
        except Exception as e:
            print(f"failed_writing_log: {e}", file=sys.stderr)

    if not args.apply:
        print("Dry-run mode; no files renamed. Use --apply to perform renames.")
        return

    # Perform renames
    for old, new, _ in renames:
        try:
            if old == new:
                continue
            os.rename(old, new)
        except Exception as e:
            print(f"rename_failed: {old.name} -> {new.name}: {e}", file=sys.stderr)

    print(f"Renamed {len(renames)} files. {len(errors)} files skipped.")


if __name__ == "__main__":
    main()
