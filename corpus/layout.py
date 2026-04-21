from __future__ import annotations

from pathlib import Path


def resolve_with_root(root: str | Path, stored_path: str | Path) -> Path:
    path = Path(stored_path)
    if path.is_absolute():
        return path
    return Path(root) / path


def relativize_to_root(path: str | Path, root: str | Path) -> str:
    source = Path(path)
    base = Path(root)
    try:
        return source.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return source.as_posix()


def is_within_root(path: str | Path, root: str | Path) -> bool:
    source = Path(path)
    base = Path(root)
    try:
        source.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False
