#!/usr/bin/env python3
"""
Download a Hugging Face model repo snapshot to a local directory and print the path
to the 'final/' subfolder expected by our inference/benchmark code.

Usage:
  python scripts/download_hf_model.py \
    --repo-id your-org/writing7-book-matcher-contrastive-v1 \
    --dest ./external/writing7-book-matcher-contrastive-v1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main():
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        print("ERROR: huggingface_hub is not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(2)

    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--dest", required=True, help="Destination directory for the snapshot")
    args = p.parse_args()

    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(repo_id=args.repo_id, local_dir=str(dest), local_dir_use_symlinks=False)
    final_path = Path(local) / "final"
    if not final_path.exists():
        print(f"WARNING: 'final/' not found under snapshot. Contents are under: {local}", file=sys.stderr)
    print(str(final_path if final_path.exists() else local))


if __name__ == "__main__":
    main()

