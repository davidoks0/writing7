from __future__ import annotations

import argparse
from pathlib import Path


def validate_downloaded_model_dir(destination: Path) -> None:
    required_files = ["config.json", "scorer_manifest.json", "style_calibration_v1.json"]
    missing = [name for name in required_files if not (destination / name).exists()]
    if missing:
        raise SystemExit(
            "downloaded scorer artifact is incomplete for benchmark use; missing "
            + ", ".join(missing)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a scorer artifact directory from Hugging Face Hub.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"huggingface_hub is required: {exc}")

    destination = Path(args.out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        local_dir=destination.as_posix(),
        local_dir_use_symlinks=False,
    )
    validate_downloaded_model_dir(destination)
    print(f"Downloaded https://huggingface.co/{args.repo_id} to {destination}")


if __name__ == "__main__":
    main()
