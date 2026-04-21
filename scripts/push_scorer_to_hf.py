from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def _validate_model_dir(model_dir: Path) -> None:
    required_files = ["config.json", "scorer_manifest.json"]
    missing = [name for name in required_files if not (model_dir / name).exists()]
    if missing:
        raise SystemExit(f"model directory is missing required files: {', '.join(missing)}")


def prepare_upload_folder(model_dir: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    calibration_in_model_dir = model_dir / "style_calibration_v1.json"
    calibration_in_parent = model_dir.parent / "style_calibration_v1.json"
    if calibration_in_model_dir.exists():
        return model_dir, None
    if not calibration_in_parent.exists():
        raise SystemExit(
            "style_calibration_v1.json is required for benchmark-ready scorer uploads; "
            "run calibration first or place the file inside the model directory."
        )
    temp_dir = tempfile.TemporaryDirectory()
    upload_dir = Path(temp_dir.name) / model_dir.name
    shutil.copytree(model_dir, upload_dir)
    shutil.copy2(calibration_in_parent, upload_dir / "style_calibration_v1.json")
    return upload_dir, temp_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a trained scorer artifact directory to Hugging Face Hub.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"huggingface_hub is required: {exc}")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"model directory does not exist: {model_dir}")
    _validate_model_dir(model_dir)
    upload_dir, temp_dir = prepare_upload_folder(model_dir)

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    try:
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=upload_dir.as_posix(),
            commit_message="Upload style scorer artifact",
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
    print(f"Uploaded {upload_dir} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
