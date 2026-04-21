import tempfile
import unittest
from pathlib import Path

from scripts.download_scorer_from_hf import validate_downloaded_model_dir
from scripts.push_scorer_to_hf import prepare_upload_folder


class ScorerArtifactScriptTests(unittest.TestCase):
    def _write_minimal_model_dir(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "scorer_manifest.json").write_text("{}", encoding="utf-8")

    def test_prepare_upload_folder_copies_parent_calibration_into_upload_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer_root = Path(tmpdir) / "scorer"
            model_dir = scorer_root / "final"
            self._write_minimal_model_dir(model_dir)
            calibration_payload = '{"selection": {"chosen": "logistic"}}'
            (scorer_root / "style_calibration_v1.json").write_text(calibration_payload, encoding="utf-8")

            upload_dir, temp_dir = prepare_upload_folder(model_dir)
            try:
                self.assertTrue((upload_dir / "style_calibration_v1.json").exists())
                self.assertEqual(
                    (upload_dir / "style_calibration_v1.json").read_text(encoding="utf-8"),
                    calibration_payload,
                )
            finally:
                if temp_dir is not None:
                    temp_dir.cleanup()

    def test_validate_downloaded_model_dir_requires_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "final"
            self._write_minimal_model_dir(model_dir)

            with self.assertRaises(SystemExit):
                validate_downloaded_model_dir(model_dir)

            (model_dir / "style_calibration_v1.json").write_text("{}", encoding="utf-8")
            validate_downloaded_model_dir(model_dir)


if __name__ == "__main__":
    unittest.main()
