import json
import tempfile
import unittest
from pathlib import Path

from eval.style_scoring import StyleScorer
from training.style_text import STYLE_MASKED_TEXT_VIEW


class StyleScoringTests(unittest.TestCase):
    def _write_minimal_model_dir(self, model_dir: Path, **extra_config: object) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(
            json.dumps(
                {"model_type": "bag_of_words_style_scorer_v1", "hashing_dim": 32, **extra_config}
            ),
            encoding="utf-8",
        )
        (model_dir / "scorer_manifest.json").write_text(json.dumps({"model_name": "test_scorer"}), encoding="utf-8")

    def test_style_scorer_loads_legacy_parent_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer_root = Path(tmpdir) / "scorer"
            model_dir = scorer_root / "final"
            self._write_minimal_model_dir(model_dir)
            (scorer_root / "style_calibration_v1.json").write_text(
                json.dumps(
                    {
                        "style_calibration": {"method": "logistic", "coef": 2.0, "intercept": -0.5},
                        "selection": {"chosen": "logistic"},
                    }
                ),
                encoding="utf-8",
            )

            scorer = StyleScorer(model_dir)

            self.assertEqual(scorer.calibration_coef, 2.0)
            self.assertEqual(scorer.calibration_intercept, -0.5)
            self.assertIn("calibrated", scorer.score_pair("A quiet harbor.", "A quiet harbor."))

    def test_style_scorer_honors_identity_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "final"
            self._write_minimal_model_dir(model_dir)
            (model_dir / "style_calibration_v1.json").write_text(
                json.dumps(
                    {
                        "style_calibration": {"method": "identity", "coef": 9.0, "intercept": 9.0},
                        "selection": {"chosen": "identity"},
                    }
                ),
                encoding="utf-8",
            )

            scorer = StyleScorer(model_dir)

            self.assertIsNone(scorer.calibration_coef)
            self.assertIsNone(scorer.calibration_intercept)
            self.assertNotIn("calibrated", scorer.score_pair("A quiet harbor.", "A quiet harbor."))

    def test_style_masked_text_view_reduces_name_sensitivity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_model_dir = Path(tmpdir) / "raw_final"
            self._write_minimal_model_dir(raw_model_dir)
            masked_model_dir = Path(tmpdir) / "masked_final"
            self._write_minimal_model_dir(
                masked_model_dir,
                text_view=STYLE_MASKED_TEXT_VIEW,
            )

            raw_scorer = StyleScorer(raw_model_dir)
            masked_scorer = StyleScorer(masked_model_dir)
            first = "Alice met Brown in London while the harbor stayed quiet."
            second = "Charles met Diana in Paris while the harbor stayed quiet."
            raw_score = float(raw_scorer.score_pair(first, second)["score_0_1"])
            masked_score = float(masked_scorer.score_pair(first, second)["score_0_1"])
            self.assertGreater(masked_score, raw_score)

    def test_dual_view_scoring_reports_blended_similarity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "final"
            self._write_minimal_model_dir(
                model_dir,
                text_view=STYLE_MASKED_TEXT_VIEW,
                score_text_views=["raw", STYLE_MASKED_TEXT_VIEW],
                blend_weights={"raw": 0.35, STYLE_MASKED_TEXT_VIEW: 0.65},
            )
            scorer = StyleScorer(model_dir)

            first = "Alice met Brown in London while the harbor stayed quiet."
            second = "Charles met Diana in Paris while the harbor stayed quiet."
            payload = scorer.score_pair(first, second)

            self.assertIn("raw_similarity", payload)
            self.assertIn("masked_similarity", payload)
            self.assertIn("blended_similarity", payload)
            self.assertGreater(float(payload["masked_similarity"]), float(payload["raw_similarity"]))
            self.assertGreaterEqual(float(payload["blended_similarity"]), float(payload["raw_similarity"]))
            self.assertLessEqual(float(payload["blended_similarity"]), float(payload["masked_similarity"]))

    def test_chunked_scoring_can_focus_on_best_matching_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_dir = Path(tmpdir) / "baseline"
            self._write_minimal_model_dir(baseline_dir)
            chunked_dir = Path(tmpdir) / "chunked"
            self._write_minimal_model_dir(
                chunked_dir,
                chunk_size_words=6,
                chunk_overlap_words=2,
                chunk_aggregation="topk_mean",
                chunk_top_k=1,
            )

            baseline = StyleScorer(baseline_dir)
            chunked = StyleScorer(chunked_dir)
            reference = "harbor bells drifted above the wharf at dusk"
            hypothesis = (
                "harbor bells drifted above the wharf at dusk "
                "engine ledgers asphalt invoices warehouse freight schedules"
            )

            baseline_score = float(baseline.score_pair(hypothesis, reference)["score_0_1"])
            chunked_payload = chunked.score_pair(hypothesis, reference)
            chunked_score = float(chunked_payload["score_0_1"])

            self.assertGreater(chunked_score, baseline_score)
            self.assertEqual(chunked_payload["aggregate"], "top1_mean")


if __name__ == "__main__":
    unittest.main()
