import json
import unittest

from eval.aggregate_benchmark_results import aggregate_benchmark_results


class AggregateResultsTests(unittest.TestCase):
    def test_aggregate_results_outputs_ci_structure(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "results.jsonl"
            rows = [
                {
                    "benchmark_version": "gutenberg_style_v1",
                    "track": "author",
                    "split": "test",
                    "case_id": "case:author:test:one:prompt_01",
                    "target_id": "author:one",
                    "prompt_id": "prompt:interpersonal:01",
                    "generator": {"provider": "stub", "model_name": "fixed_prose"},
                    "originality_metrics": {
                        "copy_flag": False,
                        "entity_transplant_flag": False,
                        "originality_pass": True,
                        "reference_group_metrics": {
                            "conditioning": {"copy_flag": False},
                            "target_evaluation": {"copy_flag": False},
                            "full_target_book": {"copy_flag": False},
                        },
                    },
                    "style_metrics": {"style_win_rate_case": 1.0, "style_margin_case": 0.2, "top1_target_case": 1, "mrr_case": 1.0, "style_percentile_case": 0.8},
                    "valid_flags": {"originality_pass": True, "fluency_pass": True, "valid": True},
                },
                {
                    "benchmark_version": "gutenberg_style_v1",
                    "track": "author",
                    "split": "test",
                    "case_id": "case:author:test:two:prompt_01",
                    "target_id": "author:two",
                    "prompt_id": "prompt:interpersonal:01",
                    "generator": {"provider": "stub", "model_name": "fixed_prose"},
                    "originality_metrics": {
                        "copy_flag": True,
                        "entity_transplant_flag": False,
                        "originality_pass": False,
                        "reference_group_metrics": {
                            "conditioning": {"copy_flag": False},
                            "target_evaluation": {"copy_flag": True},
                            "full_target_book": {"copy_flag": True},
                        },
                    },
                    "style_metrics": {"style_win_rate_case": 0.5, "style_margin_case": 0.1, "top1_target_case": 0, "mrr_case": 0.5, "style_percentile_case": 0.6},
                    "valid_flags": {"originality_pass": False, "fluency_pass": True, "valid": False},
                },
            ]
            with result_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            summary = aggregate_benchmark_results(result_path)
            self.assertEqual(summary["sample_count"], 2)
            self.assertIn("style_win_rate_valid", summary["bootstrap_ci_95"])
            self.assertIn("leaderboard_row", summary)
            self.assertIn("leaderboard_markdown", summary)
            self.assertAlmostEqual(summary["metrics_all"]["target_evaluation_copy_free_rate"], 0.5)
            self.assertAlmostEqual(summary["metrics_all"]["full_target_book_copy_free_rate"], 0.5)
            self.assertAlmostEqual(summary["leaderboard_row"]["style_mimicry_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
