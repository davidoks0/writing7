import json
import unittest

from eval.build_benchmark_leaderboard import build_benchmark_leaderboard


class BenchmarkLeaderboardTests(unittest.TestCase):
    def test_build_leaderboard_ranks_models_by_style_mimicry_score(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            first_summary = tmp_path / "first_summary.json"
            second_summary = tmp_path / "second_summary.json"
            first_summary.write_text(
                json.dumps(
                    {
                        "benchmark_version": "gutenberg_style_v1",
                        "track": "author",
                        "split": "test",
                        "generator": {"provider": "stub", "model_name": "model_alpha"},
                        "sample_count": 10,
                        "valid_sample_count": 8,
                        "metrics_all": {
                            "valid_rate": 0.8,
                            "originality_pass_rate": 0.9,
                            "fluency_pass_rate": 1.0,
                        },
                        "metrics_valid": {
                            "style_win_rate_mean": 0.9,
                            "style_margin_mean": 0.2,
                            "top1_target_accuracy_mean": 0.85,
                            "style_percentile_valid_mean": 0.8,
                        },
                    }
                ),
                encoding="utf-8",
            )
            second_summary.write_text(
                json.dumps(
                    {
                        "benchmark_version": "gutenberg_style_v1",
                        "track": "author",
                        "split": "test",
                        "generator": {"provider": "stub", "model_name": "model_beta"},
                        "sample_count": 10,
                        "valid_sample_count": 9,
                        "metrics_all": {
                            "valid_rate": 0.9,
                            "originality_pass_rate": 1.0,
                            "fluency_pass_rate": 1.0,
                        },
                        "metrics_valid": {
                            "style_win_rate_mean": 0.7,
                            "style_margin_mean": 0.15,
                            "top1_target_accuracy_mean": 0.7,
                            "style_percentile_valid_mean": 0.75,
                        },
                    }
                ),
                encoding="utf-8",
            )

            leaderboard = build_benchmark_leaderboard([first_summary, second_summary], tmp_path / "leaderboard.json")
            self.assertEqual(leaderboard["row_count"], 2)
            self.assertEqual(leaderboard["rows"][0]["model_name"], "model_alpha")
            self.assertIn("| rank | model | provider | track | split |", leaderboard["markdown"])
            self.assertIn("model_name", leaderboard["csv"])
            self.assertTrue((tmp_path / "leaderboard.csv").exists())
            self.assertTrue((tmp_path / "leaderboard.md").exists())

    def test_build_leaderboard_uses_style_margin_as_late_tie_break(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            first_summary = tmp_path / "first_summary.json"
            second_summary = tmp_path / "second_summary.json"
            shared = {
                "benchmark_version": "gutenberg_style_v1",
                "track": "author",
                "split": "test",
                "sample_count": 10,
                "valid_sample_count": 8,
                "metrics_all": {
                    "valid_rate": 0.8,
                    "originality_pass_rate": 0.9,
                    "fluency_pass_rate": 1.0,
                },
            }
            first_summary.write_text(
                json.dumps(
                    {
                        **shared,
                        "generator": {"provider": "stub", "model_name": "higher_margin"},
                        "metrics_valid": {
                            "style_win_rate_mean": 0.9,
                            "style_margin_mean": 0.25,
                            "top1_target_accuracy_mean": 0.85,
                            "style_percentile_valid_mean": 0.8,
                        },
                    }
                ),
                encoding="utf-8",
            )
            second_summary.write_text(
                json.dumps(
                    {
                        **shared,
                        "generator": {"provider": "stub", "model_name": "lower_margin"},
                        "metrics_valid": {
                            "style_win_rate_mean": 0.9,
                            "style_margin_mean": 0.2,
                            "top1_target_accuracy_mean": 0.85,
                            "style_percentile_valid_mean": 0.8,
                        },
                    }
                ),
                encoding="utf-8",
            )

            leaderboard = build_benchmark_leaderboard([first_summary, second_summary])
            self.assertEqual(leaderboard["rows"][0]["model_name"], "higher_margin")


if __name__ == "__main__":
    unittest.main()
