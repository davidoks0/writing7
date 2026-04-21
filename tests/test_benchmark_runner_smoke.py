import json
import unittest
from pathlib import Path

from corpus.manifests import build_corpus_manifests, freeze_corpus_splits
from corpus.passage_pools import build_passage_pools
from eval.aggregate_benchmark_results import aggregate_benchmark_results
from eval.benchmark_v2 import run_benchmark
from eval.build_benchmark_manifests import build_benchmark_manifests
from tests.helpers import create_synthetic_books_manifest
from training.build_scorer_dataset import build_scorer_dataset
from training.calibrate_style_scorer import calibrate_style_scorer
from training.train_style_scorer import train_style_scorer


class BenchmarkRunnerSmokeTests(unittest.TestCase):
    def test_end_to_end_smoke_runner(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = create_synthetic_books_manifest(tmp_path, author_count=6, books_per_author=3)
            corpus_root = tmp_path / "corpus"
            artifacts_root = tmp_path / "artifacts"
            benchmark_root = tmp_path / "benchmark_data"

            corpus_config = {
                "input_books_manifest": manifest_path.as_posix(),
                "output_root": corpus_root.as_posix(),
                "builder_seed": 42,
                "min_clean_words": 250,
                "min_clean_sentences": 20,
                "min_author_books": 3,
                "alpha_char_ratio_min": 0.6,
                "passage_policy": {
                    "min_words": 40,
                    "max_words": 120,
                    "preferred_words": 80,
                    "min_sentences": 4,
                    "max_sentences": 10,
                    "region_buckets": 5,
                },
            }
            build_corpus_manifests(corpus_config)
            freeze_corpus_splits(corpus_config)
            build_passage_pools(corpus_config)

            scorer_config = {
                "corpus_root": corpus_root.as_posix(),
                "artifacts_root": artifacts_root.as_posix(),
                "split_path": (corpus_root / "splits" / "scorer_train_v1.json").as_posix(),
                "calibration_split_path": (corpus_root / "splits" / "scorer_calibration_v1.json").as_posix(),
                "builder_seed": 42,
                "max_features": 512,
                "hashing_dim": 128,
            }
            build_scorer_dataset(scorer_config)
            train_style_scorer(scorer_config)
            calibrate_style_scorer(scorer_config)

            benchmark_config = {
                "benchmark_version": "gutenberg_style_v1",
                "builder_seed": 42,
                "prompt_bank_path": "eval/benchmark_data/prompts_v1.json",
                "generation_profiles": {
                    "leaderboard_v1": {
                        "temperature": 0.8,
                        "top_p": 0.95,
                        "max_tokens": 200,
                        "n_samples_per_case": 1,
                        "sample_seeds": [11],
                    }
                },
                "passage_policy": {
                    "min_words": 40,
                    "max_words": 120,
                    "preferred_words": 80,
                    "min_sentences": 4,
                    "max_sentences": 10,
                    "region_buckets": 5,
                },
                "distractor_policy": {
                    "target_count": 5,
                    "min_count": 1,
                    "book_track_same_author_policy": "prefer_other_author",
                },
                "originality_thresholds": {
                    "char_8gram_overlap_max": 0.3,
                    "token_lcs_ratio_max": 0.2,
                    "joint_char_overlap_threshold": 0.2,
                    "joint_lcs_threshold": 0.15,
                },
                "fluency_thresholds": {
                    "min_words_valid": 50,
                    "max_words_valid": 400,
                    "max_repetition_rate_6gram": 0.4,
                },
                "bootstrap_resamples": 50,
            }
            benchmark_config_path = tmp_path / "benchmark_config.json"
            benchmark_config_path.write_text(json.dumps(benchmark_config), encoding="utf-8")
            build_benchmark_manifests(
                books_manifest=corpus_root / "meta" / "books_manifest_v1.jsonl",
                out_dir=benchmark_root,
                benchmark_config=benchmark_config_path,
                split_root=corpus_root / "splits",
                scorer_dir=artifacts_root / "scorer" / "final",
            )

            result_path = tmp_path / "results.jsonl"
            run_benchmark(
                track="author",
                split="test",
                cases_path=benchmark_root / "cases_author_test_v1.jsonl",
                prompts_path=benchmark_root / "prompts_v1.json",
                model="stub:fixed_prose",
                model_dir=artifacts_root / "scorer" / "final",
                out_path=result_path,
                config_path=benchmark_root / "config_v1.json",
                limit=2,
            )
            rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(rows)
            self.assertTrue(all("style_metrics" in row for row in rows))
            self.assertIn("full_target_book", rows[0]["originality_metrics"]["reference_group_metrics"])
            self.assertTrue((artifacts_root / "scorer" / "final" / "diagnostics_v1.json").exists())
            summary = aggregate_benchmark_results(result_path, tmp_path / "summary.json")
            self.assertGreaterEqual(summary["sample_count"], 1)
            self.assertIn("full_target_book_copy_free_rate", summary["metrics_all"])


if __name__ == "__main__":
    unittest.main()
