import tempfile
import unittest
from pathlib import Path

from corpus.manifests import build_corpus_manifests, freeze_corpus_splits
from corpus.passage_pools import build_passage_pools
from eval.benchmark_io import load_json, load_jsonl
from tests.helpers import create_synthetic_books_manifest
from training.build_scorer_dataset import build_scorer_dataset


class BuildScorerDatasetTests(unittest.TestCase):
    def test_dataset_contains_style_masked_text_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = create_synthetic_books_manifest(tmp_path, author_count=6, books_per_author=3)
            corpus_root = tmp_path / "corpus"
            artifacts_root = tmp_path / "artifacts"
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
            build_scorer_dataset(
                {
                    "corpus_root": corpus_root.as_posix(),
                    "artifacts_root": artifacts_root.as_posix(),
                    "split_path": (corpus_root / "splits" / "scorer_train_v1.json").as_posix(),
                    "builder_seed": 42,
                }
            )

            rows = load_jsonl(artifacts_root / "scorer" / "datasets" / "train_pairs_v1.jsonl")
            meta = load_json(artifacts_root / "scorer" / "datasets" / "scorer_dataset_meta_v1.json")
            self.assertTrue(rows)
            self.assertIn("style_text1", rows[0])
            self.assertIn("style_text2", rows[0])
            self.assertIn("content_cluster1", rows[0])
            self.assertIn("content_cluster2", rows[0])
            self.assertIn("same_content_cluster", rows[0])
            self.assertNotEqual(rows[0]["style_text1"], "")
            self.assertGreaterEqual(meta["content_labels"]["cluster_count"], 1)


if __name__ == "__main__":
    unittest.main()
