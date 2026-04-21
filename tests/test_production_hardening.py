import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corpus.gutenberg_catalog import build_gutenberg_catalog_index, load_gutenberg_catalog_index
from corpus.manifests import build_corpus_manifests
from corpus.manifests import freeze_corpus_splits
from corpus.fetch_gutenberg import select_rsync_text_files
from corpus.fetch_gutenberg import fetch_gutenberg_http_range
from training.train_style_scorer import train_style_scorer
from training.train_style_scorer import _build_batch_author_ids
from tests.helpers import create_synthetic_books_manifest


class ProductionHardeningTests(unittest.TestCase):
    def test_author_labels_follow_concatenated_embedding_order(self) -> None:
        rows = [
            {"author1": "author:a", "author2": "author:b"},
            {"author1": "author:c", "author2": "author:d"},
        ]
        self.assertEqual(
            _build_batch_author_ids(rows),
            ["author:a", "author:c", "author:b", "author:d"],
        )

    def test_http_range_writes_chunk_specific_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            index_path = output_root / "meta" / "http_fetch_indexes" / "fetch_1_2.json"

            def _fake_fetch(book_id: int, **_: object) -> dict[str, object]:
                return {"gutenberg_id": book_id, "ok": True}

            with patch("corpus.fetch_gutenberg.fetch_gutenberg_http", side_effect=_fake_fetch):
                results = fetch_gutenberg_http_range(
                    1,
                    2,
                    output_root=output_root,
                    index_output_path=index_path,
                    max_workers=2,
                )

            self.assertEqual([row["gutenberg_id"] for row in results], [1, 2])
            self.assertTrue(index_path.exists())
            self.assertEqual(
                [row["gutenberg_id"] for row in json.loads(index_path.read_text(encoding="utf-8"))],
                [1, 2],
            )

    def test_corpus_manifests_store_relative_paths_and_materialize_raw_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = create_synthetic_books_manifest(tmp_path, author_count=2, books_per_author=2)
            corpus_root = tmp_path / "corpus"
            build_corpus_manifests(
                {
                    "input_books_manifest": manifest_path.as_posix(),
                    "output_root": corpus_root.as_posix(),
                    "min_clean_words": 10,
                    "min_clean_sentences": 2,
                    "min_author_books": 2,
                    "alpha_char_ratio_min": 0.1,
                }
            )

            rows = [
                json.loads(line)
                for line in (corpus_root / "meta" / "books_manifest_v1.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(rows)
            for row in rows:
                self.assertFalse(Path(row["source_path"]).is_absolute())
                self.assertFalse(Path(row["clean_path"]).is_absolute())
                self.assertTrue((corpus_root / row["source_path"]).exists())
                self.assertTrue((corpus_root / row["clean_path"]).exists())

    def test_select_rsync_text_files_shards_deterministically(self) -> None:
        relative_files = [f"0/{index}/pg{index}.txt" for index in range(10)]
        shard_zero = select_rsync_text_files(relative_files, max_files=8, shard_index=0, shard_count=3)
        shard_one = select_rsync_text_files(relative_files, max_files=8, shard_index=1, shard_count=3)
        shard_two = select_rsync_text_files(relative_files, max_files=8, shard_index=2, shard_count=3)
        self.assertEqual(shard_zero, ["0/0/pg0.txt", "0/3/pg3.txt", "0/6/pg6.txt"])
        self.assertEqual(shard_one, ["0/1/pg1.txt", "0/4/pg4.txt", "0/7/pg7.txt"])
        self.assertEqual(shard_two, ["0/2/pg2.txt", "0/5/pg5.txt"])

    def test_catalog_index_enriches_raw_scan_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            catalog_dir = output_root / "raw" / "gutenberg" / "catalog"
            catalog_dir.mkdir(parents=True, exist_ok=True)
            (catalog_dir / "pg_catalog.csv").write_text(
                "Text#,Title,Authors,Language,Issued,Subjects,Bookshelves,Type\n"
                "123,The Catalog Title,\"Doe, Jane\",en,2001-01-01,"
                "\"Adventure stories; England -- Social life and customs -- 19th century -- Fiction\","
                "\"Adventure\",Text\n",
                encoding="utf-8",
            )
            build_gutenberg_catalog_index(output_root=output_root)
            catalog = load_gutenberg_catalog_index(output_root)
            self.assertEqual(catalog["123"]["title"], "The Catalog Title")

            raw_dir = output_root / "raw" / "gutenberg" / "http"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "pg123.txt").write_text(
                "*** START OF THE PROJECT GUTENBERG EBOOK 123 ***\n"
                "Title: Header Title\n"
                "Author: Header Author\n"
                "Language: English\n\n"
                + ("A quiet sentence follows. " * 80),
                encoding="utf-8",
            )
            build_corpus_manifests(
                {
                    "output_root": output_root.as_posix(),
                    "min_clean_words": 10,
                    "min_clean_sentences": 2,
                    "min_author_books": 1,
                    "alpha_char_ratio_min": 0.1,
                }
            )
            rows = [
                json.loads(line)
                for line in (output_root / "meta" / "books_manifest_v1.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(rows[0]["title"], "The Catalog Title")
            self.assertEqual(rows[0]["author"], "Doe, Jane")
            self.assertEqual(rows[0]["genre"], "adventure")
            self.assertEqual(rows[0]["period_bucket"], "1800_1899")
            self.assertEqual(rows[0]["bookshelves"], ["Adventure"])
            self.assertIn("Adventure stories", rows[0]["subjects"])

    def test_benchmark_splits_reserve_noncanonical_authors_for_training_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = create_synthetic_books_manifest(tmp_path, author_count=8, books_per_author=3)
            rows = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            trimmed_rows = []
            for row in rows:
                author_id = row["author_id"]
                title = row["title"]
                if author_id == "author:author_06" and "Book 02" in title:
                    continue
                if author_id == "author:author_07" and ("Book 01" in title or "Book 02" in title):
                    continue
                trimmed_rows.append(row)
            manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in trimmed_rows),
                encoding="utf-8",
            )

            corpus_root = tmp_path / "corpus"
            corpus_config = {
                "input_books_manifest": manifest_path.as_posix(),
                "output_root": corpus_root.as_posix(),
                "min_clean_words": 10,
                "min_clean_sentences": 2,
                "min_author_books": 3,
                "benchmark_split_requires_author_track": True,
                "alpha_char_ratio_min": 0.1,
            }
            build_corpus_manifests(corpus_config)
            freeze_corpus_splits(corpus_config)

            dev_split = json.loads((corpus_root / "splits" / "benchmark_dev_v1.json").read_text(encoding="utf-8"))
            test_split = json.loads((corpus_root / "splits" / "benchmark_test_v1.json").read_text(encoding="utf-8"))
            train_split = json.loads((corpus_root / "splits" / "scorer_train_v1.json").read_text(encoding="utf-8"))

            auxiliary_authors = {"author:author_06", "author:author_07"}
            self.assertTrue(auxiliary_authors.isdisjoint(dev_split["author_ids"]))
            self.assertTrue(auxiliary_authors.isdisjoint(test_split["author_ids"]))
            self.assertTrue(auxiliary_authors.issubset(set(train_split["author_ids"])))

    def test_train_style_scorer_falls_back_to_bow_when_transformer_bootstrap_fails(self) -> None:
        config = {"artifacts_root": "/tmp/stylebench-test"}
        with patch("training.train_style_scorer._train_transformer_scorer", side_effect=OSError("offline")):
            with patch("training.train_style_scorer._train_bow_scorer", return_value={"model_dir": "bow"}) as bow_mock:
                result = train_style_scorer(config)
        self.assertEqual(result, {"model_dir": "bow"})
        bow_mock.assert_called_once()
        self.assertEqual(bow_mock.call_args.args[0]["training_backend"], "bow")
        self.assertIn("offline", bow_mock.call_args.args[0]["transformer_fallback_reason"])


if __name__ == "__main__":
    unittest.main()
