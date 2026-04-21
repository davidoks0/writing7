import unittest

from eval.benchmark_schema import BenchmarkTarget, PassageRecord
from eval.distractors import select_distractor_targets


class DistractorTests(unittest.TestCase):
    def test_distractor_selection_excludes_target_and_same_author(self) -> None:
        target = BenchmarkTarget(target_id="author:a", track="author", author_id="author:a", conditioning_book_ids=["book:a:1", "book:a:2"], evaluation_book_id="book:a:3").validate()
        candidate_same_author = BenchmarkTarget(target_id="author:a_alt", track="author", author_id="author:a", conditioning_book_ids=["book:a:4", "book:a:5"], evaluation_book_id="book:a:6").validate()
        candidate_other = BenchmarkTarget(target_id="author:b", track="author", author_id="author:b", conditioning_book_ids=["book:b:1", "book:b:2"], evaluation_book_id="book:b:3").validate()
        candidate_third = BenchmarkTarget(target_id="author:c", track="author", author_id="author:c", conditioning_book_ids=["book:c:1", "book:c:2"], evaluation_book_id="book:c:3").validate()
        candidate_targets = [target, candidate_same_author, candidate_other, candidate_third]
        books = {
            "book:a:3": {"period_bucket": "1800_1849", "genre": "novel"},
            "book:b:3": {"period_bucket": "1800_1849", "genre": "novel"},
            "book:c:3": {"period_bucket": "1850_1899", "genre": "novel"},
        }
        passages_by_target = {
            "author:a": {
                "conditioning": [PassageRecord("passage:a:0:5", "book:a:1", "author:a", text="a " * 60, start_sentence=0, end_sentence=5, word_count=60, char_count=120).validate()],
                "evaluation": [PassageRecord("passage:a:5:10", "book:a:3", "author:a", text="a " * 60, start_sentence=5, end_sentence=10, word_count=60, char_count=120).validate()],
            },
            "author:b": {
                "conditioning": [PassageRecord("passage:b:0:5", "book:b:1", "author:b", text="b " * 60, start_sentence=0, end_sentence=5, word_count=60, char_count=120).validate()],
                "evaluation": [PassageRecord("passage:b:5:10", "book:b:3", "author:b", text="b " * 60, start_sentence=5, end_sentence=10, word_count=60, char_count=120).validate()],
            },
            "author:c": {
                "conditioning": [PassageRecord("passage:c:0:5", "book:c:1", "author:c", text="c " * 60, start_sentence=0, end_sentence=5, word_count=60, char_count=120).validate()],
                "evaluation": [PassageRecord("passage:c:5:10", "book:c:3", "author:c", text="c " * 60, start_sentence=5, end_sentence=10, word_count=60, char_count=120).validate()],
            },
        }
        selected = select_distractor_targets(
            case_seed=42,
            track="author",
            target=target,
            candidate_targets=candidate_targets,
            target_book_meta=books["book:a:3"],
            target_eval_passages=passages_by_target["author:a"]["evaluation"],
            passages_by_target=passages_by_target,
            books_by_id=books,
            scorer=None,
        )
        self.assertTrue(all(candidate.target_id != target.target_id for candidate in selected))
        self.assertTrue(all(candidate.author_id != target.author_id for candidate in selected))

    def test_book_track_prefers_other_author_distractors_when_available(self) -> None:
        target = BenchmarkTarget(target_id="book:a:1", track="book", author_id="author:a", book_id="book:a:1").validate()
        candidate_same_author = BenchmarkTarget(target_id="book:a:2", track="book", author_id="author:a", book_id="book:a:2").validate()
        candidate_other_one = BenchmarkTarget(target_id="book:b:1", track="book", author_id="author:b", book_id="book:b:1").validate()
        candidate_other_two = BenchmarkTarget(target_id="book:c:1", track="book", author_id="author:c", book_id="book:c:1").validate()
        candidate_targets = [target, candidate_same_author, candidate_other_one, candidate_other_two]
        books = {
            "book:a:1": {"period_bucket": "1800_1849", "genre": "novel"},
            "book:a:2": {"period_bucket": "1800_1849", "genre": "novel"},
            "book:b:1": {"period_bucket": "1800_1849", "genre": "novel"},
            "book:c:1": {"period_bucket": "1800_1849", "genre": "novel"},
        }
        passages_by_target = {
            "book:a:1": {
                "conditioning": [PassageRecord("passage:a1:0:5", "book:a:1", "author:a", text="a " * 60, start_sentence=0, end_sentence=5, word_count=60, char_count=120).validate()],
                "evaluation": [PassageRecord("passage:a1:5:10", "book:a:1", "author:a", text="a " * 60, start_sentence=5, end_sentence=10, word_count=60, char_count=120).validate()],
            },
            "book:a:2": {
                "conditioning": [PassageRecord("passage:a2:0:5", "book:a:2", "author:a", text="a " * 60, start_sentence=0, end_sentence=5, word_count=60, char_count=120).validate()],
                "evaluation": [PassageRecord("passage:a2:5:10", "book:a:2", "author:a", text="a " * 60, start_sentence=5, end_sentence=10, word_count=60, char_count=120).validate()],
            },
            "book:b:1": {
                "conditioning": [PassageRecord("passage:b1:0:5", "book:b:1", "author:b", text="b " * 60, start_sentence=0, end_sentence=5, word_count=60, char_count=120).validate()],
                "evaluation": [PassageRecord("passage:b1:5:10", "book:b:1", "author:b", text="b " * 60, start_sentence=5, end_sentence=10, word_count=60, char_count=120).validate()],
            },
            "book:c:1": {
                "conditioning": [PassageRecord("passage:c1:0:5", "book:c:1", "author:c", text="c " * 60, start_sentence=0, end_sentence=5, word_count=60, char_count=120).validate()],
                "evaluation": [PassageRecord("passage:c1:5:10", "book:c:1", "author:c", text="c " * 60, start_sentence=5, end_sentence=10, word_count=60, char_count=120).validate()],
            },
        }
        selected = select_distractor_targets(
            case_seed=42,
            track="book",
            target=target,
            candidate_targets=candidate_targets,
            target_book_meta=books["book:a:1"],
            target_eval_passages=passages_by_target["book:a:1"]["evaluation"],
            passages_by_target=passages_by_target,
            books_by_id=books,
            scorer=None,
            target_count=2,
            book_track_same_author_policy="prefer_other_author",
        )
        self.assertEqual({candidate.target_id for candidate in selected}, {"book:b:1", "book:c:1"})


if __name__ == "__main__":
    unittest.main()
