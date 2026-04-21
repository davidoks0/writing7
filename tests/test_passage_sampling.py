import unittest

from eval.passage_sampling import extract_passage_candidates, split_sentences


TEXT = (
    "Mr. Hale arrived before breakfast and found the windows open. "
    "The room seemed colder for its politeness. "
    "He waited by the mantel until Mrs. Vale entered with a note in her hand. "
    "She read the note twice, though she had already understood it. "
    "Nothing in her face altered quickly. "
    "That was why the small alteration mattered. "
    "Outside, the street carried a bright morning noise. "
    "Inside, the house began to listen."
)


class PassageSamplingTests(unittest.TestCase):
    def test_sentence_splitter_is_deterministic(self) -> None:
        first = split_sentences(TEXT)
        second = split_sentences(TEXT)
        self.assertEqual([row.text for row in first], [row.text for row in second])
        self.assertGreaterEqual(len(first), 6)

    def test_extract_passage_candidates_respects_thresholds(self) -> None:
        passages = extract_passage_candidates(
            TEXT * 8,
            book_id="book:test:one",
            author_id="author:test",
            min_words=40,
            max_words=120,
            min_sentences=4,
            max_sentences=10,
        )
        self.assertTrue(passages)
        self.assertTrue(all(40 <= row.word_count <= 120 for row in passages))
        self.assertTrue(all(4 <= (row.end_sentence - row.start_sentence) <= 10 for row in passages))


if __name__ == "__main__":
    unittest.main()
