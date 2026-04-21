import unittest
from types import SimpleNamespace
from unittest.mock import patch

from eval.benchmark_schema import BenchmarkTarget, PromptRecord
from eval.build_benchmark_manifests import _build_reference_distribution


class ReferenceDistributionTests(unittest.TestCase):
    def test_reference_distribution_uses_all_prompts(self) -> None:
        target = BenchmarkTarget(
            target_id="author:test_author",
            track="author",
            author_id="author:test_author",
            conditioning_book_ids=["book:test_author:one", "book:test_author:two"],
            evaluation_book_id="book:test_author:three",
        ).validate()
        prompts = [
            PromptRecord(prompt_id="prompt:family:01", family="family", text="Prompt one").validate(),
            PromptRecord(prompt_id="prompt:family:02", family="family", text="Prompt two").validate(),
        ]
        seen_prompt_ids: list[str] = []

        def _fake_case_payload(target, prompt, **kwargs):
            seen_prompt_ids.append(prompt.prompt_id)
            return (
                SimpleNamespace(
                    evaluation_passage_ids=["passage:a", "passage:b", "passage:c", "passage:d"],
                    distractor_target_ids=["author:distractor"],
                ),
                {
                    target.target_id: {
                        "evaluation": [
                            {"passage_id": "passage:a", "text": "A."},
                            {"passage_id": "passage:b", "text": "B."},
                            {"passage_id": "passage:c", "text": "C."},
                            {"passage_id": "passage:d", "text": "D."},
                        ]
                    },
                    "author:distractor": {
                        "evaluation": [
                            {"passage_id": "passage:x", "text": "X."},
                            {"passage_id": "passage:y", "text": "Y."},
                            {"passage_id": "passage:z", "text": "Z."},
                            {"passage_id": "passage:w", "text": "W."},
                        ]
                    },
                },
            )

        with patch("eval.build_benchmark_manifests._case_payload", side_effect=_fake_case_payload), patch(
            "eval.build_benchmark_manifests.compute_style_metrics",
            return_value={
                "target_similarity_mean": 0.8,
                "distractor_similarity_means": {"author:distractor": 0.3},
                "style_margin_case": 0.5,
            },
        ):
            payload = _build_reference_distribution(
                track="author",
                benchmark_version="gutenberg_style_v1",
                targets=[target],
                prompts=prompts,
                books={},
                passages_by_book={},
                scorer=object(),
                builder_seed=42,
                distractor_policy={"target_count": 1, "min_count": 1, "book_track_same_author_policy": "exclude"},
            )

        self.assertEqual(seen_prompt_ids, [prompt.prompt_id for prompt in prompts])
        self.assertEqual(payload["prompt_count"], 2)
        self.assertGreater(payload["global"]["target_similarity"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
