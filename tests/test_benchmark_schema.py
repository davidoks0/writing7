import unittest

from eval.benchmark_schema import BenchmarkCase, PromptRecord


class BenchmarkSchemaTests(unittest.TestCase):
    def test_prompt_and_case_validation_accept_good_records(self) -> None:
        prompt = PromptRecord(
            prompt_id="prompt:interpersonal:01",
            family="interpersonal",
            text="Write a scene.",
            required_keywords=["scene"],
        ).validate()
        case = BenchmarkCase(
            case_id="case:author:test:jane_austen:interpersonal_01",
            benchmark_version="gutenberg_style_v1",
            track="author",
            split="test",
            target_id="author:jane_austen",
            prompt_id=prompt.prompt_id,
            conditioning_passage_ids=[
                "passage:book_jane_austen_emma:0:8",
                "passage:book_jane_austen_emma:10:18",
                "passage:book_jane_austen_emma:20:28",
            ],
            evaluation_passage_ids=[
                "passage:book_jane_austen_persuasion:10:18",
                "passage:book_jane_austen_persuasion:20:28",
                "passage:book_jane_austen_persuasion:30:38",
                "passage:book_jane_austen_persuasion:40:48",
            ],
            distractor_target_ids=["author:george_eliot"],
            distractor_passage_ids_by_target={
                "author:george_eliot": [
                    "passage:book_george_eliot_middlemarch:5:13",
                    "passage:book_george_eliot_middlemarch:15:23",
                    "passage:book_george_eliot_middlemarch:25:33",
                    "passage:book_george_eliot_middlemarch:35:43",
                ]
            },
            generation_profile_id="leaderboard_v1",
            sample_seeds=[11],
        ).validate()
        self.assertEqual(case.prompt_id, "prompt:interpersonal:01")

    def test_case_validation_rejects_bad_identifier(self) -> None:
        with self.assertRaises(ValueError):
            BenchmarkCase(
                case_id="Case:bad",
                benchmark_version="gutenberg_style_v1",
                track="author",
                split="test",
                target_id="author:jane_austen",
                prompt_id="prompt:interpersonal:01",
                conditioning_passage_ids=[
                    "passage:book_jane_austen_emma:0:8",
                    "passage:book_jane_austen_emma:10:18",
                    "passage:book_jane_austen_emma:20:28",
                ],
                evaluation_passage_ids=[
                    "passage:book_jane_austen_persuasion:10:18",
                    "passage:book_jane_austen_persuasion:20:28",
                    "passage:book_jane_austen_persuasion:30:38",
                    "passage:book_jane_austen_persuasion:40:48",
                ],
                distractor_target_ids=["author:george_eliot"],
                distractor_passage_ids_by_target={
                    "author:george_eliot": [
                        "passage:book_george_eliot_middlemarch:5:13",
                        "passage:book_george_eliot_middlemarch:15:23",
                        "passage:book_george_eliot_middlemarch:25:33",
                        "passage:book_george_eliot_middlemarch:35:43",
                    ]
                },
                generation_profile_id="leaderboard_v1",
                sample_seeds=[11],
            ).validate()


if __name__ == "__main__":
    unittest.main()
