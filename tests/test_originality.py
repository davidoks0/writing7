import unittest

from eval.originality import compute_originality_metrics


class OriginalityTests(unittest.TestCase):
    def test_exact_copy_is_flagged(self) -> None:
        reference = "The careful house received the morning as if it expected an apology."
        metrics = compute_originality_metrics(reference, [reference])
        self.assertTrue(metrics["copy_flag"])

    def test_novel_text_is_not_flagged(self) -> None:
        reference = "A harbor ledger may lie more elegantly than a frightened man."
        hypothesis = "Fresh snow covered the lane while the child counted birds above the gate."
        metrics = compute_originality_metrics(hypothesis, [reference])
        self.assertFalse(metrics["copy_flag"])

    def test_reference_groups_surface_hidden_target_matches(self) -> None:
        conditioning = "A quiet room remembered nothing but its curtains."
        hidden_target = "The cautious clerk folded the letter twice and still distrusted its silence."
        metrics = compute_originality_metrics(
            hidden_target,
            [conditioning],
            reference_groups={"target_evaluation": [hidden_target]},
        )
        self.assertTrue(metrics["copy_flag"])
        self.assertEqual(metrics["max_reference_group"], "target_evaluation")
        self.assertFalse(metrics["reference_group_metrics"]["conditioning"]["copy_flag"])
        self.assertTrue(metrics["reference_group_metrics"]["target_evaluation"]["copy_flag"])

    def test_entity_transplant_is_flagged_separately_from_copy(self) -> None:
        reference = "Miss Vane crossed Blackwater Square before the market bell and hid the parcel beneath her shawl."
        hypothesis = (
            "Miss Vane spent the afternoon near Blackwater Square describing a debt, a childhood illness, "
            "and a broken engagement while the porter argued about weather, shipping tariffs, and a ruined account book."
        )
        metrics = compute_originality_metrics(hypothesis, [reference])
        self.assertFalse(metrics["copy_flag"])
        self.assertTrue(metrics["entity_transplant_flag"])
        self.assertFalse(metrics["originality_pass"])

    def test_full_target_book_group_can_become_max_reference_group(self) -> None:
        conditioning = "A quiet room remembered nothing but its curtains."
        full_target_book = "The gardener counted the frost-bitten pears twice before he admitted the orchard was finished."
        metrics = compute_originality_metrics(
            full_target_book,
            [conditioning],
            reference_groups={"full_target_book": [full_target_book]},
        )
        self.assertEqual(metrics["max_reference_group"], "full_target_book")
        self.assertTrue(metrics["reference_group_metrics"]["full_target_book"]["copy_flag"])


if __name__ == "__main__":
    unittest.main()
