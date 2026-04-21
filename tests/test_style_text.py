import unittest

from training.style_text import STYLE_MASKED_TEXT_VIEW, apply_text_view, style_focus_text


class StyleTextTests(unittest.TestCase):
    def test_style_focus_text_masks_named_entities_and_numbers(self) -> None:
        text = "Alice met Mr. Brown in London on 14 June. The harbor stayed quiet."
        masked = style_focus_text(text)
        self.assertIn("<ENT>", masked)
        self.assertIn("<NUM>", masked)
        self.assertIn("harbor", masked)
        self.assertIn("Mr.", masked)
        self.assertNotIn("London", masked)

    def test_apply_text_view_preserves_raw_mode(self) -> None:
        text = "A quiet room kept its own counsel."
        self.assertEqual(apply_text_view(text, "raw"), text)
        self.assertNotEqual(apply_text_view(text, STYLE_MASKED_TEXT_VIEW), "")


if __name__ == "__main__":
    unittest.main()
