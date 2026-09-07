from __future__ import annotations

import unittest

from rail.baselines.vanilla_icl import parse_classification_prediction


class ClassificationParserTests(unittest.TestCase):
    def test_prefers_longer_label_when_labels_overlap(self) -> None:
        label_space = ["1", "10", "11"]
        prediction = parse_classification_prediction("10", label_space)
        self.assertEqual(prediction, "10")

    def test_handles_token_style_response(self) -> None:
        label_space = ["good", "bad"]
        prediction = parse_classification_prediction("The answer is bad.", label_space)
        self.assertEqual(prediction, "bad")

    def test_avoids_short_substring_match_inside_word(self) -> None:
        label_space = ["X", "S"]
        prediction = parse_classification_prediction("example", label_space)
        self.assertEqual(prediction, "example")

    def test_first_line_exact_match_is_accepted(self) -> None:
        label_space = ["10", "11"]
        prediction = parse_classification_prediction("10\nconfidence: high", label_space)
        self.assertEqual(prediction, "10")


if __name__ == "__main__":
    unittest.main()
