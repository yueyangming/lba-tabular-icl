from __future__ import annotations

import unittest
from collections import Counter

import pandas as pd

from rail.experiments.run_label_intervention import _sample_matched_histogram_indices


class LabelShuffleProbeTests(unittest.TestCase):
    def test_sample_matched_histogram_indices_preserves_label_counts(self) -> None:
        y_train = pd.Series(["a", "a", "a", "b", "b", "c"])
        selected_targets = ["a", "a", "b", "c"]
        indices = _sample_matched_histogram_indices(y_train, selected_targets, seed=42)
        sampled_labels = [y_train.iloc[idx] for idx in indices]
        self.assertEqual(Counter(sampled_labels), Counter(selected_targets))
        self.assertEqual(len(indices), len(selected_targets))

    def test_sample_matched_histogram_indices_respects_exclusion_when_possible(self) -> None:
        y_train = pd.Series(["a", "a", "b", "b"])
        indices = _sample_matched_histogram_indices(
            y_train,
            ["a", "b"],
            seed=42,
            exclude_indices={0, 2},
        )
        self.assertEqual(set(indices), {1, 3})


if __name__ == "__main__":
    unittest.main()
