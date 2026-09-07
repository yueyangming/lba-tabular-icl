from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rail.analysis.run_selector_ablation_proxy import _support_quality_metrics
from rail.retrieval.selectors import select_indices


class SelectorTests(unittest.TestCase):
    def test_class_balanced_diversity_covers_available_classes(self) -> None:
        y_train = pd.Series(["a", "a", "b", "b", "c", "c"])
        candidate_indices = list(range(len(y_train)))
        candidate_matrix = np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [-1.0, 0.0],
                [-0.9, 0.1],
            ]
        )
        selected = select_indices(
            candidate_indices,
            y_train,
            k=3,
            strategy="class_balanced_diversity",
            label_space=["a", "b", "c"],
            candidate_matrix=candidate_matrix,
            query_vector=np.asarray([[1.0, 0.0]]),
        )
        self.assertEqual({y_train.iloc[idx] for idx in selected}, {"a", "b", "c"})

    def test_uncertainty_only_boundary_prefers_uncertain_priority_labels(self) -> None:
        y_train = pd.Series(["a", "b", "c", "a"])
        selected = select_indices(
            [0, 1, 2, 3],
            y_train,
            k=2,
            strategy="uncertainty_only_boundary",
            candidate_matrix=np.eye(4),
            query_vector=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
            uncertainty_scores=np.asarray([0.1, 0.9, 0.8, 0.7]),
            priority_labels=["a", "b"],
        )
        self.assertEqual(selected, [1, 3])

    def test_support_quality_margin_uses_largest_competing_label(self) -> None:
        metrics = _support_quality_metrics(
            true_label="a",
            selected_labels=["a", "a", "b", "c"],
            selected_distances=[1.0, 2.0, 3.0, 4.0],
            selected_uncertainties=[0.1, 0.2, 0.3, 0.4],
        )
        self.assertEqual(metrics["true_label_purity"], 0.5)
        self.assertEqual(metrics["local_margin"], 0.25)
        self.assertEqual(metrics["class_coverage"], 3.0)
        self.assertEqual(metrics["mean_standardized_distance"], 2.5)


if __name__ == "__main__":
    unittest.main()
