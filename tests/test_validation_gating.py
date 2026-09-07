from __future__ import annotations

import unittest

from rail.experiments.run_validation_gating import (
    _hybrid_predictions,
    _random_gate_predictions,
    _select_threshold,
)


class ICDMValidationGatingTests(unittest.TestCase):
    def test_hybrid_predictions_use_retrieval_above_threshold(self) -> None:
        predictions, use_retrieval = _hybrid_predictions(
            vanilla_pred=["a", "b", "c"],
            retrieval_pred=["x", "y", "z"],
            scores=[-0.1, 0.2, 0.8],
            threshold=0.2,
        )
        self.assertEqual(predictions, ["a", "y", "z"])
        self.assertEqual(use_retrieval, [False, True, True])

    def test_select_threshold_prefers_accuracy_then_lower_retrieval_rate(self) -> None:
        selection = _select_threshold(
            y_true=["a", "b", "c"],
            vanilla_pred=["a", "x", "c"],
            retrieval_pred=["z", "b", "c"],
            scores=[0.9, 0.8, 0.1],
            thresholds=[0.0, 0.5, 1.0],
        )
        self.assertEqual(selection["selected_threshold"], 1.0)
        self.assertEqual(selection["selected_row"]["accuracy"], 2 / 3)
        self.assertEqual(selection["selected_row"]["retrieval_rate"], 0.0)

    def test_random_gate_matches_requested_count_after_rounding(self) -> None:
        _, use_retrieval = _random_gate_predictions(
            ["a", "b", "c", "d"],
            ["w", "x", "y", "z"],
            retrieval_rate=0.5,
            seed=42,
        )
        self.assertEqual(sum(use_retrieval), 2)


if __name__ == "__main__":
    unittest.main()
