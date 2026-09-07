from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rail.baselines.retrieval_icl import RetrievalICLRunner
from rail.baselines.traditional_ml import fit_baseline


class _DummyBackend:
    def complete(self, prompt: str, **kwargs) -> str:
        return "0"


class TraditionalMLBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.X = pd.DataFrame(
            {
                "num_a": [0.1, 0.2, 0.9, 1.0, 0.15, 0.85, 1.1, 0.05],
                "num_b": [1.0, 0.8, 0.2, 0.1, 0.9, 0.3, 0.0, 1.1],
                "cat": ["a", "a", "b", "b", "a", "b", "b", "a"],
            }
        )
        cls.y = pd.Series(["low", "low", "high", "high", "low", "high", "high", "low"])

    def test_logistic_regression_baseline_predicts_probabilities(self) -> None:
        fitted = fit_baseline(
            "logistic_regression",
            self.X,
            self.y,
            task_type="classification",
            random_state=42,
        )
        probabilities = fitted.predict_proba(self.X.head(2))
        self.assertEqual(probabilities.shape, (2, 2))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
        self.assertIsNotNone(fitted.label_encoder)

    def test_random_forest_baseline_predicts_probabilities(self) -> None:
        fitted = fit_baseline(
            "random_forest",
            self.X,
            self.y,
            task_type="classification",
            random_state=42,
            model_kwargs={"n_estimators": 20},
        )
        probabilities = fitted.predict_proba(self.X.head(3))
        self.assertEqual(probabilities.shape, (3, 2))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
        self.assertIsNotNone(fitted.label_encoder)

    def test_retrieval_runner_keeps_boundary_estimator_name(self) -> None:
        runner = RetrievalICLRunner(
            _DummyBackend(),
            selection_strategy="matched_pair_boundary",
            boundary_estimator_name="random_forest",
        )
        self.assertEqual(runner.boundary_estimator_name, "random_forest")


if __name__ == "__main__":
    unittest.main()
