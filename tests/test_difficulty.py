from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from rail.retrieval.difficulty import (
    compute_classification_committee_scores,
    compute_classification_difficulty_scores,
    compute_classification_oof_uncertainty_scores,
)


class DifficultyScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        X, y = make_classification(
            n_samples=120,
            n_features=8,
            n_informative=6,
            n_redundant=0,
            n_classes=3,
            random_state=42,
        )
        cls.X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        cls.y = pd.Series(y.astype(str), name="target")

    def test_difficulty_scores_match_training_size(self) -> None:
        scores = compute_classification_difficulty_scores(self.X, self.y, random_state=42)
        self.assertEqual(len(scores), len(self.X))
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(float(np.std(scores)), 0.0)

    def test_oof_uncertainty_scores_match_training_size(self) -> None:
        scores = compute_classification_oof_uncertainty_scores(self.X, self.y, random_state=42)
        self.assertEqual(len(scores), len(self.X))
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(float(np.std(scores)), 0.0)

    def test_committee_scores_match_training_size(self) -> None:
        scores = compute_classification_committee_scores(self.X, self.y, random_state=42)
        self.assertEqual(len(scores), len(self.X))
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(float(np.std(scores)), 0.0)


if __name__ == "__main__":
    unittest.main()
