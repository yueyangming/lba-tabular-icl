from __future__ import annotations

import unittest

import pandas as pd

from rail.experiments.run_modern_tabular_baselines import _cap_split


class ModernTabularBaselineRunnerTests(unittest.TestCase):
    def test_cap_split_keeps_aligned_labels(self) -> None:
        X = pd.DataFrame({"x": range(10)})
        y = pd.Series([f"label-{i}" for i in range(10)])

        X_cap, y_cap = _cap_split(X, y, cap=4, seed=7)

        self.assertEqual(len(X_cap), 4)
        self.assertEqual(len(y_cap), 4)
        for feature_value, label in zip(X_cap["x"], y_cap, strict=True):
            self.assertEqual(label, f"label-{feature_value}")

    def test_cap_split_disabled_when_cap_is_none_or_large(self) -> None:
        X = pd.DataFrame({"x": range(3)})
        y = pd.Series(["a", "b", "c"])

        X_none, y_none = _cap_split(X, y, cap=None, seed=1)
        X_large, y_large = _cap_split(X, y, cap=10, seed=1)

        self.assertEqual(X_none["x"].tolist(), [0, 1, 2])
        self.assertEqual(y_none.tolist(), ["a", "b", "c"])
        self.assertEqual(X_large["x"].tolist(), [0, 1, 2])
        self.assertEqual(y_large.tolist(), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
