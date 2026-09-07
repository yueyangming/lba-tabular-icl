from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # pragma: no cover
    root_mean_squared_error = None


TaskType = Literal["classification", "regression"]


def compute_classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Return the default metrics for classification tasks."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def compute_regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return the default metrics for regression tasks."""
    if root_mean_squared_error is not None:
        rmse = root_mean_squared_error(y_true, y_pred)
    else:
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_metrics(y_true, y_pred, task_type: TaskType) -> dict[str, float]:
    """Dispatch to the appropriate metric set."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if task_type == "classification":
        return compute_classification_metrics(y_true, y_pred)
    if task_type == "regression":
        return compute_regression_metrics(y_true, y_pred)
    raise ValueError(f"Unsupported task_type: {task_type}")
