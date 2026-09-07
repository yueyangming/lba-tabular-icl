from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from rail.baselines.traditional_ml import fit_baseline


def _fit_train_uncertainty_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int,
) -> np.ndarray:
    fitted = fit_baseline(
        "xgboost",
        X_train,
        y_train,
        task_type="classification",
        random_state=random_state,
    )
    probabilities = fitted.predict_proba(X_train)
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        raise ValueError("Unexpected probability shape from classification baseline.")
    max_confidence = probabilities.max(axis=1)
    return 1.0 - max_confidence


def _effective_stratified_splits(y_train: pd.Series, *, n_splits: int) -> int:
    class_counts = y_train.value_counts(dropna=False)
    if class_counts.empty:
        raise ValueError("Cannot compute classification scores with an empty target series.")
    return min(int(n_splits), int(class_counts.min()))


def compute_classification_difficulty_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = 42,
    n_splits: int = 5,
) -> np.ndarray:
    """Estimate how ambiguous each training point looks to a strong tabular classifier.

    This now uses out-of-fold uncertainty whenever stratified folds are possible,
    and falls back to an in-sample estimate only for very small datasets.
    """
    return compute_classification_oof_uncertainty_scores(
        X_train,
        y_train,
        random_state=random_state,
        n_splits=n_splits,
    )


def compute_classification_oof_uncertainty_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = 42,
    n_splits: int = 5,
) -> np.ndarray:
    """Estimate uncertainty from out-of-fold predictions."""
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    effective_splits = _effective_stratified_splits(y_train, n_splits=n_splits)
    if effective_splits < 2:
        return _fit_train_uncertainty_scores(
            X_train,
            y_train,
            random_state=random_state,
        )

    splitter = StratifiedKFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=random_state,
    )
    scores = np.zeros(len(X_train), dtype=float)
    for fold_idx, (fit_idx, holdout_idx) in enumerate(splitter.split(X_train, y_train)):
        fitted = fit_baseline(
            "xgboost",
            X_train.iloc[fit_idx],
            y_train.iloc[fit_idx],
            task_type="classification",
            random_state=random_state + fold_idx,
        )
        probabilities = fitted.predict_proba(X_train.iloc[holdout_idx])
        if probabilities.ndim != 2 or probabilities.shape[1] == 0:
            raise ValueError("Unexpected probability shape from classification baseline.")
        scores[holdout_idx] = 1.0 - probabilities.max(axis=1)
    return scores


def compute_classification_oof_margin_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = 42,
    n_splits: int = 5,
) -> np.ndarray:
    """Estimate boundary proximity from out-of-fold class margins.

    Higher score means the point lies closer to a decision boundary under a
    strong surrogate classifier.
    """
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    effective_splits = _effective_stratified_splits(y_train, n_splits=n_splits)
    if effective_splits < 2:
        fitted = fit_baseline(
            "xgboost",
            X_train,
            y_train,
            task_type="classification",
            random_state=random_state,
        )
        probabilities = fitted.predict_proba(X_train)
        top1 = probabilities.max(axis=1)
        if probabilities.shape[1] >= 2:
            top2 = np.partition(probabilities, -2, axis=1)[:, -2]
        else:
            top2 = np.zeros_like(top1)
        return 1.0 - (top1 - top2)

    splitter = StratifiedKFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=random_state,
    )
    scores = np.zeros(len(X_train), dtype=float)
    for fold_idx, (fit_idx, holdout_idx) in enumerate(splitter.split(X_train, y_train)):
        fitted = fit_baseline(
            "xgboost",
            X_train.iloc[fit_idx],
            y_train.iloc[fit_idx],
            task_type="classification",
            random_state=random_state + fold_idx,
        )
        probabilities = fitted.predict_proba(X_train.iloc[holdout_idx])
        top1 = probabilities.max(axis=1)
        if probabilities.shape[1] >= 2:
            top2 = np.partition(probabilities, -2, axis=1)[:, -2]
        else:
            top2 = np.zeros_like(top1)
        scores[holdout_idx] = 1.0 - (top1 - top2)
    return scores


def compute_classification_committee_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = 42,
    n_splits: int = 5,
) -> np.ndarray:
    """Estimate informativeness from out-of-fold model disagreement plus uncertainty."""
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    effective_splits = _effective_stratified_splits(y_train, n_splits=n_splits)
    if effective_splits < 2:
        xgb = fit_baseline(
            "xgboost",
            X_train,
            y_train,
            task_type="classification",
            random_state=random_state,
        )
        knn = fit_baseline(
            "knn",
            X_train,
            y_train,
            task_type="classification",
            random_state=random_state,
        )
        xgb_pred = xgb.predict(X_train)
        knn_pred = knn.predict(X_train)
        xgb_unc = 1.0 - xgb.predict_proba(X_train).max(axis=1)
        knn_unc = 1.0 - knn.predict_proba(X_train).max(axis=1)
        disagreement = (xgb_pred != knn_pred).astype(float)
        avg_uncertainty = 0.5 * (xgb_unc + knn_unc)
        return 0.7 * disagreement + 0.3 * avg_uncertainty

    splitter = StratifiedKFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=random_state,
    )
    scores = np.zeros(len(X_train), dtype=float)
    for fold_idx, (fit_idx, holdout_idx) in enumerate(splitter.split(X_train, y_train)):
        xgb = fit_baseline(
            "xgboost",
            X_train.iloc[fit_idx],
            y_train.iloc[fit_idx],
            task_type="classification",
            random_state=random_state + fold_idx,
        )
        knn = fit_baseline(
            "knn",
            X_train.iloc[fit_idx],
            y_train.iloc[fit_idx],
            task_type="classification",
            random_state=random_state + fold_idx,
        )
        xgb_pred = xgb.predict(X_train.iloc[holdout_idx])
        knn_pred = knn.predict(X_train.iloc[holdout_idx])
        xgb_unc = 1.0 - xgb.predict_proba(X_train.iloc[holdout_idx]).max(axis=1)
        knn_unc = 1.0 - knn.predict_proba(X_train.iloc[holdout_idx]).max(axis=1)
        disagreement = (xgb_pred != knn_pred).astype(float)
        avg_uncertainty = 0.5 * (xgb_unc + knn_unc)
        scores[holdout_idx] = 0.7 * disagreement + 0.3 * avg_uncertainty
    return scores
