from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from rail.evaluation.metrics import TaskType, compute_metrics


BaselineName = Literal["knn", "xgboost", "lightgbm", "logistic_regression", "random_forest", "catboost"]


@dataclass
class FittedBaseline:
    name: BaselineName
    task_type: TaskType
    pipeline: Pipeline
    label_encoder: LabelEncoder | None = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = self.pipeline.predict(X)
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions.astype(int))
        return np.asarray(predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification baselines.")
        model = self.pipeline.named_steps["model"]
        if not hasattr(model, "predict_proba"):
            raise ValueError(f"Model {self.name} does not support predict_proba.")
        return np.asarray(self.pipeline.predict_proba(X))


def _build_preprocessor(X: pd.DataFrame, *, scale_numeric: bool) -> ColumnTransformer:
    numeric_columns = list(X.select_dtypes(include=["number", "bool"]).columns)
    categorical_columns = [col for col in X.columns if col not in numeric_columns]

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def _xgboost_estimator(task_type: TaskType, random_state: int, model_kwargs: dict[str, Any]) -> Any:
    base_kwargs = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": random_state,
        "n_jobs": -1,
    }
    base_kwargs.update(model_kwargs)
    if task_type == "classification":
        return XGBClassifier(
            eval_metric="mlogloss",
            **base_kwargs,
        )
    return XGBRegressor(
        eval_metric="rmse",
        **base_kwargs,
    )


def _lightgbm_estimator(task_type: TaskType, random_state: int, model_kwargs: dict[str, Any]) -> Any:
    base_kwargs = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
    }
    base_kwargs.update(model_kwargs)
    if task_type == "classification":
        return LGBMClassifier(**base_kwargs)
    return LGBMRegressor(**base_kwargs)


def _knn_estimator(task_type: TaskType, model_kwargs: dict[str, Any]) -> Any:
    base_kwargs = {"n_neighbors": 5}
    base_kwargs.update(model_kwargs)
    if task_type == "classification":
        return KNeighborsClassifier(**base_kwargs)
    return KNeighborsRegressor(**base_kwargs)


def _logistic_regression_estimator(task_type: TaskType, random_state: int, model_kwargs: dict[str, Any]) -> Any:
    if task_type == "classification":
        base_kwargs = {
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": random_state,
        }
        base_kwargs.update(model_kwargs)
        return LogisticRegression(**base_kwargs)
    base_kwargs = {}
    base_kwargs.update(model_kwargs)
    return Ridge(**base_kwargs)


def _random_forest_estimator(task_type: TaskType, random_state: int, model_kwargs: dict[str, Any]) -> Any:
    base_kwargs = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 1,
        "random_state": random_state,
        "n_jobs": -1,
    }
    base_kwargs.update(model_kwargs)
    if task_type == "classification":
        return RandomForestClassifier(**base_kwargs)
    return RandomForestRegressor(**base_kwargs)


def _catboost_estimator(task_type: TaskType, random_state: int, model_kwargs: dict[str, Any]) -> Any:
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("catboost is required for the catboost baseline.") from exc

    base_kwargs = {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "MultiClass" if task_type == "classification" else "RMSE",
        "random_seed": random_state,
        "thread_count": -1,
        "verbose": False,
        "allow_writing_files": False,
    }
    base_kwargs.update(model_kwargs)
    if task_type == "classification":
        return CatBoostClassifier(**base_kwargs)
    return CatBoostRegressor(**base_kwargs)


def fit_baseline(
    name: BaselineName,
    X_train: pd.DataFrame,
    y_train,
    *,
    task_type: TaskType,
    random_state: int = 42,
    model_kwargs: dict[str, Any] | None = None,
) -> FittedBaseline:
    """Fit a minimal tabular baseline with mixed-type feature support."""
    model_kwargs = model_kwargs or {}
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)

    if name == "knn":
        estimator = _knn_estimator(task_type, model_kwargs)
        preprocessor = _build_preprocessor(X_train, scale_numeric=True)
        label_encoder = None
        y_fit = y_train
    elif name == "xgboost":
        estimator = _xgboost_estimator(task_type, random_state, model_kwargs)
        preprocessor = _build_preprocessor(X_train, scale_numeric=False)
        label_encoder = LabelEncoder() if task_type == "classification" else None
        y_fit = label_encoder.fit_transform(y_train) if label_encoder is not None else y_train
    elif name == "lightgbm":
        estimator = _lightgbm_estimator(task_type, random_state, model_kwargs)
        preprocessor = _build_preprocessor(X_train, scale_numeric=False)
        label_encoder = LabelEncoder() if task_type == "classification" else None
        y_fit = label_encoder.fit_transform(y_train) if label_encoder is not None else y_train
    elif name == "logistic_regression":
        estimator = _logistic_regression_estimator(task_type, random_state, model_kwargs)
        preprocessor = _build_preprocessor(X_train, scale_numeric=True)
        label_encoder = LabelEncoder() if task_type == "classification" else None
        y_fit = label_encoder.fit_transform(y_train) if label_encoder is not None else y_train
    elif name == "random_forest":
        estimator = _random_forest_estimator(task_type, random_state, model_kwargs)
        preprocessor = _build_preprocessor(X_train, scale_numeric=False)
        label_encoder = LabelEncoder() if task_type == "classification" else None
        y_fit = label_encoder.fit_transform(y_train) if label_encoder is not None else y_train
    elif name == "catboost":
        estimator = _catboost_estimator(task_type, random_state, model_kwargs)
        preprocessor = _build_preprocessor(X_train, scale_numeric=False)
        label_encoder = LabelEncoder() if task_type == "classification" else None
        y_fit = label_encoder.fit_transform(y_train) if label_encoder is not None else y_train
    else:
        raise ValueError(f"Unsupported baseline: {name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clone(estimator)),
        ]
    )
    pipeline.fit(X_train, y_fit)
    return FittedBaseline(name=name, task_type=task_type, pipeline=pipeline, label_encoder=label_encoder)


def run_baseline(
    name: BaselineName,
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    y_test=None,
    *,
    task_type: TaskType,
    random_state: int = 42,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit a baseline, run predictions, and optionally compute metrics."""
    fitted = fit_baseline(
        name,
        X_train,
        y_train,
        task_type=task_type,
        random_state=random_state,
        model_kwargs=model_kwargs,
    )
    predictions = fitted.predict(X_test)
    result: dict[str, Any] = {
        "model_name": name,
        "task_type": task_type,
        "y_pred": predictions.tolist(),
    }
    if y_test is not None:
        result["metrics"] = compute_metrics(y_test, predictions, task_type)
    return result
