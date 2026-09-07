from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from rail.baselines.traditional_ml import run_baseline
from rail.data import load_openml_dataset, train_val_test_split
from rail.evaluation import write_results_csv
from rail.evaluation.metrics import TaskType, compute_metrics


MethodName = Literal["catboost", "tabpfn", "tabicl"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_id: int
    task_type: TaskType


DATASETS: dict[str, DatasetSpec] = {
    "adult": DatasetSpec("adult", 1590, "classification"),
    "credit-g": DatasetSpec("credit-g", 31, "classification"),
    "diabetes": DatasetSpec("diabetes", 37, "classification"),
    "phoneme": DatasetSpec("phoneme", 1489, "classification"),
    "vehicle": DatasetSpec("vehicle", 54, "classification"),
    "segment": DatasetSpec("segment", 36, "classification"),
    "page-blocks": DatasetSpec("page-blocks", 30, "classification"),
    "satimage": DatasetSpec("satimage", 182, "classification"),
    "letter": DatasetSpec("letter", 6, "classification"),
    "california_housing": DatasetSpec("california_housing", 43939, "regression"),
    "wine_quality": DatasetSpec("wine_quality", 287, "regression"),
    "concrete_compressive_strength": DatasetSpec("concrete_compressive_strength", 44959, "regression"),
    "energy_efficiency": DatasetSpec("energy_efficiency", 43918, "regression"),
    "kin8nm": DatasetSpec("kin8nm", 189, "regression"),
}


def _cap_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cap: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if cap is None or cap <= 0 or len(X) <= cap:
        return X.reset_index(drop=True), y.reset_index(drop=True)
    sampled = X.sample(n=cap, random_state=seed)
    return sampled.reset_index(drop=True), y.loc[sampled.index].reset_index(drop=True)


def _tabpfn_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    task_type: TaskType,
    version: str,
) -> np.ndarray:
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    if version == "default":
        estimator = TabPFNClassifier() if task_type == "classification" else TabPFNRegressor()
    else:
        from tabpfn.constants import ModelVersion

        normalized = version.upper().replace(".", "_").replace("-", "_")
        model_version = getattr(ModelVersion, normalized)
        if task_type == "classification":
            estimator = TabPFNClassifier.create_default_for_version(model_version)
        else:
            estimator = TabPFNRegressor.create_default_for_version(model_version)

    if task_type == "classification":
        encoder = LabelEncoder()
        y_fit = encoder.fit_transform(y_train)
        estimator.fit(X_train, y_fit)
        return encoder.inverse_transform(np.asarray(estimator.predict(X_test)).astype(int))

    estimator.fit(X_train, y_train)
    return np.asarray(estimator.predict(X_test))


def _tabicl_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    task_type: TaskType,
) -> np.ndarray:
    if task_type != "classification":
        raise ValueError("TabICL is run as a classification-only reference in this protocol.")
    from tabicl import TabICLClassifier

    encoder = LabelEncoder()
    y_fit = encoder.fit_transform(y_train)
    estimator = TabICLClassifier()
    estimator.fit(X_train, y_fit)
    return encoder.inverse_transform(np.asarray(estimator.predict(X_test)).astype(int))


def run_method(
    method: MethodName,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    task_type: TaskType,
    seed: int,
    tabpfn_version: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    if method == "catboost":
        result = run_baseline(
            "catboost",
            X_train,
            y_train,
            X_test,
            y_test,
            task_type=task_type,
            random_state=seed,
        )
        metrics = result["metrics"]
    elif method == "tabpfn":
        predictions = _tabpfn_predict(X_train, y_train, X_test, task_type=task_type, version=tabpfn_version)
        metrics = compute_metrics(y_test, predictions, task_type)
    elif method == "tabicl":
        predictions = _tabicl_predict(X_train, y_train, X_test, task_type=task_type)
        metrics = compute_metrics(y_test, predictions, task_type)
    else:
        raise ValueError(f"Unsupported method: {method}")

    elapsed = time.perf_counter() - start
    return {
        "method": method,
        "status": "ok",
        "elapsed_seconds": elapsed,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run modern tabular reference baselines on the canonical RAIL splits.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--methods", nargs="+", default=["catboost", "tabpfn"], choices=["catboost", "tabpfn", "tabicl"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-cap", type=int, default=None, help="Optional cap for training rows; <=0 disables.")
    parser.add_argument("--test-cap", type=int, default=None, help="Optional cap for test rows; <=0 disables.")
    parser.add_argument("--tabpfn-version", default="default", help="default, V2, V2_5, or another ModelVersion member.")
    parser.add_argument("--output", default="outputs/modern_tabular_baselines.csv")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        spec = DATASETS[dataset_name]
        dataset = load_openml_dataset(dataset_id=spec.dataset_id, task_type_override=spec.task_type)
        split = train_val_test_split(
            dataset.features,
            dataset.target,
            task_type=spec.task_type,
            random_state=args.seed,
        )
        X_train, y_train = _cap_split(split.X_train, split.y_train, cap=args.train_cap, seed=args.seed)
        X_test, y_test = _cap_split(split.X_test, split.y_test, cap=args.test_cap, seed=args.seed + 1)

        for method in args.methods:
            row: dict[str, Any] = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": dataset_name,
                "openml_dataset_id": spec.dataset_id,
                "task_type": spec.task_type,
                "method": method,
                "seed": args.seed,
                "train_examples": len(X_train),
                "test_examples": len(X_test),
                "train_cap": args.train_cap,
                "test_cap": args.test_cap,
                "tabpfn_version": args.tabpfn_version if method == "tabpfn" else "",
            }
            try:
                row.update(
                    run_method(
                        method,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        task_type=spec.task_type,
                        seed=args.seed,
                        tabpfn_version=args.tabpfn_version,
                    )
                )
            except Exception as exc:  # keep long remote batches from losing completed datasets
                row.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})
            rows.append(row)
            write_results_csv(args.output, rows)
            print(row, flush=True)

    print(f"Saved modern tabular baseline results to {args.output}")


if __name__ == "__main__":
    main()
