"""Build query-level local-margin and flip artifacts.

This analysis reuses locally generated error-analysis flip CSVs and
computes kNN neighborhood statistics from the original train/test split.
"""

from __future__ import annotations

import argparse
import ast
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from rail.analysis.compute_leq import _build_preprocessor
from rail.data import load_openml_dataset, train_val_test_split


DATASETS = {
    "letter": {
        "dataset_id": 6,
        "split_seed": 42,
    },
    "credit-g": {
        "dataset_id": 31,
        "split_seed": 7,
    },
    "diabetes": {
        "dataset_id": 37,
        "split_seed": 42,
    },
    "page-blocks": {
        "dataset_id": 30,
        "split_seed": 42,
    },
}


def _parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _entropy(values: list[Any]) -> float:
    if not values:
        return 0.0
    counts = np.asarray(list(Counter(map(str, values)).values()), dtype=float)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _bucket(local_margin: float) -> str:
    if local_margin < 0:
        return "negative_evidence"
    if local_margin < 0.2:
        return "ambiguous_evidence"
    return "strong_local_evidence"


def _knn_stats_for_dataset(dataset_id: int, *, k: int, seed: int) -> tuple[pd.Series, list[dict[str, float]]]:
    dataset = load_openml_dataset(dataset_id=dataset_id, task_type_override="classification")
    split = train_val_test_split(
        dataset.features,
        dataset.target,
        task_type="classification",
        test_size=0.2,
        val_size=0.2,
        random_state=seed,
    )
    preprocessor = _build_preprocessor(split.X_train)
    X_train = preprocessor.fit_transform(split.X_train)
    X_test = preprocessor.transform(split.X_test)
    n_neighbors = min(k, len(split.X_train))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X_train)
    _, indices = nn.kneighbors(X_test)
    y_train = np.asarray(split.y_train.astype(str))
    y_test = split.y_test.astype(str).reset_index(drop=True)

    stats_rows: list[dict[str, float]] = []
    for pos, idx in enumerate(indices):
        true_label = str(y_test.iloc[pos])
        labels = list(y_train[idx])
        counts = Counter(labels)
        true_count = counts.get(true_label, 0)
        best_other = max((count for label, count in counts.items() if label != true_label), default=0)
        stats_rows.append(
            {
                "knn_true_freq": true_count / len(labels),
                "knn_best_other_freq": best_other / len(labels),
                "local_margin": (true_count - best_other) / len(labels),
                "local_entropy": _entropy(labels),
            }
        )
    return y_test, stats_rows


def build_query_margin_tables(
    *,
    datasets: list[str],
    k: int,
    seed: int,
    detail_output: Path,
    summary_output: Path,
    flip_root: Path = PROJECT_ROOT / "outputs" / "error_analysis",
    model: str = "qwen3.5-397b-a17b",
) -> None:
    detail_rows: list[dict[str, Any]] = []

    for dataset_name in datasets:
        info = DATASETS[dataset_name]
        flip_path = flip_root / dataset_name / model.replace("/", "_") / "flips.csv"
        if not flip_path.exists():
            raise FileNotFoundError(
                f"Missing flip file for {dataset_name}: {flip_path}. "
                "Generate traces using rail.analysis.run_error_analysis; "
                "see docs/REPRODUCIBILITY.md (query-level margin)."
            )
        flips = pd.read_csv(flip_path)
        if "selection_strategy" in flips.columns:
            flips = flips[flips["selection_strategy"] == "matched_pair_boundary"].copy()
        split_seed = int(info.get("split_seed", seed))
        y_test, knn_stats = _knn_stats_for_dataset(int(info["dataset_id"]), k=k, seed=split_seed)

        for _, row in flips.iterrows():
            eval_position = int(row["eval_position"])
            if eval_position >= len(knn_stats):
                continue
            true_label = str(row["ground_truth"])
            support_targets = _parse_list(row.get("retrieval_selected_targets"))
            support_targets_str = [str(value) for value in support_targets]
            support_count = len(support_targets_str)
            retrieval_correct = int(bool(row["retrieval_correct"]))
            vanilla_correct = int(bool(row["vanilla_correct"]))
            margin_stats = knn_stats[eval_position]
            detail_rows.append(
                {
                    "dataset": dataset_name,
                    "query_id": eval_position,
                    "row_index": int(row["row_index"]),
                    "true_label": true_label,
                    "split_true_label": str(y_test.iloc[eval_position]),
                    "split_seed": split_seed,
                    "vanilla_pred": str(row["vanilla_prediction"]),
                    "retrieval_pred": str(row["retrieval_prediction"]),
                    "vanilla_correct": vanilla_correct,
                    "retrieval_correct": retrieval_correct,
                    "gain": retrieval_correct - vanilla_correct,
                    "flip_type": row["flip_type"],
                    "knn_true_freq": margin_stats["knn_true_freq"],
                    "knn_best_other_freq": margin_stats["knn_best_other_freq"],
                    "local_margin": margin_stats["local_margin"],
                    "local_margin_bucket": _bucket(float(margin_stats["local_margin"])),
                    "local_entropy": margin_stats["local_entropy"],
                    "support_true_purity": (
                        support_targets_str.count(true_label) / support_count if support_count else np.nan
                    ),
                    "support_entropy": _entropy(support_targets_str),
                    "support_size": support_count,
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    summary_rows: list[dict[str, Any]] = []
    for (dataset_name, bucket), group in detail_df.groupby(["dataset", "local_margin_bucket"], sort=True):
        wrong_to_correct = int((group["flip_type"] == "wrong_to_correct").sum())
        correct_to_wrong = int((group["flip_type"] == "correct_to_wrong").sum())
        summary_rows.append(
            {
                "dataset": dataset_name,
                "bucket": bucket,
                "query_count": int(len(group)),
                "mean_gain": float(group["gain"].mean()),
                "wrong_to_correct": wrong_to_correct,
                "correct_to_wrong": correct_to_wrong,
                "net_flips": wrong_to_correct - correct_to_wrong,
                "mean_local_margin": float(group["local_margin"].mean()),
                "mean_local_entropy": float(group["local_entropy"].mean()),
                "mean_support_true_purity": float(group["support_true_purity"].mean()),
                "mean_support_entropy": float(group["support_entropy"].mean()),
            }
        )

    detail_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    detail_df.sort_values(["dataset", "query_id"]).to_csv(detail_output, index=False)
    pd.DataFrame(summary_rows).sort_values(["dataset", "bucket"]).to_csv(summary_output, index=False)
    print(f"Saved detail to {detail_output}")
    print(f"Saved bucket summary to {summary_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build query-level local-margin flip tables.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--flip-root", default="outputs/error_analysis")
    parser.add_argument("--model", default="qwen3.5-397b-a17b")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detail-output", default="outputs/query_margin_queries.csv")
    parser.add_argument("--summary-output", default="outputs/query_margin_buckets.csv")
    args = parser.parse_args()

    unknown = sorted(set(args.datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")

    build_query_margin_tables(
        datasets=args.datasets,
        flip_root=PROJECT_ROOT / args.flip_root,
        model=args.model,
        k=args.k,
        seed=args.seed,
        detail_output=PROJECT_ROOT / args.detail_output,
        summary_output=PROJECT_ROOT / args.summary_output,
    )


if __name__ == "__main__":
    main()
