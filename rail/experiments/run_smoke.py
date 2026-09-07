from __future__ import annotations

import argparse
from datetime import datetime, UTC

from rail.baselines import run_baseline
from rail.data import load_openml_dataset, train_val_test_split
from rail.evaluation import write_results_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal baseline smoke test on one OpenML dataset.")
    parser.add_argument("--dataset", required=True, help="OpenML dataset name, for example diabetes.")
    parser.add_argument("--dataset-id", type=int, default=None, help="Optional OpenML dataset id.")
    parser.add_argument("--task-type-override", choices=["classification", "regression"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--output", default="outputs/smoke_summary.csv")
    args = parser.parse_args()

    dataset = load_openml_dataset(
        dataset_id=args.dataset_id,
        name=args.dataset,
        task_type_override=args.task_type_override,
    )
    split = train_val_test_split(
        dataset.features,
        dataset.target,
        task_type=dataset.task_type,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
    )

    rows: list[dict[str, object]] = []
    for name in ["knn", "xgboost", "lightgbm"]:
        result = run_baseline(
            name,
            split.X_train,
            split.y_train,
            split.X_test,
            split.y_test,
            task_type=dataset.task_type,
            random_state=args.seed,
        )
        row = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "dataset": dataset.name,
            "task_type": dataset.task_type,
            "model_name": result["model_name"],
        }
        row.update(result["metrics"])
        rows.append(row)
        print(row)

    write_results_csv(args.output, rows)
    print(f"Saved smoke results to {args.output}")


if __name__ == "__main__":
    main()
