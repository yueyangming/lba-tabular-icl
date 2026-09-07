from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rail.baselines import run_retrieval_icl, run_vanilla_icl
from rail.config import SiliconFlowConfig
from rail.data import load_openml_dataset, train_val_test_split
from rail.evaluation.io import write_results_csv
from rail.experiments.run_icl_smoke import resolve_model
from rail.llm import build_backend


DATASETS = {
    "letter": {"dataset_id": 6, "task_type": "classification"},
    "diabetes": {"dataset_id": 37, "task_type": "classification"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classification k-sensitivity study.")
    parser.add_argument("--datasets", nargs="+", default=["letter", "diabetes"])
    parser.add_argument("--k-values", nargs="+", type=int, default=[16, 32, 64, 100])
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--model", default="latest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinking-budget", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--request-log", default="outputs/logs/classification_k_sensitivity_requests.jsonl")
    parser.add_argument("--output", default="outputs/classification_k_sensitivity.csv")
    parser.add_argument("--figure-output", default="outputs/classification_k_sensitivity.pdf")
    args = parser.parse_args()

    config = SiliconFlowConfig.from_env()
    model_name = resolve_model(args.model, config)
    backend = build_backend(
        config,
        timeout_seconds=args.timeout_seconds,
        request_log_path=args.request_log,
    )

    rows: list[dict[str, object]] = []

    for dataset_name in args.datasets:
        info = DATASETS[dataset_name]
        dataset = load_openml_dataset(
            dataset_id=info["dataset_id"],
            task_type_override=info["task_type"],
        )
        split = train_val_test_split(
            dataset.features,
            dataset.target,
            task_type=info["task_type"],
            test_size=0.2,
            val_size=0.2,
            random_state=args.seed,
        )

        for k in args.k_values:
            vanilla = run_vanilla_icl(
                backend,
                split.X_train,
                split.y_train,
                split.X_test,
                split.y_test,
                target_name=dataset.target_name,
                task_type="classification",
                k=k,
                seed=args.seed,
                model=model_name,
                max_examples=args.max_examples,
                thinking_budget=args.thinking_budget,
                sleep_seconds=args.sleep_seconds,
            )
            retrieval = run_retrieval_icl(
                backend,
                split.X_train,
                split.y_train,
                split.X_test,
                split.y_test,
                target_name=dataset.target_name,
                task_type="classification",
                k=k,
                seed=args.seed,
                model=model_name,
                max_examples=args.max_examples,
                thinking_budget=args.thinking_budget,
                selection_strategy="matched_pair_boundary",
                sleep_seconds=args.sleep_seconds,
            )
            for method_name, result in [("vanilla_icl", vanilla), ("retrieval_icl", retrieval)]:
                rows.append(
                    {
                        "dataset": dataset_name,
                        "k": k,
                        "method": method_name,
                        "query_count": args.max_examples,
                        "accuracy": float(result["metrics"]["accuracy"]),
                        "f1_macro": float(result["metrics"]["f1_macro"]),
                    }
                )

    out_path = Path(args.output)
    fig_path = Path(args.figure_output)
    write_results_csv(out_path, rows)

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharex=True)
    for ax, metric in zip(axes, ["accuracy", "f1_macro"], strict=True):
        for (dataset_name, method_name), sub in df.groupby(["dataset", "method"], sort=False):
            sub = sub.sort_values("k")
            label = f"{dataset_name} / {'retrieval' if method_name == 'retrieval_icl' else 'vanilla'}"
            ax.plot(sub["k"], sub[metric], marker="o", linewidth=1.8, label=label)
        ax.set_xlabel("k")
        ax.set_ylabel(metric)
        ax.set_title(f"Classification k-Sensitivity ({metric})")
        ax.grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path)

    print(f"Saved results to {out_path}")
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
