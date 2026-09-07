from __future__ import annotations

import argparse
from datetime import UTC, datetime

from rail.baselines import run_retrieval_icl, run_vanilla_icl
from rail.config import SiliconFlowConfig
from rail.data import load_openml_dataset, train_val_test_split
from rail.evaluation import write_results_csv
from rail.llm import build_backend


def resolve_model(alias: str, config: SiliconFlowConfig) -> str:
    if alias == "default":
        return config.default_model
    if alias == "latest":
        return config.latest_model
    if alias == "debug":
        return config.debug_model
    return alias


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal Vanilla ICL vs retrieval ICL smoke test.")
    parser.add_argument("--dataset", required=True, help="OpenML dataset name, for example diabetes.")
    parser.add_argument("--dataset-id", type=int, default=None, help="Optional OpenML dataset id.")
    parser.add_argument("--task-type-override", choices=["classification", "regression"], default=None)
    parser.add_argument("--model", default="debug", help="Model alias: debug/default/latest or a full model name.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--output", default="outputs/icl_smoke_summary.csv")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--request-log", default="outputs/logs/icl_requests.jsonl")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--progress-log", default=None, help="Optional JSONL file for incremental metric updates.")
    parser.add_argument("--progress-every", type=int, default=25, help="Write one progress record every N examples.")
    parser.add_argument("--selection-strategy", choices=["topk", "balanced", "diverse", "class_balanced_diversity", "contrastive", "difficulty", "oof_difficulty", "committee", "boundary", "uncertainty_only_boundary", "adaptive_boundary", "counter_boundary", "paired_boundary", "matched_pair_boundary", "knn_pair_boundary", "bidirectional_pair_boundary", "skeptical_pair_boundary", "margin_pair_rerank", "density_margin_pair_rerank", "relevance_constrained_density_rerank", "imbalance_pair_rerank", "query_difficulty_gated_rerank"], default="balanced")
    parser.add_argument("--boundary-estimator", choices=["knn", "xgboost", "lightgbm", "logistic_regression", "random_forest"], default="xgboost")
    args = parser.parse_args()

    config = SiliconFlowConfig.from_env()
    model_name = resolve_model(args.model, config)
    backend = build_backend(
        config,
        timeout_seconds=args.timeout_seconds,
        request_log_path=args.request_log,
    )

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

    runs = [
        run_vanilla_icl(
            backend,
            split.X_train,
            split.y_train,
            split.X_test,
            split.y_test,
            target_name=dataset.target_name,
            task_type=dataset.task_type,
            k=args.k,
            seed=args.seed,
            model=model_name,
            max_examples=args.max_examples,
            thinking_budget=args.thinking_budget,
            sleep_seconds=args.sleep_seconds,
            progress_log_path=args.progress_log,
            progress_every=args.progress_every,
            progress_context={
                "dataset": dataset.name,
                "task_type": dataset.task_type,
                "llm_model": model_name,
                "selection_strategy": "none",
                "boundary_estimator_name": "none",
                "k": args.k,
            },
        ),
        run_retrieval_icl(
            backend,
            split.X_train,
            split.y_train,
            split.X_test,
            split.y_test,
            target_name=dataset.target_name,
            task_type=dataset.task_type,
            k=args.k,
            seed=args.seed,
            model=model_name,
            max_examples=args.max_examples,
            thinking_budget=args.thinking_budget,
            selection_strategy=args.selection_strategy,
            boundary_estimator_name=args.boundary_estimator,
            sleep_seconds=args.sleep_seconds,
            progress_log_path=args.progress_log,
            progress_every=args.progress_every,
            progress_context={
                "dataset": dataset.name,
                "task_type": dataset.task_type,
                "llm_model": model_name,
                "selection_strategy": args.selection_strategy,
                "boundary_estimator_name": args.boundary_estimator,
                "k": args.k,
            },
        ),
    ]

    rows: list[dict[str, object]] = []
    for result in runs:
        row = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "dataset": dataset.name,
            "task_type": dataset.task_type,
            "llm_model": model_name,
            "eval_examples": args.max_examples,
            "k": args.k,
            "thinking_budget": args.thinking_budget,
            "selection_strategy": (result.get("selection_strategy") or ("none" if result["model_name"] == "vanilla_icl" else args.selection_strategy)),
            "boundary_estimator_name": result.get("boundary_estimator_name") or ("none" if result["model_name"] == "vanilla_icl" else args.boundary_estimator),
            "method": result["model_name"],
        }
        row.update(result["metrics"])
        rows.append(row)
        print(row)

    write_results_csv(args.output, rows)
    print(f"Saved ICL smoke results to {args.output}")


if __name__ == "__main__":
    main()
