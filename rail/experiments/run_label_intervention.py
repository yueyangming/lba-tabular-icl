from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score, f1_score

from rail.baselines.retrieval_icl import RetrievalICLRunner
from rail.baselines.vanilla_icl import ICLExample, VanillaICLRunner, build_vanilla_prompt
from rail.config import SiliconFlowConfig
from rail.data import load_openml_dataset, train_val_test_split
from rail.evaluation.io import write_results_csv
from rail.experiments.run_icl_smoke import resolve_model
from rail.llm import build_backend


DATASETS = {
    "letter": {"dataset_id": 6, "task_type": "classification", "k": 100},
    "credit-g": {"dataset_id": 31, "task_type": "classification", "k": 100},
    "diabetes": {"dataset_id": 37, "task_type": "classification", "k": 100},
    "phoneme": {"dataset_id": 1489, "task_type": "classification", "k": 100},
}


def _parse_pred(text: str, label_space: list[Any]) -> Any:
    return VanillaICLRunner.parse_prediction(
        text,
        task_type="classification",
        label_space=label_space,
    )


def _macro_f1(y_true: list[Any], y_pred: list[Any]) -> float:
    return float(f1_score(list(map(str, y_true)), list(map(str, y_pred)), average="macro", zero_division=0))


def _accuracy(y_true: list[Any], y_pred: list[Any]) -> float:
    return float(accuracy_score(list(map(str, y_true)), list(map(str, y_pred))))


def _summary_row(dataset: str, condition: str, y_true: list[Any], y_pred: list[Any]) -> dict[str, object]:
    return {
        "dataset": dataset,
        "condition": condition,
        "query_count": len(y_true),
        "accuracy": _accuracy(y_true, y_pred),
        "f1_macro": _macro_f1(y_true, y_pred),
    }


def _sample_matched_histogram_indices(
    y_train: pd.Series,
    selected_targets: list[Any],
    *,
    seed: int,
    exclude_indices: set[int] | None = None,
) -> list[int]:
    """Sample random training rows with the same label histogram as selected_targets."""
    rng = random.Random(seed)
    exclude_indices = exclude_indices or set()
    selected_indices: list[int] = []
    y_values = list(y_train)
    by_label: dict[str, list[int]] = {}
    for idx, label in enumerate(y_values):
        if idx in exclude_indices:
            continue
        by_label.setdefault(str(label), []).append(idx)

    for label, count in pd.Series(list(map(str, selected_targets))).value_counts().to_dict().items():
        candidates = by_label.get(str(label), [])
        if not candidates:
            continue
        if len(candidates) >= count:
            selected_indices.extend(rng.sample(candidates, k=count))
        else:
            selected_indices.extend(candidates)
            selected_indices.extend(rng.choices(candidates, k=count - len(candidates)))
    rng.shuffle(selected_indices)
    return selected_indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Run label-shuffle causal probe for retrieval support sets.")
    parser.add_argument("--datasets", nargs="+", default=["letter", "credit-g"])
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--model", default="latest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinking-budget", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--request-log", default="outputs/logs/label_intervention_requests.jsonl")
    parser.add_argument("--summary-output", default="outputs/label_intervention_summary.csv")
    parser.add_argument("--detail-output", default="outputs/label_intervention_detail.csv")
    args = parser.parse_args()

    config = SiliconFlowConfig.from_env()
    model_name = resolve_model(args.model, config)
    backend = build_backend(
        config,
        timeout_seconds=args.timeout_seconds,
        request_log_path=args.request_log,
    )

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

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
        X_eval = split.X_test.head(args.max_examples)
        y_eval = split.y_test.head(args.max_examples)
        label_space = sorted(split.y_train.dropna().unique().tolist(), key=lambda value: str(value))

        vanilla_runner = VanillaICLRunner(backend, seed=args.seed)
        retrieval_runner = RetrievalICLRunner(
            backend,
            seed=args.seed,
            selection_strategy="matched_pair_boundary",
        )
        retrieval_runner.retriever.fit(split.X_train)
        retrieval_runner._difficulty_scores = None

        vanilla_preds: list[Any] = []
        retrieval_preds: list[Any] = []
        shuffle_preds: list[Any] = []
        matched_hist_preds: list[Any] = []

        for eval_position, (row_index, row) in enumerate(X_eval.iterrows()):
            # vanilla
            vanilla_trace = vanilla_runner.predict_one_with_trace(
                X_train=split.X_train,
                y_train=split.y_train,
                query=row,
                target_name=dataset.target_name,
                task_type="classification",
                label_space=label_space,
                k=info["k"],
                model=model_name,
                thinking_budget=args.thinking_budget,
            )
            vanilla_pred = _parse_pred(str(vanilla_trace["response"]), label_space)
            vanilla_preds.append(vanilla_pred)

            # retrieval control
            retrieval_trace = retrieval_runner.predict_one_with_trace(
                X_train=split.X_train,
                y_train=split.y_train,
                query=row,
                target_name=dataset.target_name,
                task_type="classification",
                label_space=label_space,
                k=info["k"],
                model=model_name,
                thinking_budget=args.thinking_budget,
            )
            retrieval_pred = _parse_pred(str(retrieval_trace["response"]), label_space)
            retrieval_preds.append(retrieval_pred)

            # shuffled-label retrieval
            selected_indices = retrieval_trace["indices"]
            selected_targets = [split.y_train.iloc[idx] for idx in selected_indices]
            shuffled_targets = selected_targets.copy()
            random.Random(args.seed + eval_position).shuffle(shuffled_targets)
            examples = [
                ICLExample(features=split.X_train.iloc[idx], target=target)
                for idx, target in zip(selected_indices, shuffled_targets, strict=True)
            ]
            prompt = build_vanilla_prompt(
                examples=examples,
                query=row,
                target_name=dataset.target_name,
                task_type="classification",
                label_space=label_space,
            )
            shuffle_response = backend.complete(
                prompt,
                model=model_name,
                thinking_budget=args.thinking_budget,
            )
            shuffle_pred = _parse_pred(str(shuffle_response), label_space)
            shuffle_preds.append(shuffle_pred)

            # matched-histogram random support: preserve the retrieved label
            # histogram but break feature locality.
            matched_hist_indices = _sample_matched_histogram_indices(
                split.y_train,
                selected_targets,
                seed=args.seed * 1009 + eval_position,
                exclude_indices=set(selected_indices),
            )
            matched_hist_examples = [
                ICLExample(features=split.X_train.iloc[idx], target=split.y_train.iloc[idx])
                for idx in matched_hist_indices
            ]
            matched_hist_prompt = build_vanilla_prompt(
                examples=matched_hist_examples,
                query=row,
                target_name=dataset.target_name,
                task_type="classification",
                label_space=label_space,
            )
            matched_hist_response = backend.complete(
                matched_hist_prompt,
                model=model_name,
                thinking_budget=args.thinking_budget,
            )
            matched_hist_pred = _parse_pred(str(matched_hist_response), label_space)
            matched_hist_preds.append(matched_hist_pred)

            detail_rows.append(
                {
                    "dataset": dataset_name,
                    "eval_position": eval_position,
                    "row_index": int(row_index),
                    "ground_truth": y_eval.loc[row_index],
                    "vanilla_prediction": vanilla_pred,
                    "retrieval_prediction": retrieval_pred,
                    "shuffle_prediction": shuffle_pred,
                    "matched_hist_prediction": matched_hist_pred,
                    "vanilla_correct": str(vanilla_pred) == str(y_eval.loc[row_index]),
                    "retrieval_correct": str(retrieval_pred) == str(y_eval.loc[row_index]),
                    "shuffle_correct": str(shuffle_pred) == str(y_eval.loc[row_index]),
                    "matched_hist_correct": str(matched_hist_pred) == str(y_eval.loc[row_index]),
                }
            )

        y_true = y_eval.tolist()
        summary_rows.extend(
            [
                _summary_row(dataset_name, "vanilla", y_true, vanilla_preds),
                _summary_row(dataset_name, "retrieval_control", y_true, retrieval_preds),
                _summary_row(dataset_name, "retrieval_label_shuffle", y_true, shuffle_preds),
                _summary_row(dataset_name, "matched_histogram_random_support", y_true, matched_hist_preds),
            ]
        )

        retrieval_correct = [int(str(p) == str(y)) for p, y in zip(retrieval_preds, y_true, strict=True)]
        shuffle_correct = [int(str(p) == str(y)) for p, y in zip(shuffle_preds, y_true, strict=True)]
        stat, p = ttest_rel(retrieval_correct, shuffle_correct)
        summary_rows.append(
            {
                "dataset": dataset_name,
                "condition": "retrieval_vs_shuffle_paired_ttest",
                "query_count": len(y_true),
                "accuracy": None,
                "f1_macro": None,
                "paired_t_stat": float(stat),
                "paired_t_p_value": float(p),
                "retrieval_minus_shuffle_accuracy": _accuracy(y_true, retrieval_preds) - _accuracy(y_true, shuffle_preds),
                "retrieval_minus_shuffle_f1_macro": _macro_f1(y_true, retrieval_preds) - _macro_f1(y_true, shuffle_preds),
            }
        )
        matched_hist_correct = [int(str(p) == str(y)) for p, y in zip(matched_hist_preds, y_true, strict=True)]
        stat_hist, p_hist = ttest_rel(retrieval_correct, matched_hist_correct)
        summary_rows.append(
            {
                "dataset": dataset_name,
                "condition": "retrieval_vs_matched_histogram_paired_ttest",
                "query_count": len(y_true),
                "accuracy": None,
                "f1_macro": None,
                "paired_t_stat": float(stat_hist),
                "paired_t_p_value": float(p_hist),
                "retrieval_minus_matched_hist_accuracy": _accuracy(y_true, retrieval_preds) - _accuracy(y_true, matched_hist_preds),
                "retrieval_minus_matched_hist_f1_macro": _macro_f1(y_true, retrieval_preds) - _macro_f1(y_true, matched_hist_preds),
            }
        )

    write_results_csv(Path(args.summary_output), summary_rows)
    write_results_csv(Path(args.detail_output), detail_rows)
    print(f"Saved summary to {args.summary_output}")
    print(f"Saved detail to {args.detail_output}")


if __name__ == "__main__":
    main()
