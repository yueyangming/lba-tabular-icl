from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from rail.baselines import run_retrieval_icl, run_vanilla_icl
from rail.config import SiliconFlowConfig
from rail.context.serializer import serialize_example
from rail.data import load_openml_dataset, train_val_test_split
from rail.evaluation.io import append_jsonl, ensure_parent_dir, write_results_csv
from rail.experiments.run_icl_smoke import resolve_model
from rail.llm import build_backend


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _label_histogram(values: list[Any]) -> dict[str, int]:
    return {str(label): count for label, count in Counter(map(str, values)).items()}


def _correct(task_type: str, prediction: Any, ground_truth: Any) -> bool | None:
    if ground_truth is None or task_type != "classification":
        return None
    return str(prediction) == str(ground_truth)


def _flip_type(vanilla_correct: bool | None, retrieval_correct: bool | None) -> str:
    if vanilla_correct is None or retrieval_correct is None:
        return "unknown"
    if vanilla_correct and retrieval_correct:
        return "stable_correct"
    if not vanilla_correct and not retrieval_correct:
        return "stable_wrong"
    if vanilla_correct and not retrieval_correct:
        return "correct_to_wrong"
    return "wrong_to_correct"


def _trace_to_record(
    trace: dict[str, Any],
    *,
    query_text: str,
    strategy: str,
    llm_model: str,
    dataset_name: str,
    task_type: str,
) -> dict[str, Any]:
    selected_targets = trace.get("selected_targets", [])
    record = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "dataset": dataset_name,
        "task_type": task_type,
        "llm_model": llm_model,
        "method": trace["method"],
        "selection_strategy": strategy,
        "eval_position": int(trace["eval_position"]),
        "row_index": int(trace["row_index"]),
        "ground_truth": _jsonable(trace.get("ground_truth")),
        "prediction": _jsonable(trace.get("prediction")),
        "raw_output": trace.get("raw_output"),
        "query_text": query_text,
        "selected_indices": _jsonable(trace.get("selected_indices")),
        "selected_targets": _jsonable(selected_targets),
        "selected_label_hist": _jsonable(_label_histogram(selected_targets)),
        "candidate_indices": _jsonable(trace.get("candidate_indices")),
        "priority_labels": _jsonable(trace.get("priority_labels")),
        "priority_weights": _jsonable(trace.get("priority_weights")),
        "query_class_probabilities": _jsonable(trace.get("query_class_probabilities")),
        "effective_selection_strategy": _jsonable(trace.get("effective_selection_strategy")),
        "query_margin": _jsonable(trace.get("query_margin")),
        "query_committee_disagreement": _jsonable(trace.get("query_committee_disagreement")),
        "query_gate_score": _jsonable(trace.get("query_gate_score")),
    }
    record["correct"] = _correct(task_type, record["prediction"], record["ground_truth"])
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run per-sample error analysis for vanilla vs retrieval ICL.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-id", type=int, default=None)
    parser.add_argument("--model", default="latest", help="Model alias: debug/default/latest or full name.")
    parser.add_argument("--strategies", nargs="+", default=["balanced", "adaptive_boundary"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--thinking-budget", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--output-dir", default="outputs/error_analysis")
    parser.add_argument("--request-log", default="outputs/logs/error_analysis_requests.jsonl")
    args = parser.parse_args()

    config = SiliconFlowConfig.from_env()
    model_name = resolve_model(args.model, config)
    backend = build_backend(
        config,
        timeout_seconds=args.timeout_seconds,
        request_log_path=args.request_log,
    )

    dataset = load_openml_dataset(dataset_id=args.dataset_id, name=args.dataset)
    split = train_val_test_split(
        dataset.features,
        dataset.target,
        task_type=dataset.task_type,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
    )
    X_eval = split.X_test.head(args.max_examples)

    output_dir = Path(args.output_dir) / dataset.name / model_name.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    flip_rows: list[dict[str, Any]] = []

    vanilla_result = run_vanilla_icl(
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
    )
    summary_rows.append(
        {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "dataset": dataset.name,
            "task_type": dataset.task_type,
            "llm_model": model_name,
            "method": "vanilla_icl",
            "selection_strategy": "none",
            **vanilla_result["metrics"],
        }
    )

    vanilla_records_by_position: dict[int, dict[str, Any]] = {}
    for trace in vanilla_result["traces"]:
        row_index = trace["row_index"]
        query_text = serialize_example(X_eval.loc[row_index], include_target=False, target_name=dataset.target_name)
        record = _trace_to_record(
            trace,
            query_text=query_text,
            strategy="none",
            llm_model=model_name,
            dataset_name=dataset.name,
            task_type=dataset.task_type,
        )
        trace_rows.append(record)
        vanilla_records_by_position[int(trace["eval_position"])] = record

    for strategy in args.strategies:
        retrieval_result = run_retrieval_icl(
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
            selection_strategy=strategy,
        )
        summary_rows.append(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "dataset": dataset.name,
                "task_type": dataset.task_type,
                "llm_model": model_name,
                "method": "retrieval_icl",
                "selection_strategy": strategy,
                **retrieval_result["metrics"],
            }
        )
        for trace in retrieval_result["traces"]:
            row_index = trace["row_index"]
            query_text = serialize_example(X_eval.loc[row_index], include_target=False, target_name=dataset.target_name)
            record = _trace_to_record(
                trace,
                query_text=query_text,
                strategy=strategy,
                llm_model=model_name,
                dataset_name=dataset.name,
                task_type=dataset.task_type,
            )
            trace_rows.append(record)
            vanilla_record = vanilla_records_by_position[int(trace["eval_position"])]
            flip_rows.append(
                {
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "dataset": dataset.name,
                    "llm_model": model_name,
                    "selection_strategy": strategy,
                    "eval_position": int(trace["eval_position"]),
                    "row_index": int(trace["row_index"]),
                    "query_text": query_text,
                    "ground_truth": _jsonable(record["ground_truth"]),
                    "vanilla_prediction": _jsonable(vanilla_record["prediction"]),
                    "retrieval_prediction": _jsonable(record["prediction"]),
                    "vanilla_correct": vanilla_record["correct"],
                    "retrieval_correct": record["correct"],
                    "flip_type": _flip_type(vanilla_record["correct"], record["correct"]),
                    "vanilla_selected_targets": _jsonable(vanilla_record["selected_targets"]),
                    "retrieval_selected_targets": _jsonable(record["selected_targets"]),
                    "retrieval_selected_label_hist": _jsonable(record["selected_label_hist"]),
                    "priority_labels": _jsonable(record["priority_labels"]),
                    "priority_weights": _jsonable(record["priority_weights"]),
                    "query_class_probabilities": _jsonable(record["query_class_probabilities"]),
                    "effective_selection_strategy": _jsonable(record["effective_selection_strategy"]),
                    "query_margin": _jsonable(record["query_margin"]),
                    "query_committee_disagreement": _jsonable(record["query_committee_disagreement"]),
                    "query_gate_score": _jsonable(record["query_gate_score"]),
                }
            )

    write_results_csv(output_dir / "summary.csv", summary_rows)
    write_results_csv(output_dir / "trace.csv", trace_rows)
    write_results_csv(output_dir / "flips.csv", flip_rows)

    trace_jsonl = ensure_parent_dir(output_dir / "trace.jsonl")
    flips_jsonl = ensure_parent_dir(output_dir / "flips.jsonl")
    for row in trace_rows:
        append_jsonl(trace_jsonl, row)
    for row in flip_rows:
        append_jsonl(flips_jsonl, row)

    metadata = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "dataset": dataset.name,
        "dataset_id": dataset.dataset_id,
        "task_type": dataset.task_type,
        "llm_model": model_name,
        "strategies": args.strategies,
        "k": args.k,
        "max_examples": args.max_examples,
        "output_dir": str(output_dir),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
