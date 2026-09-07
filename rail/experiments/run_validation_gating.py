from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from rail.analysis.compute_leq import _build_preprocessor
from rail.baselines.retrieval_icl import RetrievalICLRunner
from rail.baselines.vanilla_icl import VanillaICLRunner
from rail.config import SiliconFlowConfig
from rail.data import load_openml_dataset, train_val_test_split
from rail.evaluation.io import append_jsonl, write_results_csv
from rail.experiments.run_icl_smoke import resolve_model
from rail.llm import build_backend


DATASETS = {
    "adult": {"dataset_id": 1590},
    "credit-g": {"dataset_id": 31},
    "diabetes": {"dataset_id": 37},
    "letter": {"dataset_id": 6},
    "page-blocks": {"dataset_id": 30},
    "phoneme": {"dataset_id": 1489},
    "satimage": {"dataset_id": 182},
    "segment": {"dataset_id": 36},
    "vehicle": {"dataset_id": 54},
}

DEFAULT_THRESHOLDS = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]


def _label_free_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_query: pd.DataFrame,
    *,
    k: int,
) -> list[float]:
    preprocessor = _build_preprocessor(X_train)
    X_train_matrix = preprocessor.fit_transform(X_train)
    X_query_matrix = preprocessor.transform(X_query)
    n_neighbors = min(k, len(X_train))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X_train_matrix)
    _, indices = nn.kneighbors(X_query_matrix)

    y_values = np.asarray(y_train.astype(str))
    global_counts = pd.Series(y_values).value_counts(normalize=True)
    global_majority = float(global_counts.max())
    scores: list[float] = []
    for idx in indices:
        local_counts = pd.Series(y_values[idx]).value_counts(normalize=True)
        scores.append(float(local_counts.max() - global_majority))
    return scores


def _metrics(y_true: list[Any], y_pred: list[Any]) -> dict[str, float]:
    y_true_str = list(map(str, y_true))
    y_pred_str = list(map(str, y_pred))
    return {
        "accuracy": float(accuracy_score(y_true_str, y_pred_str)),
        "f1_macro": float(f1_score(y_true_str, y_pred_str, average="macro", zero_division=0)),
    }


def _hybrid_predictions(
    vanilla_pred: list[Any],
    retrieval_pred: list[Any],
    scores: list[float],
    threshold: float,
) -> tuple[list[Any], list[bool]]:
    use_retrieval = [score >= threshold for score in scores]
    hybrid = [ret if use else van for van, ret, use in zip(vanilla_pred, retrieval_pred, use_retrieval, strict=True)]
    return hybrid, use_retrieval


def _select_threshold(
    *,
    y_true: list[Any],
    vanilla_pred: list[Any],
    retrieval_pred: list[Any],
    scores: list[float],
    thresholds: list[float],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        hybrid, use_retrieval = _hybrid_predictions(vanilla_pred, retrieval_pred, scores, threshold)
        metric = _metrics(y_true, hybrid)
        rows.append(
            {
                "threshold": float(threshold),
                "retrieval_rate": float(np.mean(use_retrieval)),
                **metric,
            }
        )
    best = max(rows, key=lambda row: (row["accuracy"], row["f1_macro"], -row["retrieval_rate"]))
    return {"selected_threshold": best["threshold"], "validation_rows": rows, "selected_row": best}


def _random_gate_predictions(
    vanilla_pred: list[Any],
    retrieval_pred: list[Any],
    *,
    retrieval_rate: float,
    seed: int,
) -> tuple[list[Any], list[bool]]:
    n = len(vanilla_pred)
    n_retrieve = int(round(retrieval_rate * n))
    selected = set(random.Random(seed).sample(range(n), k=min(n_retrieve, n)))
    use_retrieval = [idx in selected for idx in range(n)]
    hybrid = [ret if use else van for idx, (van, ret, use) in enumerate(zip(vanilla_pred, retrieval_pred, use_retrieval, strict=True))]
    return hybrid, use_retrieval


def _run_phase(
    *,
    phase: str,
    dataset_name: str,
    model_name: str,
    vanilla_runner: VanillaICLRunner,
    retrieval_runner: RetrievalICLRunner,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    target_name: str,
    k: int,
    max_examples: int,
    thinking_budget: int | None,
    sleep_seconds: float,
    progress_log: Path | None,
    progress_every: int,
) -> dict[str, Any]:
    X_eval = X_eval.head(max_examples)
    y_eval = y_eval.head(max_examples)
    label_space = sorted(y_train.dropna().unique().tolist(), key=lambda value: str(value))
    scores = _label_free_scores(X_train, y_train, X_eval, k=k)

    vanilla = vanilla_runner.run(
        X_train=X_train,
        y_train=y_train,
        X_test=X_eval,
        y_test=y_eval,
        target_name=target_name,
        task_type="classification",
        k=k,
        model=model_name,
        max_examples=max_examples,
        thinking_budget=thinking_budget,
        sleep_seconds=sleep_seconds,
        progress_log_path=progress_log,
        progress_every=progress_every,
        progress_context={"dataset": dataset_name, "phase": phase, "policy_component": "vanilla"},
    )
    retrieval = retrieval_runner.run(
        X_train=X_train,
        y_train=y_train,
        X_test=X_eval,
        y_test=y_eval,
        target_name=target_name,
        task_type="classification",
        k=k,
        model=model_name,
        max_examples=max_examples,
        thinking_budget=thinking_budget,
        sleep_seconds=sleep_seconds,
        progress_log_path=progress_log,
        progress_every=progress_every,
        progress_context={"dataset": dataset_name, "phase": phase, "policy_component": "retrieval"},
    )

    detail_rows: list[dict[str, Any]] = []
    for pos, (truth, vanilla_pred, retrieval_pred, score) in enumerate(
        zip(y_eval.tolist(), vanilla["y_pred"], retrieval["y_pred"], scores, strict=True)
    ):
        detail_rows.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": dataset_name,
                "phase": phase,
                "eval_position": pos,
                "row_index": int(X_eval.index[pos]),
                "ground_truth": truth,
                "vanilla_prediction": vanilla_pred,
                "retrieval_prediction": retrieval_pred,
                "label_free_lba": score,
                "vanilla_correct": str(vanilla_pred) == str(truth),
                "retrieval_correct": str(retrieval_pred) == str(truth),
                "label_space_size": len(label_space),
            }
        )
    return {
        "y_true": y_eval.tolist(),
        "vanilla_pred": vanilla["y_pred"],
        "retrieval_pred": retrieval["y_pred"],
        "scores": scores,
        "detail_rows": detail_rows,
        "vanilla_metrics": vanilla["metrics"],
        "retrieval_metrics": retrieval["metrics"],
    }


def run_dataset(
    *,
    dataset_name: str,
    model_name: str,
    backend: Any,
    seed: int,
    k: int,
    val_examples: int,
    test_examples: int,
    thresholds: list[float],
    thinking_budget: int | None,
    sleep_seconds: float,
    progress_log: Path | None,
    progress_every: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_info = DATASETS[dataset_name]
    dataset = load_openml_dataset(
        dataset_id=int(dataset_info["dataset_id"]),
        task_type_override="classification",
    )
    split = train_val_test_split(
        dataset.features,
        dataset.target,
        task_type="classification",
        test_size=0.2,
        val_size=0.2,
        random_state=seed,
    )

    vanilla_runner = VanillaICLRunner(backend, seed=seed)
    retrieval_runner = RetrievalICLRunner(
        backend,
        seed=seed,
        selection_strategy="matched_pair_boundary",
        boundary_estimator_name="xgboost",
    )
    val = _run_phase(
        phase="validation",
        dataset_name=dataset.name,
        model_name=model_name,
        vanilla_runner=vanilla_runner,
        retrieval_runner=retrieval_runner,
        X_train=split.X_train,
        y_train=split.y_train,
        X_eval=split.X_val,
        y_eval=split.y_val,
        target_name=dataset.target_name,
        k=k,
        max_examples=val_examples,
        thinking_budget=thinking_budget,
        sleep_seconds=sleep_seconds,
        progress_log=progress_log,
        progress_every=progress_every,
    )
    selection = _select_threshold(
        y_true=val["y_true"],
        vanilla_pred=val["vanilla_pred"],
        retrieval_pred=val["retrieval_pred"],
        scores=val["scores"],
        thresholds=thresholds,
    )

    test = _run_phase(
        phase="test",
        dataset_name=dataset.name,
        model_name=model_name,
        vanilla_runner=vanilla_runner,
        retrieval_runner=retrieval_runner,
        X_train=split.X_train,
        y_train=split.y_train,
        X_eval=split.X_test,
        y_eval=split.y_test,
        target_name=dataset.target_name,
        k=k,
        max_examples=test_examples,
        thinking_budget=thinking_budget,
        sleep_seconds=sleep_seconds,
        progress_log=progress_log,
        progress_every=progress_every,
    )

    selected_threshold = float(selection["selected_threshold"])
    leq_pred, leq_use = _hybrid_predictions(
        test["vanilla_pred"],
        test["retrieval_pred"],
        test["scores"],
        selected_threshold,
    )
    random_pred, random_use = _random_gate_predictions(
        test["vanilla_pred"],
        test["retrieval_pred"],
        retrieval_rate=float(np.mean(leq_use)),
        seed=seed + 991,
    )

    summary_rows = [
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset.name,
            "phase": "validation",
            "policy": "selected_threshold",
            "selected_threshold": selected_threshold,
            "retrieval_rate": selection["selected_row"]["retrieval_rate"],
            "accuracy": selection["selected_row"]["accuracy"],
            "f1_macro": selection["selected_row"]["f1_macro"],
            "val_examples": val_examples,
            "test_examples": test_examples,
            "k": k,
            "model": model_name,
        }
    ]
    for row in selection["validation_rows"]:
        summary_rows.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": dataset.name,
                "phase": "validation_threshold_sweep",
                "policy": "label_free_gate",
                "selected_threshold": selected_threshold,
                **row,
                "val_examples": val_examples,
                "test_examples": test_examples,
                "k": k,
                "model": model_name,
            }
        )

    test_policies = [
        ("never_retrieve", test["vanilla_pred"], [False] * len(test["y_true"])),
        ("always_retrieve", test["retrieval_pred"], [True] * len(test["y_true"])),
        ("random_gate", random_pred, random_use),
        ("label_free_gate", leq_pred, leq_use),
    ]
    for policy, predictions, use_retrieval in test_policies:
        summary_rows.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": dataset.name,
                "phase": "test",
                "policy": policy,
                "selected_threshold": selected_threshold if policy == "label_free_gate" else None,
                "threshold": selected_threshold if policy == "label_free_gate" else None,
                "retrieval_rate": float(np.mean(use_retrieval)),
                **_metrics(test["y_true"], predictions),
                "val_examples": val_examples,
                "test_examples": test_examples,
                "k": k,
                "model": model_name,
            }
        )

    detail_rows = val["detail_rows"] + test["detail_rows"]
    for row, use_retrieval, pred in zip(
        [r for r in detail_rows if r["phase"] == "test"],
        leq_use,
        leq_pred,
        strict=True,
    ):
        row["selected_threshold"] = selected_threshold
        row["label_free_gate_use_retrieval"] = bool(use_retrieval)
        row["label_free_gate_prediction"] = pred
        row["label_free_gate_correct"] = str(pred) == str(row["ground_truth"])

    return summary_rows, detail_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation-selected label-free retrieval gating.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--model", default="latest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--val-examples", type=int, default=50)
    parser.add_argument("--test-examples", type=int, default=100)
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--thinking-budget", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--request-log", default="outputs/logs/validation_gating_requests.jsonl")
    parser.add_argument("--progress-log", default="outputs/logs/validation_gating_progress.jsonl")
    parser.add_argument("--summary-output", default="outputs/validation_gating_summary.csv")
    parser.add_argument("--detail-output", default="outputs/validation_gating_detail.csv")
    args = parser.parse_args()

    unknown = sorted(set(args.datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")

    config = SiliconFlowConfig.from_env()
    model_name = resolve_model(args.model, config)
    backend = build_backend(
        config,
        timeout_seconds=args.timeout_seconds,
        request_log_path=args.request_log,
    )

    all_summary_rows: list[dict[str, Any]] = []
    all_detail_rows: list[dict[str, Any]] = []
    progress_log = PROJECT_ROOT / args.progress_log if args.progress_log else None
    for dataset_name in args.datasets:
        print(f"[{datetime.now(timezone.utc).isoformat()}] Running {dataset_name}")
        summary_rows, detail_rows = run_dataset(
            dataset_name=dataset_name,
            model_name=model_name,
            backend=backend,
            seed=args.seed,
            k=args.k,
            val_examples=args.val_examples,
            test_examples=args.test_examples,
            thresholds=args.thresholds,
            thinking_budget=args.thinking_budget,
            sleep_seconds=args.sleep_seconds,
            progress_log=progress_log,
            progress_every=args.progress_every,
        )
        all_summary_rows.extend(summary_rows)
        all_detail_rows.extend(detail_rows)
        write_results_csv(PROJECT_ROOT / args.summary_output, all_summary_rows)
        write_results_csv(PROJECT_ROOT / args.detail_output, all_detail_rows)
        append_jsonl(
            progress_log or (PROJECT_ROOT / "outputs/logs/validation_gating_progress.jsonl"),
            {"dataset": dataset_name, "status": "completed", "summary_rows": len(summary_rows)},
        )

    print(f"Saved summary to {PROJECT_ROOT / args.summary_output}")
    print(f"Saved detail to {PROJECT_ROOT / args.detail_output}")


if __name__ == "__main__":
    main()
