from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from rail.baselines.traditional_ml import fit_baseline
from rail.data import load_openml_dataset, train_val_test_split
from rail.evaluation.io import write_results_csv
from rail.retrieval.difficulty import compute_classification_oof_uncertainty_scores
from rail.retrieval.knn_retriever import TabularKNNRetriever
from rail.retrieval.selectors import select_indices


DATASETS: dict[str, int] = {
    "adult": 1590,
    "credit-g": 31,
    "diabetes": 37,
    "phoneme": 1489,
    "vehicle": 54,
    "segment": 36,
    "page-blocks": 30,
    "satimage": 182,
    "letter": 6,
}

SELECTORS: dict[str, str] = {
    "knn_topk": "topk",
    "class_balanced_knn": "balanced",
    "diversity_knn": "diverse",
    "class_balanced_diversity_knn": "class_balanced_diversity",
    "uncertainty_only_boundary": "uncertainty_only_boundary",
    "matched_pair_boundary": "matched_pair_boundary",
}


def _label_entropy(labels: list[Any]) -> float:
    if not labels:
        return 0.0
    counts = Counter(map(str, labels))
    total = float(sum(counts.values()))
    return float(-sum((count / total) * math.log(count / total) for count in counts.values()))


def _support_quality_metrics(
    *,
    true_label: Any,
    selected_labels: list[Any],
    selected_distances: list[float],
    selected_uncertainties: list[float],
) -> dict[str, float]:
    if not selected_labels:
        return {
            "true_label_purity": 0.0,
            "local_margin": 0.0,
            "label_entropy": 0.0,
            "class_coverage": 0.0,
            "mean_standardized_distance": 0.0,
            "mean_boundary_uncertainty": 0.0,
        }
    labels = [str(label) for label in selected_labels]
    truth = str(true_label)
    counts = Counter(labels)
    total = float(len(labels))
    true_freq = counts.get(truth, 0) / total
    competing = [count / total for label, count in counts.items() if label != truth]
    max_competing = max(competing) if competing else 0.0
    return {
        "true_label_purity": float(true_freq),
        "local_margin": float(true_freq - max_competing),
        "label_entropy": _label_entropy(labels),
        "class_coverage": float(len(counts)),
        "mean_standardized_distance": float(np.mean(selected_distances)) if selected_distances else 0.0,
        "mean_boundary_uncertainty": float(np.mean(selected_uncertainties)) if selected_uncertainties else 0.0,
    }


def _row_distance(candidate_matrix, query_vector, position: int) -> float:
    diff = candidate_matrix[position] - query_vector
    if sparse.issparse(diff):
        return float(np.sqrt(diff.multiply(diff).sum()))
    arr = np.asarray(diff)
    return float(np.sqrt(np.sum(arr * arr)))


def _priority_labels(boundary_model: Any, query: pd.Series, columns: pd.Index) -> tuple[list[Any], float | None]:
    query_frame = pd.DataFrame([query], columns=columns)
    probabilities = boundary_model.predict_proba(query_frame)[0]
    if boundary_model.label_encoder is not None:
        class_labels = list(boundary_model.label_encoder.classes_)
    else:
        class_labels = []
    ranked = np.argsort(-probabilities)
    top = ranked[: min(2, len(ranked))]
    labels = [class_labels[int(pos)] for pos in top]
    margin = None
    if len(top) >= 2:
        margin = float(probabilities[int(top[0])] - probabilities[int(top[1])])
    return labels, margin


def _evaluate_dataset(
    dataset_name: str,
    *,
    seed: int,
    k: int,
    max_queries: int | None,
    oof_splits: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = load_openml_dataset(
        dataset_id=DATASETS[dataset_name],
        task_type_override="classification",
    )
    split = train_val_test_split(
        dataset.features,
        dataset.target,
        task_type="classification",
        random_state=seed,
    )
    X_eval = split.X_test.head(max_queries) if max_queries is not None else split.X_test
    y_eval = split.y_test.head(max_queries) if max_queries is not None else split.y_test
    label_space = sorted(split.y_train.dropna().unique().tolist(), key=lambda value: str(value))

    retriever = TabularKNNRetriever().fit(split.X_train)
    uncertainty_scores = compute_classification_oof_uncertainty_scores(
        split.X_train,
        split.y_train,
        random_state=seed,
        n_splits=oof_splits,
    )
    boundary_model = fit_baseline(
        "xgboost",
        split.X_train,
        split.y_train,
        task_type="classification",
        random_state=seed,
    )

    detail_rows: list[dict[str, Any]] = []
    candidate_k = min(max(k * 8, 64), len(split.X_train))
    timestamp = datetime.now(timezone.utc).isoformat()
    for eval_position, (row_index, row) in enumerate(X_eval.iterrows()):
        result = retriever.retrieve(row, k=candidate_k)
        candidate_matrix = retriever.get_candidate_matrix(result.indices)
        query_vector = retriever.transform_query(row)
        priorities, query_margin = _priority_labels(boundary_model, row, split.X_train.columns)
        position_by_index = {idx: pos for pos, idx in enumerate(result.indices)}

        for selector_name, strategy in SELECTORS.items():
            selected = select_indices(
                result.indices,
                split.y_train,
                k=k,
                strategy=strategy,
                label_space=label_space,
                candidate_matrix=candidate_matrix,
                query_vector=query_vector,
                uncertainty_scores=uncertainty_scores,
                priority_labels=priorities,
                query_margin=query_margin,
            )
            selected_labels = [split.y_train.iloc[idx] for idx in selected]
            positions = [position_by_index[idx] for idx in selected if idx in position_by_index]
            distances = [_row_distance(candidate_matrix, query_vector, pos) for pos in positions]
            uncertainties = [float(uncertainty_scores[idx]) for idx in selected]
            metric_row = _support_quality_metrics(
                true_label=y_eval.loc[row_index],
                selected_labels=selected_labels,
                selected_distances=distances,
                selected_uncertainties=uncertainties,
            )
            detail_rows.append(
                {
                    "timestamp_utc": timestamp,
                    "dataset": dataset.name,
                    "eval_position": eval_position,
                    "row_index": int(row_index),
                    "selector": selector_name,
                    "strategy": strategy,
                    "k": k,
                    "candidate_k": candidate_k,
                    "true_label": y_eval.loc[row_index],
                    "query_top1_label": priorities[0] if priorities else None,
                    "query_top2_label": priorities[1] if len(priorities) > 1 else None,
                    "query_margin": query_margin,
                    **metric_row,
                }
            )

    frame = pd.DataFrame(detail_rows)
    summary_rows: list[dict[str, Any]] = []
    metric_columns = [
        "true_label_purity",
        "local_margin",
        "label_entropy",
        "class_coverage",
        "mean_standardized_distance",
        "mean_boundary_uncertainty",
    ]
    for selector_name, group in frame.groupby("selector", sort=False):
        row: dict[str, Any] = {
            "timestamp_utc": timestamp,
            "dataset": dataset.name,
            "selector": selector_name,
            "strategy": group["strategy"].iloc[0],
            "queries": int(group["eval_position"].nunique()),
            "k": k,
            "candidate_k": candidate_k,
        }
        for metric in metric_columns:
            summary_metric = metric if metric.startswith("mean_") else f"mean_{metric}"
            row[summary_metric] = float(group[metric].mean())
        summary_rows.append(row)
    return summary_rows, detail_rows


def _format_metric_cell(row: pd.Series) -> str:
    return f"{row['mean_true_label_purity']:.3f}/{row['mean_local_margin']:.3f}"


def _write_markdown(summary: pd.DataFrame, output: Path) -> None:
    selector_order = list(SELECTORS)
    labels = {
        "knn_topk": "kNN",
        "class_balanced_knn": "Bal-kNN",
        "diversity_knn": "Div-kNN",
        "class_balanced_diversity_knn": "Bal+Div",
        "uncertainty_only_boundary": "Boundary",
        "matched_pair_boundary": "Matched-pair",
    }
    lines = [
        "# ICDM Selector Ablation Proxy Results",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Each cell reports `mean true-label purity / mean local margin` over the test split.",
        "",
        "| Dataset | " + " | ".join(labels[name] for name in selector_order) + " |",
        "|---|" + "|".join("---:" for _ in selector_order) + "|",
    ]
    for dataset in sorted(summary["dataset"].unique()):
        sub = summary[summary["dataset"] == dataset].set_index("selector")
        cells = []
        for selector in selector_order:
            if selector in sub.index:
                cells.append(_format_metric_cell(sub.loc[selector]))
            else:
                cells.append("")
        lines.append(f"| {dataset} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- This is a query-label-aware support-quality proxy, not an LLM accuracy table.",
            "- `uncertainty_only_boundary` uses the auxiliary classifier only to locate uncertain boundary-like training examples; it does not construct matched support/counterexample pairs.",
            "- `matched_pair_boundary` should be favored only when it improves local evidence quality rather than because it is defined as the method.",
            "",
            "## Artifact Paths",
            "",
            "- Summary CSV: `outputs/selector_ablation_proxy_summary.csv`",
            "- Query-level detail CSV: `outputs/selector_ablation_proxy_detail.csv`",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute offline selector-ablation support-quality proxies.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--max-queries", type=int, default=0, help="<=0 means full test split.")
    parser.add_argument("--oof-splits", type=int, default=3)
    parser.add_argument("--summary-output", default="outputs/selector_ablation_proxy_summary.csv")
    parser.add_argument("--detail-output", default="outputs/selector_ablation_proxy_detail.csv")
    parser.add_argument("--markdown-output", default="outputs/selector_ablation_proxy.md")
    args = parser.parse_args()

    max_queries = args.max_queries if args.max_queries > 0 else None
    all_summary: list[dict[str, Any]] = []
    all_detail: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        print(f"[{datetime.now(timezone.utc).isoformat()}] selector proxy: {dataset_name}", flush=True)
        summary_rows, detail_rows = _evaluate_dataset(
            dataset_name,
            seed=args.seed,
            k=args.k,
            max_queries=max_queries,
            oof_splits=args.oof_splits,
        )
        all_summary.extend(summary_rows)
        all_detail.extend(detail_rows)
        write_results_csv(PROJECT_ROOT / args.summary_output, all_summary)
        write_results_csv(PROJECT_ROOT / args.detail_output, all_detail)

    summary_frame = pd.DataFrame(all_summary)
    _write_markdown(summary_frame, PROJECT_ROOT / args.markdown_output)
    print(f"Saved summary to {PROJECT_ROOT / args.summary_output}")
    print(f"Saved detail to {PROJECT_ROOT / args.detail_output}")
    print(f"Saved markdown to {PROJECT_ROOT / args.markdown_output}")


if __name__ == "__main__":
    main()
