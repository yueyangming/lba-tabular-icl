from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rail.baselines.vanilla_icl import ICLExample, VanillaICLRunner, build_vanilla_prompt
from rail.baselines.traditional_ml import BaselineName, fit_baseline
from rail.evaluation.io import append_jsonl
from rail.evaluation.metrics import compute_metrics
from rail.llm.base import ChatBackend
from rail.llm.cache import LLMCache
from rail.retrieval.difficulty import (
    compute_classification_committee_scores,
    compute_classification_difficulty_scores,
    compute_classification_oof_margin_scores,
    compute_classification_oof_uncertainty_scores,
)
from rail.retrieval.knn_retriever import TabularKNNRetriever
from rail.retrieval.selectors import SelectionStrategy, select_indices


class RetrievalICLRunner:
    def __init__(
        self,
        backend: ChatBackend,
        *,
        retriever: TabularKNNRetriever | None = None,
        cache: LLMCache | None = None,
        seed: int = 42,
        selection_strategy: SelectionStrategy = "balanced",
        boundary_estimator_name: BaselineName = "xgboost",
    ) -> None:
        self.backend = backend
        self.retriever = retriever or TabularKNNRetriever()
        self.cache = cache or LLMCache()
        self.seed = seed
        self.selection_strategy = selection_strategy
        self.boundary_estimator_name = boundary_estimator_name
        self._difficulty_scores = None
        self._boundary_model = None
        self._margin_scores = None
        self._class_priors = None
        self._committee_model = None

    def predict_one(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        query: pd.Series,
        target_name: str,
        task_type: str,
        label_space: list[Any] | None = None,
        k: int = 16,
        model: str | None = None,
        system_prompt: str | None = None,
        thinking_budget: int | None = None,
    ) -> str:
        trace = self.predict_one_with_trace(
            X_train=X_train,
            y_train=y_train,
            query=query,
            target_name=target_name,
            task_type=task_type,
            label_space=label_space,
            k=k,
            model=model,
            system_prompt=system_prompt,
            thinking_budget=thinking_budget,
        )
        return str(trace["response"])

    def predict_one_with_trace(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        query: pd.Series,
        target_name: str,
        task_type: str,
        label_space: list[Any] | None = None,
        k: int = 16,
        model: str | None = None,
        system_prompt: str | None = None,
        thinking_budget: int | None = None,
    ) -> dict[str, Any]:
        candidate_k = (k * 4) if task_type == "classification" else k
        if task_type == "classification" and self.selection_strategy in {"contrastive", "class_balanced_diversity"}:
            candidate_k = max(candidate_k, k * 8, 64)
        if task_type == "classification" and self.selection_strategy in {"difficulty", "oof_difficulty", "committee", "boundary", "uncertainty_only_boundary", "adaptive_boundary", "counter_boundary", "paired_boundary", "matched_pair_boundary", "knn_pair_boundary", "bidirectional_pair_boundary", "skeptical_pair_boundary", "margin_pair_rerank", "density_margin_pair_rerank", "relevance_constrained_density_rerank", "imbalance_pair_rerank", "query_difficulty_gated_rerank"}:
            candidate_k = max(candidate_k, k * 8, 64)
        result = self.retriever.retrieve(query, k=candidate_k)
        candidate_matrix = self.retriever.get_candidate_matrix(result.indices)
        query_vector = self.retriever.transform_query(query)
        priority_labels = None
        priority_weights = None
        query_class_probabilities = None
        query_margin = None
        query_committee_disagreement = None
        query_gate_score = None
        effective_selection_strategy = self.selection_strategy
        if task_type == "classification" and self.selection_strategy in {"boundary", "uncertainty_only_boundary", "adaptive_boundary", "counter_boundary", "paired_boundary", "matched_pair_boundary", "bidirectional_pair_boundary", "skeptical_pair_boundary", "margin_pair_rerank", "density_margin_pair_rerank", "relevance_constrained_density_rerank", "imbalance_pair_rerank", "query_difficulty_gated_rerank"} and self._boundary_model is not None:
            query_frame = pd.DataFrame([query], columns=X_train.columns)
            probabilities = self._boundary_model.predict_proba(query_frame)[0]
            query_class_probabilities = probabilities.tolist()
            if self._boundary_model.label_encoder is not None:
                class_labels = list(self._boundary_model.label_encoder.classes_)
            else:
                class_labels = label_space or []
            ranked_classes = np.argsort(-probabilities)
            top_positions = ranked_classes[: min(2, len(ranked_classes))]
            priority_labels = [class_labels[int(pos)] for pos in top_positions]
            raw_weights = [float(probabilities[int(pos)]) for pos in top_positions]
            if len(raw_weights) >= 2:
                query_margin = raw_weights[0] - raw_weights[1]
            if raw_weights:
                max_weight = max(raw_weights)
                priority_weights = {
                    label: (weight / max_weight if max_weight > 0 else 1.0)
                    for label, weight in zip(priority_labels, raw_weights, strict=False)
                }
            if self.selection_strategy == "query_difficulty_gated_rerank" and self._committee_model is not None:
                xgb_pred = self._boundary_model.predict(query_frame)[0]
                knn_probabilities = self._committee_model.predict_proba(query_frame)[0]
                knn_pred = self._committee_model.predict(query_frame)[0]
                knn_top1 = float(np.max(knn_probabilities))
                xgb_top1 = float(np.max(probabilities))
                xgb_uncertainty = 1.0 - xgb_top1
                knn_uncertainty = 1.0 - knn_top1
                query_committee_disagreement = float(str(xgb_pred) != str(knn_pred))
                query_gate_score = (
                    0.45 * xgb_uncertainty
                    + 0.25 * knn_uncertainty
                    + 0.20 * query_committee_disagreement
                    + 0.10 * (1.0 - float(query_margin if query_margin is not None else 0.0))
                )
                effective_selection_strategy = (
                    "density_margin_pair_rerank"
                    if query_gate_score >= 0.40 or query_committee_disagreement >= 1.0
                    else "matched_pair_boundary"
                )
        chosen_indices = select_indices(
            result.indices,
            y_train,
            k=k,
            strategy=effective_selection_strategy,
            label_space=label_space,
            candidate_matrix=candidate_matrix,
            query_vector=query_vector,
            uncertainty_scores=self._difficulty_scores,
            margin_scores=self._margin_scores,
            class_priors=self._class_priors,
            priority_labels=priority_labels,
            priority_weights=priority_weights,
            query_margin=query_margin,
        )
        examples = [ICLExample(features=X_train.iloc[idx], target=y_train.iloc[idx]) for idx in chosen_indices]
        prompt = build_vanilla_prompt(
            examples=examples,
            query=query,
            target_name=target_name,
            task_type=task_type,
            label_space=label_space,
        )
        payload = {
            "method": "retrieval_icl",
            "prompt": prompt,
            "model": model,
            "system_prompt": system_prompt,
            "k": k,
            "task_type": task_type,
            "target_name": target_name,
            "indices": chosen_indices,
            "selection_strategy": self.selection_strategy,
            "effective_selection_strategy": effective_selection_strategy,
        }
        cached = self.cache.get(payload)
        if cached is not None:
            return {
                "response": cached,
                "prompt": prompt,
                "indices": chosen_indices,
                "candidate_indices": result.indices,
                "selected_targets": [y_train.iloc[idx] for idx in chosen_indices],
                "priority_labels": priority_labels,
                "priority_weights": priority_weights,
                "query_class_probabilities": query_class_probabilities,
                "query_margin": query_margin,
                "query_committee_disagreement": query_committee_disagreement,
                "query_gate_score": query_gate_score,
                "effective_selection_strategy": effective_selection_strategy,
            }
        response = self.backend.complete(
            prompt,
            model=model,
            system_prompt=system_prompt,
            thinking_budget=thinking_budget,
        )
        self.cache.set(payload, response)
        return {
            "response": response,
            "prompt": prompt,
            "indices": chosen_indices,
            "candidate_indices": result.indices,
            "selected_targets": [y_train.iloc[idx] for idx in chosen_indices],
            "priority_labels": priority_labels,
            "priority_weights": priority_weights,
            "query_class_probabilities": query_class_probabilities,
            "query_margin": query_margin,
            "query_committee_disagreement": query_committee_disagreement,
            "query_gate_score": query_gate_score,
            "effective_selection_strategy": effective_selection_strategy,
        }

    def run(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series | None,
        target_name: str,
        task_type: str,
        k: int = 16,
        model: str | None = None,
        system_prompt: str | None = None,
        max_examples: int | None = None,
        thinking_budget: int | None = None,
        sleep_seconds: float = 0.0,
        progress_log_path: str | Path | None = None,
        progress_every: int = 25,
        progress_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.retriever.fit(X_train)
        if task_type == "classification" and self.selection_strategy == "difficulty":
            self._difficulty_scores = compute_classification_difficulty_scores(X_train, y_train, random_state=self.seed)
            self._margin_scores = None
            self._boundary_model = None
            self._class_priors = None
        elif task_type == "classification" and self.selection_strategy == "oof_difficulty":
            self._difficulty_scores = compute_classification_oof_uncertainty_scores(
                X_train,
                y_train,
                random_state=self.seed,
            )
            self._margin_scores = None
            self._boundary_model = None
            self._class_priors = None
        elif task_type == "classification" and self.selection_strategy == "committee":
            self._difficulty_scores = compute_classification_committee_scores(
                X_train,
                y_train,
                random_state=self.seed,
            )
            self._margin_scores = None
            self._boundary_model = None
            self._class_priors = None
        elif task_type == "classification" and self.selection_strategy in {"boundary", "uncertainty_only_boundary", "adaptive_boundary", "counter_boundary", "paired_boundary", "matched_pair_boundary", "bidirectional_pair_boundary", "skeptical_pair_boundary", "margin_pair_rerank", "density_margin_pair_rerank", "relevance_constrained_density_rerank", "imbalance_pair_rerank", "query_difficulty_gated_rerank"}:
            self._difficulty_scores = compute_classification_oof_uncertainty_scores(
                X_train,
                y_train,
                random_state=self.seed,
            )
            self._margin_scores = compute_classification_oof_margin_scores(
                X_train,
                y_train,
                random_state=self.seed,
            )
            self._boundary_model = fit_baseline(
                self.boundary_estimator_name,
                X_train,
                y_train,
                task_type="classification",
                random_state=self.seed,
            )
            self._class_priors = {
                label: float(prob)
                for label, prob in y_train.value_counts(normalize=True).to_dict().items()
            }
            self._committee_model = fit_baseline(
                "knn",
                X_train,
                y_train,
                task_type="classification",
                random_state=self.seed,
            )
        elif task_type == "classification" and self.selection_strategy == "knn_pair_boundary":
            self._difficulty_scores = None
            self._margin_scores = None
            self._boundary_model = None
            self._class_priors = None
            self._committee_model = None
        else:
            self._difficulty_scores = None
            self._boundary_model = None
            self._margin_scores = None
            self._class_priors = None
            self._committee_model = None
        X_eval = X_test.head(max_examples) if max_examples is not None else X_test
        y_eval = y_test.head(max_examples) if y_test is not None and max_examples is not None else y_test
        label_space = sorted(y_train.dropna().unique().tolist(), key=lambda value: str(value)) if task_type == "classification" else None

        raw_outputs: list[str] = []
        predictions: list[Any] = []
        parse_failures = 0
        regression_fallback = float(pd.to_numeric(y_train, errors="coerce").dropna().median()) if task_type == "regression" else None
        traces: list[dict[str, Any]] = []
        for eval_position, (row_index, row) in enumerate(X_eval.iterrows()):
            trace = self.predict_one_with_trace(
                X_train=X_train,
                y_train=y_train,
                query=row,
                target_name=target_name,
                task_type=task_type,
                label_space=label_space,
                k=k,
                model=model,
                system_prompt=system_prompt,
                thinking_budget=thinking_budget,
            )
            response = str(trace["response"])
            raw_outputs.append(response)
            try:
                prediction = VanillaICLRunner.parse_prediction(
                    response,
                    task_type=task_type,
                    label_space=label_space,
                )
            except Exception:
                parse_failures += 1
                if task_type == "classification":
                    prediction = "__PARSE_ERROR__"
                else:
                    prediction = regression_fallback
            predictions.append(prediction)
            trace_row = {
                "eval_position": eval_position,
                "row_index": row_index,
                "method": "retrieval_icl",
                "selection_strategy": self.selection_strategy,
                "prediction": prediction,
                "raw_output": response,
                "selected_indices": trace["indices"],
                "candidate_indices": trace["candidate_indices"],
                "selected_targets": trace["selected_targets"],
                "priority_labels": trace["priority_labels"],
                "priority_weights": trace["priority_weights"],
                "query_class_probabilities": trace["query_class_probabilities"],
                "query_margin": trace["query_margin"],
                "query_committee_disagreement": trace["query_committee_disagreement"],
                "query_gate_score": trace["query_gate_score"],
                "effective_selection_strategy": trace["effective_selection_strategy"],
            }
            if y_eval is not None:
                trace_row["ground_truth"] = y_eval.loc[row_index]
            traces.append(trace_row)
            processed = eval_position + 1
            if (
                progress_log_path is not None
                and y_eval is not None
                and (processed % max(progress_every, 1) == 0 or processed == len(X_eval))
            ):
                metrics = compute_metrics(y_eval.iloc[:processed], predictions, task_type)
                progress_row: dict[str, Any] = {
                    "method": "retrieval_icl",
                    "selection_strategy": self.selection_strategy,
                    "processed_examples": processed,
                    "sample_count_target": len(X_eval),
                    "parse_failures": parse_failures,
                }
                if progress_context:
                    progress_row.update(progress_context)
                progress_row.update(metrics)
                append_jsonl(progress_log_path, progress_row)
            if sleep_seconds > 0 and eval_position + 1 < len(X_eval):
                time.sleep(sleep_seconds)

        result: dict[str, Any] = {
            "model_name": "retrieval_icl",
            "task_type": task_type,
            "sample_count": len(predictions),
            "raw_outputs": raw_outputs,
            "y_pred": predictions,
            "parse_failures": parse_failures,
            "selection_strategy": self.selection_strategy,
            "boundary_estimator_name": self.boundary_estimator_name,
            "traces": traces,
        }
        if y_eval is not None:
            result["metrics"] = compute_metrics(y_eval, predictions, task_type)
        return result


def run_retrieval_icl(
    backend: ChatBackend,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series | None,
    *,
    target_name: str,
    task_type: str,
    k: int = 16,
    seed: int = 42,
    model: str | None = None,
    max_examples: int | None = None,
    thinking_budget: int | None = None,
    selection_strategy: SelectionStrategy = "balanced",
    sleep_seconds: float = 0.0,
    progress_log_path: str | Path | None = None,
    progress_every: int = 25,
    progress_context: dict[str, Any] | None = None,
    boundary_estimator_name: BaselineName = "xgboost",
) -> dict[str, Any]:
    runner = RetrievalICLRunner(
        backend,
        seed=seed,
        selection_strategy=selection_strategy,
        boundary_estimator_name=boundary_estimator_name,
    )
    return runner.run(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        target_name=target_name,
        task_type=task_type,
        k=k,
        model=model,
        max_examples=max_examples,
        thinking_budget=thinking_budget,
        sleep_seconds=sleep_seconds,
        progress_log_path=progress_log_path,
        progress_every=progress_every,
        progress_context=progress_context,
    )
