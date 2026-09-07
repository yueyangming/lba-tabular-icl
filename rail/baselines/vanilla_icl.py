from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from rail.context.serializer import serialize_example
from rail.evaluation.io import append_jsonl
from rail.evaluation.metrics import compute_metrics
from rail.llm.base import ChatBackend
from rail.llm.cache import LLMCache


TaskType = Literal["classification", "regression"]


@dataclass
class ICLExample:
    features: pd.Series
    target: Any


def build_prompt(
    *,
    examples: list[ICLExample],
    query: pd.Series,
    target_name: str,
    task_type: TaskType,
    label_space: list[Any] | None = None,
) -> str:
    if task_type == "classification" and label_space:
        labels_text = ", ".join(str(label) for label in label_space)
        task_instruction = (
            f"You are solving a tabular classification task. "
            f"Use the examples to predict the {target_name} for the final query. "
            f"Return exactly one label from this set: {labels_text}."
        )
    else:
        task_instruction = (
            f"You are solving a tabular {task_type} task. "
            f"Use the examples to predict the {target_name} for the final query. "
            f"Return only the predicted {target_name}."
        )
    serialized_examples = "\n".join(
        f"Example {idx + 1}: {serialize_example(example.features, target=example.target, target_name=target_name)}"
        for idx, example in enumerate(examples)
    )
    query_text = serialize_example(query, include_target=False, target_name=target_name)
    return f"{task_instruction}\n\n{serialized_examples}\n\nQuery: {query_text}\nAnswer:"


def build_vanilla_prompt(
    *,
    examples: list[ICLExample],
    query: pd.Series,
    target_name: str,
    task_type: TaskType,
    label_space: list[Any] | None = None,
) -> str:
    return build_prompt(
        examples=examples,
        query=query,
        target_name=target_name,
        task_type=task_type,
        label_space=label_space,
    )


def sample_demonstrations(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    task_type: TaskType,
    k: int,
    seed: int = 42,
    class_balance: bool = True,
) -> list[ICLExample]:
    if k <= 0:
        raise ValueError("k must be positive.")
    chosen = sample_demonstration_indices(
        X_train,
        y_train,
        task_type=task_type,
        k=k,
        seed=seed,
        class_balance=class_balance,
    )
    return [ICLExample(features=X_train.iloc[idx], target=y_train.iloc[idx]) for idx in chosen]


def sample_demonstration_indices(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    task_type: TaskType,
    k: int,
    seed: int = 42,
    class_balance: bool = True,
) -> list[int]:
    if k <= 0:
        raise ValueError("k must be positive.")
    rng = random.Random(seed)
    if task_type == "classification" and class_balance:
        chosen = _sample_balanced_indices(y_train, k=k, rng=rng)
    else:
        population = list(range(len(X_train)))
        chosen = rng.sample(population, k=min(k, len(population)))
    return chosen


def _sample_balanced_indices(y_train: pd.Series, *, k: int, rng: random.Random) -> list[int]:
    by_class: dict[Any, list[int]] = {}
    for idx, value in enumerate(y_train.tolist()):
        by_class.setdefault(value, []).append(idx)

    if not by_class:
        return []

    selected: list[int] = []
    class_items = list(by_class.items())
    for _, indices in class_items:
        if len(selected) >= k:
            break
        selected.append(rng.choice(indices))

    remaining_pool = [idx for _, indices in class_items for idx in indices if idx not in selected]
    rng.shuffle(remaining_pool)
    for idx in remaining_pool:
        if len(selected) >= k:
            break
        selected.append(idx)
    return selected[:k]


def parse_classification_prediction(text: str, label_space: list[Any] | None = None) -> Any:
    stripped = text.strip()
    if not stripped:
        return "__EMPTY_RESPONSE__"
    if not label_space:
        return stripped.splitlines()[0].strip()
    first_line = stripped.splitlines()[0].strip()
    for label in label_space:
        if stripped == str(label):
            return label
        if first_line == str(label):
            return label
    lowered = stripped.lower()
    first_line_lowered = first_line.lower()
    for label in label_space:
        label_lower = str(label).lower()
        if lowered == label_lower or first_line_lowered == label_lower:
            return label

    # Fallback matching prefers longer labels first, which avoids cases like
    # matching "1" before "10" or "good" before "not_good".
    sorted_labels = sorted(label_space, key=lambda value: len(str(value)), reverse=True)
    for label in sorted_labels:
        label_text = str(label)
        label_lower = label_text.lower()
        if re.fullmatch(r"[A-Za-z0-9_]+", label_text):
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(label_lower)}(?![A-Za-z0-9_])"
            if re.search(pattern, lowered):
                return label
            continue
        if label_lower in lowered:
            return label
    return stripped.splitlines()[0].strip()


def parse_regression_prediction(text: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    if match is None:
        raise ValueError(f"Could not parse numeric prediction from: {text!r}")
    return float(match.group(0))


def predict_single(
    backend: ChatBackend,
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    query: pd.Series,
    target_name: str,
    task_type: TaskType,
    k: int = 16,
    model: str | None = None,
    system_prompt: str | None = None,
    cache: LLMCache | None = None,
    seed: int = 42,
    thinking_budget: int | None = None,
) -> str:
    runner = VanillaICLRunner(backend, cache=cache, seed=seed)
    return runner.predict_one(
        X_train=X_train,
        y_train=y_train,
        query=query,
        target_name=target_name,
        task_type=task_type,
        k=k,
        model=model,
        system_prompt=system_prompt,
        thinking_budget=thinking_budget,
    )


class VanillaICLRunner:
    def __init__(self, backend: ChatBackend, *, cache: LLMCache | None = None, seed: int = 42) -> None:
        self.backend = backend
        self.cache = cache or LLMCache()
        self.seed = seed

    def sample_examples(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        task_type: TaskType,
        k: int,
    ) -> list[ICLExample]:
        return sample_demonstrations(
            X_train,
            y_train,
            task_type=task_type,
            k=k,
            seed=self.seed,
        )

    def sample_example_indices(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        task_type: TaskType,
        k: int,
    ) -> list[int]:
        return sample_demonstration_indices(
            X_train,
            y_train,
            task_type=task_type,
            k=k,
            seed=self.seed,
        )

    def predict_one(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        query: pd.Series,
        target_name: str,
        task_type: TaskType,
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
        task_type: TaskType,
        label_space: list[Any] | None = None,
        k: int = 16,
        model: str | None = None,
        system_prompt: str | None = None,
        thinking_budget: int | None = None,
    ) -> dict[str, Any]:
        chosen_indices = self.sample_example_indices(X_train, y_train, task_type=task_type, k=k)
        examples = [ICLExample(features=X_train.iloc[idx], target=y_train.iloc[idx]) for idx in chosen_indices]
        prompt = build_prompt(
            examples=examples,
            query=query,
            target_name=target_name,
            task_type=task_type,
            label_space=label_space,
        )
        payload = {
            "prompt": prompt,
            "model": model,
            "system_prompt": system_prompt,
            "k": k,
            "task_type": task_type,
            "target_name": target_name,
        }
        cached = self.cache.get(payload)
        if cached is not None:
            return {
                "response": cached,
                "prompt": prompt,
                "indices": chosen_indices,
                "selected_targets": [y_train.iloc[idx] for idx in chosen_indices],
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
            "selected_targets": [y_train.iloc[idx] for idx in chosen_indices],
        }

    @staticmethod
    def parse_prediction(text: str, *, task_type: TaskType, label_space: list[Any] | None = None) -> Any:
        if task_type == "regression":
            return parse_regression_prediction(text)
        if not label_space:
            raise ValueError("label_space is required for classification.")
        return parse_classification_prediction(text, label_space=label_space)

    def run(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series | None,
        target_name: str,
        task_type: TaskType,
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
                prediction = self.parse_prediction(
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
                "method": "vanilla_icl",
                "prediction": prediction,
                "raw_output": response,
                "selected_indices": trace["indices"],
                "selected_targets": trace["selected_targets"],
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
                    "method": "vanilla_icl",
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
            "model_name": "vanilla_icl",
            "task_type": task_type,
            "sample_count": len(predictions),
            "raw_outputs": raw_outputs,
            "y_pred": predictions,
            "parse_failures": parse_failures,
            "traces": traces,
        }
        if y_eval is not None:
            result["metrics"] = compute_metrics(y_eval, predictions, task_type)
        return result


def run_vanilla_icl(
    backend: ChatBackend,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series | None,
    *,
    target_name: str,
    task_type: TaskType,
    k: int = 16,
    seed: int = 42,
    model: str | None = None,
    max_examples: int | None = None,
    thinking_budget: int | None = None,
    sleep_seconds: float = 0.0,
    progress_log_path: str | Path | None = None,
    progress_every: int = 25,
    progress_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runner = VanillaICLRunner(backend, seed=seed)
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
