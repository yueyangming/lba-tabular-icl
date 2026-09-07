from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal
from urllib.parse import quote
from urllib.request import urlopen

import openml
import pandas as pd


TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class TabularDataset:
    name: str
    dataset_id: int
    task_type: TaskType
    features: pd.DataFrame
    target: pd.Series
    target_name: str

    @property
    def categorical_columns(self) -> list[str]:
        return [
            col
            for col in self.features.columns
            if pd.api.types.is_object_dtype(self.features[col])
            or isinstance(self.features[col].dtype, pd.CategoricalDtype)
        ]

    @property
    def numeric_columns(self) -> list[str]:
        return [col for col in self.features.columns if pd.api.types.is_numeric_dtype(self.features[col])]


def _infer_task_type(target: pd.Series) -> TaskType:
    if pd.api.types.is_numeric_dtype(target):
        nunique = target.nunique(dropna=True)
        if nunique > 20:
            return "regression"
    return "classification"


def _resolve_dataset_id_by_name(name: str, version: int | str | None = "active") -> int:
    api_name = quote(name, safe="")
    url = f"https://www.openml.org/api/v1/json/data/list/data_name/{api_name}/limit/20"
    with urlopen(url, timeout=30) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))

    datasets = payload.get("data", {}).get("dataset", [])
    if not datasets:
        raise ValueError(f"OpenML dataset not found: {name}")

    candidates = [item for item in datasets if item.get("name") == name]
    if version != "active":
        candidates = [item for item in candidates if int(item.get("version", -1)) == int(version)]
        if not candidates:
            raise ValueError(f"OpenML dataset not found: {name} version={version}")

    if not candidates:
        raise ValueError(f"OpenML dataset not found: {name}")

    # Prefer the earliest active version to stay consistent with previous
    # name-based resolution that effectively selected the first matching row.
    candidates.sort(key=lambda item: (int(item.get("version", 0)), int(item.get("did", 10**12))))
    return int(candidates[0]["did"])


def load_openml_dataset(
    *,
    dataset_id: int | None = None,
    name: str | None = None,
    version: int | str | None = "active",
    target_name: str | None = None,
    task_type_override: TaskType | None = None,
) -> TabularDataset:
    if dataset_id is None and name is None:
        raise ValueError("Either dataset_id or name must be provided.")

    if dataset_id is not None:
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
    else:
        resolved_id = _resolve_dataset_id_by_name(name=name, version=version)
        dataset = openml.datasets.get_dataset(resolved_id, download_data=True)

    resolved_target = target_name or dataset.default_target_attribute
    if not resolved_target:
        raise ValueError(f"Dataset {dataset.name} has no default target; target_name is required.")

    X, y, _, _ = dataset.get_data(target=resolved_target, dataset_format="dataframe")
    y = pd.Series(y, name=resolved_target)
    task_type = task_type_override or _infer_task_type(y)

    return TabularDataset(
        name=dataset.name,
        dataset_id=int(dataset.dataset_id),
        task_type=task_type,
        features=X,
        target=y,
        target_name=resolved_target,
    )
