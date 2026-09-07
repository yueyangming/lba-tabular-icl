from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


def _normalize_value(value: Any, *, missing_value_token: str) -> str:
    if pd.isna(value):
        return missing_value_token
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def serialize_features(
    features: pd.Series | Mapping[str, Any],
    *,
    feature_order: Sequence[str] | None = None,
    missing_value_token: str = "N/A",
) -> str:
    if isinstance(features, pd.Series):
        items = features.to_dict()
    else:
        items = dict(features)

    ordered_keys = list(feature_order) if feature_order is not None else list(items.keys())
    pairs = [
        f"{key}: {_normalize_value(items.get(key), missing_value_token=missing_value_token)}"
        for key in ordered_keys
    ]
    return ", ".join(pairs)


def serialize_example(
    features: pd.Series | Mapping[str, Any],
    *,
    target: Any | None = None,
    target_name: str = "Label",
    feature_order: Sequence[str] | None = None,
    include_target: bool = True,
    missing_value_token: str = "N/A",
) -> str:
    body = serialize_features(
        features,
        feature_order=feature_order,
        missing_value_token=missing_value_token,
    )
    if include_target:
        target_text = _normalize_value(target, missing_value_token=missing_value_token)
        return f"{body}, {target_name}: {target_text}"
    return body
