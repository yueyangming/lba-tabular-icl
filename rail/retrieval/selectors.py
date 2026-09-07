from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd


SelectionStrategy = Literal["topk", "balanced", "diverse", "class_balanced_diversity", "contrastive", "difficulty", "oof_difficulty", "committee", "boundary", "uncertainty_only_boundary", "adaptive_boundary", "counter_boundary", "paired_boundary", "matched_pair_boundary", "knn_pair_boundary", "bidirectional_pair_boundary", "skeptical_pair_boundary", "margin_pair_rerank", "density_margin_pair_rerank", "relevance_constrained_density_rerank", "imbalance_pair_rerank", "query_difficulty_gated_rerank"]


def select_indices(
    candidate_indices: list[int],
    y_train: pd.Series,
    *,
    k: int,
    strategy: SelectionStrategy,
    label_space: list[Any] | None = None,
    candidate_matrix=None,
    query_vector=None,
    uncertainty_scores=None,
    margin_scores=None,
    class_priors: dict[Any, float] | None = None,
    priority_labels: list[Any] | None = None,
    priority_weights: dict[Any, float] | None = None,
    query_margin: float | None = None,
) -> list[int]:
    # `difficulty`, `oof_difficulty`, and `committee` intentionally share the
    # same selection routine. Their behavioral difference comes from the score
    # source passed in from `retrieval_icl.py`, not from different routing here.
    if strategy == "topk":
        return candidate_indices[:k]
    if strategy == "balanced":
        return select_balanced_indices(candidate_indices, y_train, k=k, label_space=label_space)
    if strategy == "diverse":
        return select_diverse_indices(
            candidate_indices,
            y_train,
            k=k,
            label_space=label_space,
            candidate_matrix=candidate_matrix,
            query_vector=query_vector,
        )
    if strategy == "class_balanced_diversity":
        return select_class_balanced_diversity_indices(
            candidate_indices,
            y_train,
            k=k,
            label_space=label_space,
            candidate_matrix=candidate_matrix,
            query_vector=query_vector,
        )
    if strategy == "contrastive":
        return select_contrastive_indices(
            candidate_indices,
            y_train,
            k=k,
            label_space=label_space,
            candidate_matrix=candidate_matrix,
            query_vector=query_vector,
        )
    if strategy == "difficulty":
        return select_difficulty_indices(
            candidate_indices,
            y_train,
            k=k,
            label_space=label_space,
            candidate_matrix=candidate_matrix,
            query_vector=query_vector,
            uncertainty_scores=uncertainty_scores,
        )
    if strategy == "oof_difficulty":
        return select_difficulty_indices(
            candidate_indices,
            y_train,
            k=k,
            label_space=label_space,
            candidate_matrix=candidate_matrix,
            query_vector=query_vector,
            uncertainty_scores=uncertainty_scores,
        )
    if strategy == "committee":
        return select_difficulty_indices(
            candidate_indices,
            y_train,
            k=k,
            label_space=label_space,
            candidate_matrix=candidate_matrix,
            query_vector=query_vector,
            uncertainty_scores=uncertainty_scores,
        )
    if strategy == "boundary":
        return select_boundary_indices(
            candidate_indices,
            y_train,
            k=k,
            label_space=label_space,
            candidate_matrix=candidate_matrix,
            query_vector=query_vector,
            uncertainty_scores=uncertainty_scores,
            priority_labels=priority_labels,
            priority_weights=priority_weights,
        )
    if strategy == "uncertainty_only_boundary":
        return select_uncertainty_only_boundary_indices(
            candidate_indices,
            y_train,
            k=k,
            candidate_matrix=candidate_matrix,
            query_vector=query_vector,
            uncertainty_scores=uncertainty_scores,
            priority_labels=priority_labels,
        )
    if strategy == "adaptive_boundary":
        return select_boundary_indices(
            candidate_indices,
            y_train,
            k=k,
            label_space=label_space,
            candidate_matrix=candidate_matrix,
            query_vector=query_vector,
            uncertainty_scores=uncertainty_scores,
            priority_labels=priority_labels,
            priority_weights=priority_weights,
            adaptive_quota=True,
        )
    if strategy == "counter_boundary":
        return select_boundary_indices(
            candidate_indices,
            y_train,
            k=k,
            label_space=label_space,
            candidate_matrix=candidate_matrix,
            query_vector=query_vector,
            uncertainty_scores=uncertainty_scores,
            priority_labels=priority_labels,
            priority_weights=priority_weights,
            adaptive_quota=True,
            counterexample_protection=True,
        )
    if strategy == "paired_boundary":
        return select_boundary_indices(
            candidate_indices,
            y_train,
            k=k,
            label_space=label_space,
            candidate_matrix=candidate_matrix,
            query_vector=query_vector,
            uncertainty_scores=uncertainty_scores,
            priority_labels=priority_labels,
            priority_weights=priority_weights,
            interleave_labels=True,
        )
    if strategy == "matched_pair_boundary":
        return select_matched_pair_boundary_indices(
            candidate_indices,
            y_train,
            k=k,
            candidate_matrix=candidate_matrix,
            query_vector=query_vector,
            uncertainty_scores=uncertainty_scores,
            priority_labels=priority_labels,
            query_margin=query_margin,
        )
    if strategy == "knn_pair_boundary":
        return select_knn_pair_boundary_indices(
            candidate_indices,
            y_train,
            k=k,
            candidate_matrix=candidate_matrix,
            query_vector=query_vector,
        )
    if strategy == "bidirectional_pair_boundary":
        return select_matched_pair_boundary_indices(
            candidate_indices,
            y_train,
            k=k,
            candidate_matrix=candidate_matrix,
            query_vector=query_vector,
            uncertainty_scores=uncertainty_scores,
            priority_labels=priority_labels,
            query_margin=query_margin,
            anchor_mode="bidirectional",
        )
    if strategy == "skeptical_pair_boundary":
        return select_matched_pair_boundary_indices(
            candidate_indices,
            y_train,
            k=k,
            candidate_matrix=candidate_matrix,
            query_vector=query_vector,
            uncertainty_scores=uncertainty_scores,
            priority_labels=priority_labels,
            query_margin=query_margin,
            anchor_mode="skeptical",
        )
    if strategy == "margin_pair_rerank":
        return select_matched_pair_boundary_indices(
            candidate_indices,
            y_train,
            k=k,
            candidate_matrix=candidate_matrix,
            query_vector=query_vector,
            uncertainty_scores=uncertainty_scores,
            margin_scores=margin_scores,
            priority_labels=priority_labels,
            query_margin=query_margin,
            anchor_mode="margin_rerank",
        )
    if strategy == "density_margin_pair_rerank":
        return select_matched_pair_boundary_indices(
            candidate_indices,
            y_train,
            k=k,
            candidate_matrix=candidate_matrix,
            query_vector=query_vector,
            uncertainty_scores=uncertainty_scores,
            margin_scores=margin_scores,
            priority_labels=priority_labels,
            query_margin=query_margin,
            anchor_mode="density_margin_rerank",
        )
    if strategy == "relevance_constrained_density_rerank":
        return select_matched_pair_boundary_indices(
            candidate_indices,
            y_train,
            k=k,
            candidate_matrix=candidate_matrix,
            query_vector=query_vector,
            uncertainty_scores=uncertainty_scores,
            margin_scores=margin_scores,
            priority_labels=priority_labels,
            query_margin=query_margin,
            anchor_mode="relevance_constrained_density_rerank",
        )
    if strategy == "imbalance_pair_rerank":
        return select_matched_pair_boundary_indices(
            candidate_indices,
            y_train,
            k=k,
            candidate_matrix=candidate_matrix,
            query_vector=query_vector,
            uncertainty_scores=uncertainty_scores,
            margin_scores=margin_scores,
            class_priors=class_priors,
            priority_labels=priority_labels,
            query_margin=query_margin,
            anchor_mode="imbalance_rerank",
        )
    if strategy == "query_difficulty_gated_rerank":
        raise ValueError("query_difficulty_gated_rerank must be resolved before select_indices.")
    raise ValueError(f"Unsupported selection strategy: {strategy}")


def select_balanced_indices(
    candidate_indices: list[int],
    y_train: pd.Series,
    *,
    k: int,
    label_space: list[Any] | None,
) -> list[int]:
    if not label_space:
        return candidate_indices[:k]

    selected: list[int] = []
    selected_set: set[int] = set()
    seen_labels: set[Any] = set()
    for idx in candidate_indices:
        label = y_train.iloc[idx]
        if label not in seen_labels:
            selected.append(idx)
            selected_set.add(idx)
            seen_labels.add(label)
        if len(selected) >= min(k, len(label_space)):
            break

    for idx in candidate_indices:
        if len(selected) >= k:
            break
        if idx not in selected_set:
            selected.append(idx)
            selected_set.add(idx)
    return selected[:k]


def select_diverse_indices(
    candidate_indices: list[int],
    y_train: pd.Series,
    *,
    k: int,
    label_space: list[Any] | None,
    candidate_matrix,
    query_vector,
) -> list[int]:
    if candidate_matrix is None or query_vector is None:
        return candidate_indices[:k]
    if len(candidate_indices) <= k:
        return candidate_indices

    candidate_sim = cosine_similarity(candidate_matrix, query_vector).reshape(-1)
    pairwise_sim = cosine_similarity(candidate_matrix)

    selected_positions: list[int] = []
    selected_position_set: set[int] = set()

    if label_space:
        seen_labels: set[Any] = set()
        ranked_positions = np.argsort(-candidate_sim)
        for pos in ranked_positions:
            label = y_train.iloc[candidate_indices[int(pos)]]
            if label not in seen_labels:
                selected_positions.append(int(pos))
                selected_position_set.add(int(pos))
                seen_labels.add(label)
            if len(selected_positions) >= min(k, len(label_space)):
                break

    if not selected_positions:
        first_pos = int(np.argmax(candidate_sim))
        selected_positions.append(first_pos)
        selected_position_set.add(first_pos)

    lambda_relevance = 0.7
    while len(selected_positions) < k and len(selected_positions) < len(candidate_indices):
        best_pos = None
        best_score = None
        for pos in range(len(candidate_indices)):
            if pos in selected_position_set:
                continue
            max_selected_sim = max(pairwise_sim[pos, chosen] for chosen in selected_positions)
            score = lambda_relevance * candidate_sim[pos] - (1 - lambda_relevance) * max_selected_sim
            if best_score is None or score > best_score:
                best_score = score
                best_pos = pos
        if best_pos is None:
            break
        selected_positions.append(best_pos)
        selected_position_set.add(best_pos)

    return [candidate_indices[pos] for pos in selected_positions[:k]]


def select_class_balanced_diversity_indices(
    candidate_indices: list[int],
    y_train: pd.Series,
    *,
    k: int,
    label_space: list[Any] | None,
    candidate_matrix,
    query_vector,
) -> list[int]:
    if candidate_matrix is None or query_vector is None:
        return select_balanced_indices(candidate_indices, y_train, k=k, label_space=label_space)
    if len(candidate_indices) <= k:
        return candidate_indices

    labels = label_space or sorted({y_train.iloc[idx] for idx in candidate_indices}, key=lambda value: str(value))
    if not labels:
        return candidate_indices[:k]

    candidate_sim = cosine_similarity(candidate_matrix, query_vector).reshape(-1)
    pairwise_sim = cosine_similarity(candidate_matrix)
    ranked_by_label: dict[Any, list[int]] = {label: [] for label in labels}
    for pos in np.argsort(-candidate_sim):
        label = y_train.iloc[candidate_indices[int(pos)]]
        if label in ranked_by_label:
            ranked_by_label[label].append(int(pos))

    active_labels = [label for label in labels if ranked_by_label[label]]
    if not active_labels:
        return candidate_indices[:k]
    base_quota = max(1, k // len(active_labels))
    quotas = {label: base_quota for label in active_labels}
    remainder = max(0, k - base_quota * len(active_labels))
    for label in active_labels:
        if remainder <= 0:
            break
        quotas[label] += 1
        remainder -= 1

    selected_positions: list[int] = []
    selected_set: set[int] = set()
    lambda_relevance = 0.55
    for label in active_labels:
        label_positions = ranked_by_label[label]
        label_selected: list[int] = []
        if label_positions:
            first = label_positions[0]
            label_selected.append(first)
            selected_positions.append(first)
            selected_set.add(first)
        while len(label_selected) < quotas[label] and len(selected_positions) < k:
            best_pos = None
            best_score = None
            for pos in label_positions:
                if pos in selected_set:
                    continue
                max_selected_sim = max(pairwise_sim[pos, chosen] for chosen in label_selected)
                score = lambda_relevance * candidate_sim[pos] - (1.0 - lambda_relevance) * max_selected_sim
                if best_score is None or score > best_score:
                    best_score = score
                    best_pos = pos
            if best_pos is None:
                break
            label_selected.append(best_pos)
            selected_positions.append(best_pos)
            selected_set.add(best_pos)

    if len(selected_positions) < k:
        for pos in np.argsort(-candidate_sim):
            pos = int(pos)
            if pos in selected_set:
                continue
            selected_positions.append(pos)
            selected_set.add(pos)
            if len(selected_positions) >= k:
                break

    return [candidate_indices[pos] for pos in selected_positions[:k]]


def select_contrastive_indices(
    candidate_indices: list[int],
    y_train: pd.Series,
    *,
    k: int,
    label_space: list[Any] | None,
    candidate_matrix,
    query_vector,
) -> list[int]:
    if not label_space:
        return candidate_indices[:k]
    if candidate_matrix is None or query_vector is None:
        return select_balanced_indices(candidate_indices, y_train, k=k, label_space=label_space)

    candidate_sim = cosine_similarity(candidate_matrix, query_vector).reshape(-1)
    ranked_positions = np.argsort(-candidate_sim)

    per_label_positions: dict[Any, list[int]] = {label: [] for label in label_space}
    for pos in ranked_positions:
        label = y_train.iloc[candidate_indices[int(pos)]]
        if label in per_label_positions:
            per_label_positions[label].append(int(pos))

    # Enforce a near-equal per-label quota, so the context contains both support
    # and counterexamples instead of being overwhelmed by one local mode.
    num_labels = len(label_space)
    base_quota = max(1, k // max(num_labels, 1))
    remainder = max(0, k - base_quota * num_labels)

    quotas = {label: base_quota for label in label_space}
    for label in label_space:
        if remainder <= 0:
            break
        quotas[label] += 1
        remainder -= 1

    selected_positions: list[int] = []
    for label in label_space:
        positions = per_label_positions.get(label, [])
        selected_positions.extend(positions[: quotas[label]])

    if len(selected_positions) < k:
        seen = set(selected_positions)
        for pos in ranked_positions:
            pos = int(pos)
            if pos in seen:
                continue
            selected_positions.append(pos)
            seen.add(pos)
            if len(selected_positions) >= k:
                break

    return [candidate_indices[pos] for pos in selected_positions[:k]]


def select_difficulty_indices(
    candidate_indices: list[int],
    y_train: pd.Series,
    *,
    k: int,
    label_space: list[Any] | None,
    candidate_matrix,
    query_vector,
    uncertainty_scores,
) -> list[int]:
    if uncertainty_scores is None:
        return candidate_indices[:k]
    if candidate_matrix is None or query_vector is None:
        return candidate_indices[:k]

    candidate_sim = cosine_similarity(candidate_matrix, query_vector).reshape(-1)
    sim_min = float(candidate_sim.min())
    sim_max = float(candidate_sim.max())
    if sim_max > sim_min:
        sim_norm = (candidate_sim - sim_min) / (sim_max - sim_min)
    else:
        sim_norm = np.ones_like(candidate_sim)

    candidate_uncertainty = np.asarray([float(uncertainty_scores[idx]) for idx in candidate_indices])
    unc_min = float(candidate_uncertainty.min())
    unc_max = float(candidate_uncertainty.max())
    if unc_max > unc_min:
        unc_norm = (candidate_uncertainty - unc_min) / (unc_max - unc_min)
    else:
        unc_norm = np.zeros_like(candidate_uncertainty)

    combined_score = 0.65 * sim_norm + 0.35 * unc_norm

    if not label_space:
        ranked_positions = np.argsort(-combined_score)
        return [candidate_indices[int(pos)] for pos in ranked_positions[:k]]

    per_label_positions: dict[Any, list[int]] = {label: [] for label in label_space}
    ranked_positions = np.argsort(-combined_score)
    for pos in ranked_positions:
        label = y_train.iloc[candidate_indices[int(pos)]]
        if label in per_label_positions:
            per_label_positions[label].append(int(pos))

    num_labels = len(label_space)
    base_quota = max(1, k // max(num_labels, 1))
    remainder = max(0, k - base_quota * num_labels)
    quotas = {label: base_quota for label in label_space}
    for label in label_space:
        if remainder <= 0:
            break
        quotas[label] += 1
        remainder -= 1

    selected_positions: list[int] = []
    for label in label_space:
        selected_positions.extend(per_label_positions[label][: quotas[label]])

    if len(selected_positions) < k:
        seen = set(selected_positions)
        for pos in ranked_positions:
            pos = int(pos)
            if pos in seen:
                continue
            selected_positions.append(pos)
            seen.add(pos)
            if len(selected_positions) >= k:
                break

    return [candidate_indices[pos] for pos in selected_positions[:k]]


def select_uncertainty_only_boundary_indices(
    candidate_indices: list[int],
    y_train: pd.Series,
    *,
    k: int,
    candidate_matrix,
    query_vector,
    uncertainty_scores,
    priority_labels: list[Any] | None,
) -> list[int]:
    if uncertainty_scores is None or candidate_matrix is None or query_vector is None:
        return candidate_indices[:k]

    candidate_sim = cosine_similarity(candidate_matrix, query_vector).reshape(-1)
    sim_min = float(candidate_sim.min())
    sim_max = float(candidate_sim.max())
    sim_norm = (candidate_sim - sim_min) / (sim_max - sim_min) if sim_max > sim_min else np.ones_like(candidate_sim)

    candidate_uncertainty = np.asarray([float(uncertainty_scores[idx]) for idx in candidate_indices])
    unc_min = float(candidate_uncertainty.min())
    unc_max = float(candidate_uncertainty.max())
    unc_norm = (candidate_uncertainty - unc_min) / (unc_max - unc_min) if unc_max > unc_min else np.zeros_like(candidate_uncertainty)

    allowed_labels = set(priority_labels[:2]) if priority_labels and len(priority_labels) >= 2 else None
    ranked_positions = np.argsort(-(0.25 * sim_norm + 0.75 * unc_norm))
    selected_positions: list[int] = []
    selected_set: set[int] = set()
    if allowed_labels:
        for pos in ranked_positions:
            pos = int(pos)
            if y_train.iloc[candidate_indices[pos]] not in allowed_labels:
                continue
            selected_positions.append(pos)
            selected_set.add(pos)
            if len(selected_positions) >= k:
                break

    if len(selected_positions) < k:
        for pos in ranked_positions:
            pos = int(pos)
            if pos in selected_set:
                continue
            selected_positions.append(pos)
            selected_set.add(pos)
            if len(selected_positions) >= k:
                break

    return [candidate_indices[pos] for pos in selected_positions[:k]]


def select_boundary_indices(
    candidate_indices: list[int],
    y_train: pd.Series,
    *,
    k: int,
    label_space: list[Any] | None,
    candidate_matrix,
    query_vector,
    uncertainty_scores,
    priority_labels: list[Any] | None,
    priority_weights: dict[Any, float] | None,
    adaptive_quota: bool = False,
    counterexample_protection: bool = False,
    interleave_labels: bool = False,
) -> list[int]:
    if uncertainty_scores is None or candidate_matrix is None or query_vector is None:
        return candidate_indices[:k]
    if not priority_labels:
        return select_difficulty_indices(
            candidate_indices,
            y_train,
            k=k,
            label_space=label_space,
            candidate_matrix=candidate_matrix,
            query_vector=query_vector,
            uncertainty_scores=uncertainty_scores,
        )

    candidate_sim = cosine_similarity(candidate_matrix, query_vector).reshape(-1)
    sim_min = float(candidate_sim.min())
    sim_max = float(candidate_sim.max())
    sim_norm = (candidate_sim - sim_min) / (sim_max - sim_min) if sim_max > sim_min else np.ones_like(candidate_sim)

    candidate_uncertainty = np.asarray([float(uncertainty_scores[idx]) for idx in candidate_indices])
    unc_min = float(candidate_uncertainty.min())
    unc_max = float(candidate_uncertainty.max())
    unc_norm = (candidate_uncertainty - unc_min) / (unc_max - unc_min) if unc_max > unc_min else np.zeros_like(candidate_uncertainty)

    base_score = 0.55 * sim_norm + 0.45 * unc_norm
    priority_weights = priority_weights or {}
    weighted_score = np.copy(base_score)
    for pos, idx in enumerate(candidate_indices):
        label = y_train.iloc[idx]
        if label in priority_labels:
            weighted_score[pos] *= priority_weights.get(label, 1.0)
        else:
            weighted_score[pos] *= 0.2
        if counterexample_protection and len(priority_labels) >= 2:
            if label == priority_labels[0]:
                weighted_score[pos] *= 0.9
            elif label == priority_labels[1]:
                weighted_score[pos] *= 1.15

    ranked_positions = np.argsort(-weighted_score)
    per_label_positions: dict[Any, list[int]] = {label: [] for label in priority_labels}
    for pos in ranked_positions:
        label = y_train.iloc[candidate_indices[int(pos)]]
        if label in per_label_positions:
            per_label_positions[label].append(int(pos))

    quotas: dict[Any, int] = {}
    if len(priority_labels) == 1:
        quotas[priority_labels[0]] = k
    elif adaptive_quota and priority_weights:
        clipped_weights = {}
        for label in priority_labels[:2]:
            weight = float(priority_weights.get(label, 1.0))
            clipped_weights[label] = min(max(weight, 0.25), 0.75)
        total = sum(clipped_weights.values()) or 1.0
        first = priority_labels[0]
        second = priority_labels[1]
        quotas[first] = max(1, int(round(k * clipped_weights[first] / total)))
        quotas[second] = max(1, k - quotas[first])
        if counterexample_protection and len(priority_labels) >= 2:
            max_first = int(np.floor(0.45 * k))
            if quotas[first] > max_first:
                quotas[first] = max(1, max_first)
                quotas[second] = max(1, k - quotas[first])
        for label in priority_labels[2:]:
            quotas[label] = 0
    else:
        quotas[priority_labels[0]] = (k + 1) // 2
        remaining = k - quotas[priority_labels[0]]
        quotas[priority_labels[1]] = remaining
        for label in priority_labels[2:]:
            quotas[label] = 0

    selected_positions: list[int] = []
    if interleave_labels and len(priority_labels) >= 2:
        per_label_cursors = {label: 0 for label in priority_labels}
        while len(selected_positions) < k:
            progressed = False
            for label in priority_labels:
                cursor = per_label_cursors[label]
                allowed = quotas.get(label, 0)
                if cursor >= allowed:
                    continue
                label_positions = per_label_positions.get(label, [])
                if cursor >= len(label_positions):
                    continue
                selected_positions.append(label_positions[cursor])
                per_label_cursors[label] += 1
                progressed = True
                if len(selected_positions) >= k:
                    break
            if not progressed:
                break
    else:
        for label in priority_labels:
            selected_positions.extend(per_label_positions.get(label, [])[: quotas.get(label, 0)])

    if len(selected_positions) < k:
        seen = set(selected_positions)
        for pos in ranked_positions:
            pos = int(pos)
            if pos in seen:
                continue
            selected_positions.append(pos)
            seen.add(pos)
            if len(selected_positions) >= k:
                break

    return [candidate_indices[pos] for pos in selected_positions[:k]]


def select_matched_pair_boundary_indices(
    candidate_indices: list[int],
    y_train: pd.Series,
    *,
    k: int,
    candidate_matrix,
    query_vector,
    uncertainty_scores,
    margin_scores=None,
    class_priors: dict[Any, float] | None = None,
    priority_labels: list[Any] | None,
    query_margin: float | None,
    anchor_mode: str = "support",
) -> list[int]:
    if uncertainty_scores is None or candidate_matrix is None or query_vector is None:
        return candidate_indices[:k]
    if not priority_labels or len(priority_labels) < 2:
        return candidate_indices[:k]

    support_label = priority_labels[0]
    counter_label = priority_labels[1]

    candidate_sim = cosine_similarity(candidate_matrix, query_vector).reshape(-1)
    sim_min = float(candidate_sim.min())
    sim_max = float(candidate_sim.max())
    sim_norm = (candidate_sim - sim_min) / (sim_max - sim_min) if sim_max > sim_min else np.ones_like(candidate_sim)

    candidate_uncertainty = np.asarray([float(uncertainty_scores[idx]) for idx in candidate_indices])
    unc_min = float(candidate_uncertainty.min())
    unc_max = float(candidate_uncertainty.max())
    unc_norm = (candidate_uncertainty - unc_min) / (unc_max - unc_min) if unc_max > unc_min else np.zeros_like(candidate_uncertainty)

    if margin_scores is not None:
        candidate_margin = np.asarray([float(margin_scores[idx]) for idx in candidate_indices])
        margin_min = float(candidate_margin.min())
        margin_max = float(candidate_margin.max())
        margin_norm = (candidate_margin - margin_min) / (margin_max - margin_min) if margin_max > margin_min else np.zeros_like(candidate_margin)
    else:
        margin_norm = np.zeros_like(sim_norm)

    pairwise_density = None
    if anchor_mode in {"density_margin_rerank", "relevance_constrained_density_rerank", "imbalance_rerank"}:
        pairwise_sim = cosine_similarity(candidate_matrix)
        density_values = []
        for pos in range(len(candidate_indices)):
            row = np.asarray(pairwise_sim[pos]).reshape(-1)
            top_local = np.sort(row)[-6:-1]
            density_values.append(float(np.mean(top_local)) if len(top_local) > 0 else 0.0)
        density_values = np.asarray(density_values)
        density_min = float(density_values.min())
        density_max = float(density_values.max())
        pairwise_density = (density_values - density_min) / (density_max - density_min) if density_max > density_min else np.ones_like(density_values)

    support_positions = [pos for pos, idx in enumerate(candidate_indices) if y_train.iloc[idx] == support_label]
    counter_positions = [pos for pos, idx in enumerate(candidate_indices) if y_train.iloc[idx] == counter_label]
    if not support_positions or not counter_positions:
        return candidate_indices[:k]

    support_score = {pos: 0.70 * sim_norm[pos] + 0.30 * unc_norm[pos] for pos in support_positions}
    counter_score = {pos: 0.70 * sim_norm[pos] + 0.30 * unc_norm[pos] for pos in counter_positions}
    support_base = {pos: 0.55 * sim_norm[pos] + 0.45 * unc_norm[pos] for pos in support_positions}
    counter_base = {pos: 0.55 * sim_norm[pos] + 0.45 * unc_norm[pos] for pos in counter_positions}

    # These rerank modes differ in how anchor/base scores are built above.
    # The downstream pair-strength combiner intentionally stays fixed so the
    # experiment isolates the effect of anchor/base scoring rather than mixing
    # in another changing pair-level formula.
    if anchor_mode == "margin_rerank":
        support_score = {pos: 0.45 * sim_norm[pos] + 0.20 * unc_norm[pos] + 0.35 * margin_norm[pos] for pos in support_positions}
        counter_score = {pos: 0.45 * sim_norm[pos] + 0.20 * unc_norm[pos] + 0.35 * margin_norm[pos] for pos in counter_positions}
        support_base = {pos: 0.40 * sim_norm[pos] + 0.20 * unc_norm[pos] + 0.40 * margin_norm[pos] for pos in support_positions}
        counter_base = {pos: 0.40 * sim_norm[pos] + 0.20 * unc_norm[pos] + 0.40 * margin_norm[pos] for pos in counter_positions}
    elif anchor_mode == "density_margin_rerank" and pairwise_density is not None:
        support_score = {pos: 0.38 * sim_norm[pos] + 0.17 * unc_norm[pos] + 0.25 * margin_norm[pos] + 0.20 * pairwise_density[pos] for pos in support_positions}
        counter_score = {pos: 0.38 * sim_norm[pos] + 0.17 * unc_norm[pos] + 0.25 * margin_norm[pos] + 0.20 * pairwise_density[pos] for pos in counter_positions}
        support_base = {pos: 0.32 * sim_norm[pos] + 0.18 * unc_norm[pos] + 0.25 * margin_norm[pos] + 0.25 * pairwise_density[pos] for pos in support_positions}
        counter_base = {pos: 0.32 * sim_norm[pos] + 0.18 * unc_norm[pos] + 0.25 * margin_norm[pos] + 0.25 * pairwise_density[pos] for pos in counter_positions}
    elif anchor_mode == "relevance_constrained_density_rerank" and pairwise_density is not None:
        relevance_floor = float(np.quantile(sim_norm, 0.35))
        relevance_penalty = np.maximum(0.0, relevance_floor - sim_norm)
        support_score = {
            pos: 0.44 * sim_norm[pos] + 0.14 * unc_norm[pos] + 0.22 * margin_norm[pos] + 0.20 * pairwise_density[pos] - 0.18 * relevance_penalty[pos]
            for pos in support_positions
        }
        counter_score = {
            pos: 0.44 * sim_norm[pos] + 0.14 * unc_norm[pos] + 0.22 * margin_norm[pos] + 0.20 * pairwise_density[pos] - 0.18 * relevance_penalty[pos]
            for pos in counter_positions
        }
        support_base = {
            pos: 0.39 * sim_norm[pos] + 0.14 * unc_norm[pos] + 0.22 * margin_norm[pos] + 0.25 * pairwise_density[pos] - 0.20 * relevance_penalty[pos]
            for pos in support_positions
        }
        counter_base = {
            pos: 0.39 * sim_norm[pos] + 0.14 * unc_norm[pos] + 0.22 * margin_norm[pos] + 0.25 * pairwise_density[pos] - 0.20 * relevance_penalty[pos]
            for pos in counter_positions
        }
    elif anchor_mode == "imbalance_rerank" and pairwise_density is not None:
        support_prior = float(class_priors.get(support_label, 0.5)) if class_priors else 0.5
        counter_prior = float(class_priors.get(counter_label, 0.5)) if class_priors else 0.5
        minority_boost = 1.0
        if counter_prior < support_prior and query_margin is not None and query_margin >= 0.10:
            ratio = min((support_prior / max(counter_prior, 1e-6)) ** 0.5, 1.8)
            minority_boost = 1.0 + 0.20 * (ratio - 1.0)
        support_score = {pos: 0.36 * sim_norm[pos] + 0.16 * unc_norm[pos] + 0.23 * margin_norm[pos] + 0.25 * pairwise_density[pos] for pos in support_positions}
        counter_score = {pos: (0.33 * sim_norm[pos] + 0.16 * unc_norm[pos] + 0.26 * margin_norm[pos] + 0.25 * pairwise_density[pos]) * minority_boost for pos in counter_positions}
        support_base = {pos: 0.30 * sim_norm[pos] + 0.17 * unc_norm[pos] + 0.23 * margin_norm[pos] + 0.30 * pairwise_density[pos] for pos in support_positions}
        counter_base = {pos: (0.27 * sim_norm[pos] + 0.18 * unc_norm[pos] + 0.25 * margin_norm[pos] + 0.30 * pairwise_density[pos]) * minority_boost for pos in counter_positions}

    target_pairs = min(k // 2, len(support_positions), len(counter_positions))
    pairs: list[tuple[int, int, float]] = []

    if anchor_mode == "bidirectional":
        ranked_support = sorted(support_positions, key=lambda pos: support_score[pos], reverse=True)
        ranked_counter = sorted(counter_positions, key=lambda pos: counter_score[pos], reverse=True)
        remaining_support = set(support_positions)
        remaining_counter = set(counter_positions)
        support_cursor = 0
        counter_cursor = 0
        anchor_from_support = True

        while len(pairs) < target_pairs and remaining_support and remaining_counter:
            chosen_pair: tuple[int, int, float] | None = None

            if anchor_from_support:
                while support_cursor < len(ranked_support) and ranked_support[support_cursor] not in remaining_support:
                    support_cursor += 1
                if support_cursor < len(ranked_support):
                    support_pos = ranked_support[support_cursor]
                    chosen_pair = _match_cross_label_pair(
                        anchor_pos=support_pos,
                        anchor_score=support_score[support_pos],
                        candidate_positions=list(remaining_counter),
                        candidate_base=counter_base,
                        candidate_matrix=candidate_matrix,
                    )
                    if chosen_pair is not None:
                        remaining_support.remove(support_pos)
                        remaining_counter.remove(chosen_pair[1])
                    support_cursor += 1
            else:
                while counter_cursor < len(ranked_counter) and ranked_counter[counter_cursor] not in remaining_counter:
                    counter_cursor += 1
                if counter_cursor < len(ranked_counter):
                    counter_pos = ranked_counter[counter_cursor]
                    reverse_pair = _match_cross_label_pair(
                        anchor_pos=counter_pos,
                        anchor_score=counter_score[counter_pos],
                        candidate_positions=list(remaining_support),
                        candidate_base=support_base,
                        candidate_matrix=candidate_matrix,
                    )
                    if reverse_pair is not None:
                        support_pos = reverse_pair[1]
                        chosen_pair = (support_pos, counter_pos, reverse_pair[2])
                        remaining_counter.remove(counter_pos)
                        remaining_support.remove(support_pos)
                    counter_cursor += 1

            anchor_from_support = not anchor_from_support
            if chosen_pair is not None:
                pairs.append(chosen_pair)
    elif anchor_mode == "skeptical":
        majority_label = y_train.value_counts().idxmax()
        anchor_from_counter = support_label == majority_label and query_margin is not None and query_margin >= 0.15
        if anchor_from_counter:
            ranked_counter = sorted(counter_positions, key=lambda pos: counter_score[pos], reverse=True)
            remaining_support = set(support_positions)
            for counter_pos in ranked_counter:
                if len(pairs) >= target_pairs or not remaining_support:
                    break
                reverse_pair = _match_cross_label_pair(
                    anchor_pos=counter_pos,
                    anchor_score=counter_score[counter_pos],
                    candidate_positions=list(remaining_support),
                    candidate_base=support_base,
                    candidate_matrix=candidate_matrix,
                )
                if reverse_pair is None:
                    continue
                support_pos = reverse_pair[1]
                pairs.append((support_pos, counter_pos, reverse_pair[2]))
                remaining_support.remove(support_pos)
        else:
            ranked_support = sorted(support_positions, key=lambda pos: support_score[pos], reverse=True)
            remaining_counter = set(counter_positions)
            for support_pos in ranked_support:
                if len(pairs) >= target_pairs or not remaining_counter:
                    break
                chosen_pair = _match_cross_label_pair(
                    anchor_pos=support_pos,
                    anchor_score=support_score[support_pos],
                    candidate_positions=list(remaining_counter),
                    candidate_base=counter_base,
                    candidate_matrix=candidate_matrix,
                )
                if chosen_pair is None:
                    continue
                pairs.append(chosen_pair)
                remaining_counter.remove(chosen_pair[1])
    else:
        ranked_support = sorted(support_positions, key=lambda pos: support_score[pos], reverse=True)
        remaining_counter = set(counter_positions)
        for support_pos in ranked_support:
            if len(pairs) >= target_pairs or not remaining_counter:
                break
            chosen_pair = _match_cross_label_pair(
                anchor_pos=support_pos,
                anchor_score=support_score[support_pos],
                candidate_positions=list(remaining_counter),
                candidate_base=counter_base,
                candidate_matrix=candidate_matrix,
                pair_mode=anchor_mode,
                support_pos=support_pos,
            )
            if chosen_pair is None:
                continue
            pairs.append(chosen_pair)
            remaining_counter.remove(chosen_pair[1])

    pairs.sort(key=lambda item: item[2], reverse=True)
    counter_first = query_margin is not None and query_margin >= 0.15

    selected_positions: list[int] = []
    for support_pos, counter_pos, _ in pairs:
        if counter_first:
            selected_positions.extend([counter_pos, support_pos])
        else:
            selected_positions.extend([support_pos, counter_pos])

    if len(selected_positions) < k:
        ranked_all = np.argsort(-(0.60 * sim_norm + 0.40 * unc_norm))
        seen = set(selected_positions)
        for pos in ranked_all:
            pos = int(pos)
            if pos in seen:
                continue
            selected_positions.append(pos)
            seen.add(pos)
            if len(selected_positions) >= k:
                break

    return [candidate_indices[pos] for pos in selected_positions[:k]]


def select_knn_pair_boundary_indices(
    candidate_indices: list[int],
    y_train: pd.Series,
    *,
    k: int,
    candidate_matrix,
    query_vector,
) -> list[int]:
    if candidate_matrix is None or query_vector is None:
        return candidate_indices[:k]
    if len(candidate_indices) <= k:
        return candidate_indices

    support_label, counter_label = _infer_local_top2_labels(candidate_indices, y_train, candidate_matrix, query_vector)
    if support_label is None or counter_label is None:
        return candidate_indices[:k]

    candidate_sim = cosine_similarity(candidate_matrix, query_vector).reshape(-1)
    sim_min = float(candidate_sim.min())
    sim_max = float(candidate_sim.max())
    sim_norm = (candidate_sim - sim_min) / (sim_max - sim_min) if sim_max > sim_min else np.ones_like(candidate_sim)

    pairwise_sim = cosine_similarity(candidate_matrix)
    mixing_scores = _compute_local_mixing_scores(
        candidate_indices=candidate_indices,
        y_train=y_train,
        pairwise_sim=pairwise_sim,
        support_label=support_label,
        counter_label=counter_label,
    )

    support_positions = [pos for pos, idx in enumerate(candidate_indices) if y_train.iloc[idx] == support_label]
    counter_positions = [pos for pos, idx in enumerate(candidate_indices) if y_train.iloc[idx] == counter_label]
    if not support_positions or not counter_positions:
        return candidate_indices[:k]

    support_score = {
        pos: 0.70 * sim_norm[pos] + 0.30 * mixing_scores[pos]
        for pos in support_positions
    }
    ranked_support = sorted(support_positions, key=lambda pos: support_score[pos], reverse=True)

    target_pairs = min(k // 2, len(ranked_support), len(counter_positions))
    remaining_counter = set(counter_positions)
    pairs: list[tuple[int, int, float]] = []
    for support_pos in ranked_support:
        if len(pairs) >= target_pairs or not remaining_counter:
            break
        support_vec = pairwise_sim[support_pos]
        best_counter = None
        best_counter_score = None
        for counter_pos in remaining_counter:
            counter_score = 0.50 * sim_norm[counter_pos] + 0.50 * float(support_vec[counter_pos])
            if best_counter_score is None or counter_score > best_counter_score:
                best_counter = counter_pos
                best_counter_score = counter_score
        if best_counter is None or best_counter_score is None:
            continue
        pair_score = 0.50 * support_score[support_pos] + 0.50 * best_counter_score
        pairs.append((support_pos, best_counter, pair_score))
        remaining_counter.remove(best_counter)

    pairs.sort(key=lambda item: item[2], reverse=True)
    selected_positions: list[int] = []
    for support_pos, counter_pos, _ in pairs:
        selected_positions.extend([support_pos, counter_pos])

    if len(selected_positions) < k:
        seen = set(selected_positions)
        for pos in np.argsort(-sim_norm):
            pos = int(pos)
            if pos in seen:
                continue
            selected_positions.append(pos)
            seen.add(pos)
            if len(selected_positions) >= k:
                break

    return [candidate_indices[pos] for pos in selected_positions[:k]]


def _infer_local_top2_labels(
    candidate_indices: list[int],
    y_train: pd.Series,
    candidate_matrix,
    query_vector,
) -> tuple[Any | None, Any | None]:
    candidate_sim = cosine_similarity(candidate_matrix, query_vector).reshape(-1)
    label_counts: dict[Any, int] = {}
    label_best_sim: dict[Any, float] = {}
    for pos, idx in enumerate(candidate_indices):
        label = y_train.iloc[idx]
        label_counts[label] = label_counts.get(label, 0) + 1
        current = float(candidate_sim[pos])
        label_best_sim[label] = max(current, label_best_sim.get(label, -np.inf))

    if len(label_counts) < 2:
        return None, None

    ranked_labels = sorted(
        label_counts,
        key=lambda label: (-label_counts[label], -label_best_sim[label], str(label)),
    )
    return ranked_labels[0], ranked_labels[1]


def _compute_local_mixing_scores(
    *,
    candidate_indices: list[int],
    y_train: pd.Series,
    pairwise_sim: np.ndarray,
    support_label: Any,
    counter_label: Any,
) -> np.ndarray:
    num_candidates = len(candidate_indices)
    mixing_scores = np.zeros(num_candidates, dtype=float)
    neighborhood_size = min(16, num_candidates)
    if neighborhood_size <= 1:
        return mixing_scores

    candidate_labels = [y_train.iloc[idx] for idx in candidate_indices]
    for pos in range(num_candidates):
        row = np.asarray(pairwise_sim[pos]).reshape(-1)
        ranked_neighbors = np.argsort(-row)
        local_neighbors = [int(neighbor) for neighbor in ranked_neighbors if int(neighbor) != pos][: neighborhood_size - 1]
        if not local_neighbors:
            continue
        support_count = sum(candidate_labels[neighbor] == support_label for neighbor in local_neighbors)
        counter_count = sum(candidate_labels[neighbor] == counter_label for neighbor in local_neighbors)
        total = support_count + counter_count
        if total == 0:
            mixing_scores[pos] = 0.0
            continue
        support_share = support_count / total
        counter_share = counter_count / total
        mixing_scores[pos] = 1.0 - abs(support_share - counter_share)

    return mixing_scores


def _match_cross_label_pair(
    *,
    anchor_pos: int,
    anchor_score: float,
    candidate_positions: list[int],
    candidate_base: dict[int, float],
    candidate_matrix,
    pair_mode: str = "support",
    support_pos: int | None = None,
) -> tuple[int, int, float] | None:
    if not candidate_positions:
        return None
    anchor_vec = candidate_matrix[anchor_pos]
    if hasattr(anchor_vec, "ndim") and anchor_vec.ndim == 1:
        anchor_vec = anchor_vec.reshape(1, -1)
    local_similarity = cosine_similarity(candidate_matrix[candidate_positions], anchor_vec).reshape(-1)

    best_candidate_pos = None
    best_pair_score = None
    best_local_compactness = None
    for idx_in_list, candidate_pos in enumerate(candidate_positions):
        local_compactness = float(local_similarity[idx_in_list])
        pair_score = 0.50 * candidate_base[candidate_pos] + 0.50 * local_compactness
        if best_pair_score is None or pair_score > best_pair_score:
            best_candidate_pos = candidate_pos
            best_pair_score = pair_score
            best_local_compactness = local_compactness

    if best_candidate_pos is None or best_pair_score is None or best_local_compactness is None:
        return None
    # Keep a common pair-strength form across rerank variants. The variants are
    # already differentiated by `anchor_score` and `candidate_base`, so this
    # function does not introduce a second source of variation.
    if pair_mode == "margin_rerank":
        pair_strength = 0.45 * anchor_score + 0.25 * candidate_base[best_candidate_pos] + 0.30 * best_local_compactness
    elif pair_mode == "density_margin_rerank":
        pair_strength = 0.45 * anchor_score + 0.25 * candidate_base[best_candidate_pos] + 0.30 * best_local_compactness
    elif pair_mode == "relevance_constrained_density_rerank":
        pair_strength = 0.45 * anchor_score + 0.25 * candidate_base[best_candidate_pos] + 0.30 * best_local_compactness
    else:
        pair_strength = 0.60 * anchor_score + 0.40 * best_pair_score
    return anchor_pos, best_candidate_pos, pair_strength
