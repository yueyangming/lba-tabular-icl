"""Class-sensitive LEQ variant for imbalanced datasets (page-blocks).

Computes LEQ based on local macro-F1 improvement over global macro-F1,
addressing the limitation where accuracy-based LEQ misses minority-class rescue.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from rail.analysis.compute_leq import (
    CLASSIFICATION_DATASETS, _build_preprocessor,
)
from rail.data import load_openml_dataset, train_val_test_split
from sklearn.neighbors import NearestNeighbors


def compute_leq_class_sensitive(X_train, y_train, X_test, y_test, k=100):
    """Compute class-sensitive LEQ: local kNN macro-F1 minus global majority macro-F1."""
    preprocessor = _build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    nn = NearestNeighbors(n_neighbors=min(k, len(X_train) - 1), metric="euclidean")
    nn.fit(X_train_proc)
    _, indices = nn.kneighbors(X_test_proc)

    y_train_arr = np.array(y_train)
    y_test_arr = np.array(y_test)

    # kNN predictions (majority vote among neighbors)
    knn_preds = []
    for i in range(len(X_test)):
        neighbor_labels = y_train_arr[indices[i]]
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        knn_preds.append(unique[np.argmax(counts)])
    knn_preds = np.array(knn_preds)

    # Global majority baseline predictions
    unique_train, counts_train = np.unique(y_train_arr, return_counts=True)
    majority_class = unique_train[np.argmax(counts_train)]
    majority_preds = np.full(len(y_test), majority_class)

    # Compute macro-F1 for both
    knn_macro_f1 = f1_score(y_test_arr, knn_preds, average='macro', zero_division=0)
    majority_macro_f1 = f1_score(y_test_arr, majority_preds, average='macro', zero_division=0)

    # Also compute accuracy-based LEQ for comparison
    knn_acc = np.mean(knn_preds == y_test_arr)
    majority_acc = np.mean(majority_preds == y_test_arr)

    return {
        "leq_acc": knn_acc - majority_acc,
        "leq_macro_f1": knn_macro_f1 - majority_macro_f1,
        "knn_acc": knn_acc,
        "knn_macro_f1": knn_macro_f1,
        "majority_acc": majority_acc,
        "majority_macro_f1": majority_macro_f1,
    }


def run():
    gains_df = pd.read_csv(PROJECT_ROOT / "artifacts" / "results" / "lba_main.csv")
    gains = dict(zip(gains_df["dataset"], gains_df["retrieval_gain"]))

    print("=" * 70)
    print("Class-Sensitive LEQ Analysis")
    print("=" * 70)

    rows = []
    for ds in CLASSIFICATION_DATASETS:
        name = ds["name"]
        if name not in gains:
            continue
        try:
            d = load_openml_dataset(dataset_id=ds["dataset_id"])
            sp = train_val_test_split(d.features, d.target, task_type=str(d.task_type),
                                      test_size=0.2, val_size=0.2, random_state=42)
            r = compute_leq_class_sensitive(sp.X_train, sp.y_train, sp.X_test, sp.y_test, k=100)
            r["dataset"] = name
            r["retrieval_gain"] = gains[name]
            rows.append(r)
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    df = pd.DataFrame(rows)

    print(f"\n{'Dataset':15s} {'LEQ_acc':>8s} {'LEQ_F1':>8s} {'Gain':>8s}")
    print("-" * 45)
    for _, row in df.iterrows():
        print(f"{row['dataset']:15s} {row['leq_acc']:>8.3f} {row['leq_macro_f1']:>8.3f} {row['retrieval_gain']:>8.3f}")

    # Correlations
    leq_acc = df["leq_acc"].values
    leq_f1 = df["leq_macro_f1"].values
    gain = df["retrieval_gain"].values

    r_acc, p_acc = stats.spearmanr(leq_acc, gain)
    r_f1, p_f1 = stats.spearmanr(leq_f1, gain)

    print(f"\nSpearman correlations (n={len(df)}):")
    print(f"  LEQ_accuracy  vs gain: r={r_acc:.4f}, p={p_acc:.4f}")
    print(f"  LEQ_macro_F1  vs gain: r={r_f1:.4f}, p={p_f1:.4f}")

    # Save
    out = PROJECT_ROOT / "outputs" / "analysis" / "leq_class_sensitive.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    run()
