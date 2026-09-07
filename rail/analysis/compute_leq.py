"""Compute Local Evidence Quality (LEQ) metrics for canonical datasets.

LEQ measures how informative a query's local neighborhood is.
The hypothesis: LEQ positively correlates with retrieval gain.

LEQ_A variants:
  - Classification: neighborhood label purity (fraction of same-class neighbors)
  - Regression: neighborhood target variance compression (1 - local_var / global_var)

LEQ_B variants (if A is insufficient):
  - Classification: local kNN margin (nearest same-class dist - nearest diff-class dist)
  - Regression: local linear fit R² in neighborhood
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from rail.data import load_openml_dataset, train_val_test_split

logger = logging.getLogger(__name__)

# -- Canonical datasets (also recorded in configs/datasets.yaml) --

CLASSIFICATION_DATASETS = [
    {"name": "adult", "dataset_id": 1590},
    {"name": "credit-g", "dataset_id": 31},
    {"name": "diabetes", "dataset_id": 37},
    {"name": "letter", "dataset_id": 6},
    {"name": "page-blocks", "dataset_id": 30},
    {"name": "phoneme", "dataset_id": 1489},
    {"name": "satimage", "dataset_id": 182},
    {"name": "segment", "dataset_id": 36},
    {"name": "vehicle", "dataset_id": 54},
]

REGRESSION_DATASETS = [
    {"name": "california_housing", "dataset_id": 43939},
    {"name": "concrete_compressive_strength", "dataset_id": 44959},
    {"name": "energy_efficiency", "dataset_id": 43918},
    {"name": "kin8nm", "dataset_id": 189},
    {"name": "wine_quality", "dataset_id": 287},
]


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessor matching the retrieval pipeline."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_cols,
        ))

    return ColumnTransformer(transformers, remainder="drop")


def compute_leq_a_classification(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    k: int = 100,
) -> dict:
    """Compute LEQ_A for classification.

    Key insight from v1 failure: purity measures "how good the neighborhood already is",
    but we need "how much can locality IMPROVE over global evidence".

    Metrics computed:
    - purity: fraction of same-class neighbors (for reference)
    - knn_accuracy: kNN classifier accuracy using k neighbors
    - majority_baseline: global majority class accuracy
    - locality_benefit: kNN accuracy - majority baseline (the LEQ we actually want)
    - entropy_reduction: global entropy - local entropy (normalized)
    - class_concentration_gain: how much more concentrated local class dist is vs global
    """
    preprocessor = _build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    nn = NearestNeighbors(n_neighbors=min(k, len(X_train) - 1), metric="euclidean")
    nn.fit(X_train_proc)
    distances, indices = nn.kneighbors(X_test_proc)

    y_train_arr = np.array(y_train)
    y_test_arr = np.array(y_test)
    n_test = len(X_test)
    classes = np.unique(y_train_arr)
    n_classes = len(classes)

    # Global baselines
    global_counts = np.array([np.sum(y_train_arr == c) for c in classes])
    global_probs = global_counts / global_counts.sum()
    global_entropy = -np.sum(global_probs * np.log2(global_probs + 1e-12))
    majority_class = classes[np.argmax(global_counts)]
    majority_baseline = float(np.mean(y_test_arr == majority_class))

    purities = np.zeros(n_test)
    knn_correct = np.zeros(n_test)  # 1 if kNN predicts correctly
    local_entropies = np.zeros(n_test)
    entropy_reductions = np.zeros(n_test)
    class_concentration_gains = np.zeros(n_test)

    for i in range(n_test):
        neighbor_labels = y_train_arr[indices[i]]
        true_label = y_test_arr[i]

        # Purity
        purities[i] = np.mean(neighbor_labels == true_label)

        # kNN prediction: majority vote among neighbors
        unique_labels, label_counts = np.unique(neighbor_labels, return_counts=True)
        knn_pred = unique_labels[np.argmax(label_counts)]
        knn_correct[i] = 1.0 if knn_pred == true_label else 0.0

        # Local entropy
        local_probs = label_counts / label_counts.sum()
        local_entropy = -np.sum(local_probs * np.log2(local_probs + 1e-12))
        local_entropies[i] = local_entropy

        # Entropy reduction: how much more concentrated is local vs global
        max_entropy = np.log2(n_classes) if n_classes > 1 else 1.0
        entropy_reductions[i] = (global_entropy - local_entropy) / max_entropy

        # Class concentration gain:
        # KL divergence from uniform to local (higher = more concentrated)
        # minus KL divergence from uniform to global
        # Simplified: just use max local prob - max global prob
        class_concentration_gains[i] = np.max(local_probs) - np.max(global_probs)

    knn_accuracy = float(np.mean(knn_correct))
    locality_benefit = knn_accuracy - majority_baseline

    return {
        "leq_a_purity": purities,
        "knn_correct": knn_correct,
        "entropy_reductions": entropy_reductions,
        "class_concentration_gains": class_concentration_gains,
        "mean_purity": float(np.mean(purities)),
        "knn_accuracy": knn_accuracy,
        "majority_baseline": majority_baseline,
        "locality_benefit": locality_benefit,
        "mean_entropy_reduction": float(np.mean(entropy_reductions)),
        "mean_class_concentration_gain": float(np.mean(class_concentration_gains)),
        "n_test": n_test,
        "n_classes": n_classes,
        "global_entropy": float(global_entropy),
    }


def compute_leq_a_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    k: int = 16,
) -> dict:
    """Compute LEQ_A for regression: neighborhood target variance compression.

    LEQ_A = 1 - (local_var / global_var)
    Higher = smoother local neighborhood = better local evidence.
    """
    preprocessor = _build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    nn = NearestNeighbors(n_neighbors=min(k, len(X_train) - 1), metric="euclidean")
    nn.fit(X_train_proc)
    distances, indices = nn.kneighbors(X_test_proc)

    y_train_arr = np.array(y_train, dtype=float)
    global_var = np.var(y_train_arr)
    if global_var < 1e-12:
        global_var = 1e-12  # avoid division by zero

    n_test = len(X_test)
    variance_compressions = np.zeros(n_test)
    local_vars = np.zeros(n_test)

    for i in range(n_test):
        neighbor_targets = y_train_arr[indices[i]]
        local_var = np.var(neighbor_targets)
        local_vars[i] = local_var
        variance_compressions[i] = 1.0 - (local_var / global_var)

    return {
        "leq_a_variance_compression": variance_compressions,
        "local_vars": local_vars,
        "global_var": float(global_var),
        "mean_variance_compression": float(np.mean(variance_compressions)),
        "n_test": n_test,
    }


def load_retrieval_gains(results_dir: Path) -> pd.DataFrame:
    """Derive the paper's primary retrieval gains from the released main tables."""
    classification = pd.read_csv(results_dir / "classification_main.csv")
    regression = pd.read_csv(results_dir / "regression_main.csv")
    rows: list[dict[str, object]] = []

    for dataset in classification["dataset"].unique():
        subset = classification[
            (classification["dataset"] == dataset)
            & (classification["llm_model"] == "qwen3.5-397b-a17b")
        ]
        vanilla = subset[subset["method"] == "vanilla_icl"].iloc[0]
        retrieval = subset[
            (subset["method"] == "retrieval_icl")
            & (subset["selection_strategy"] == "matched_pair_boundary")
        ].iloc[0]
        rows.append(
            {
                "task_family": "classification",
                "dataset": dataset,
                "metric": "accuracy",
                "improvement_score": float(retrieval["accuracy"] - vanilla["accuracy"]),
            }
        )

    numeric_k = pd.to_numeric(regression["k"], errors="coerce")
    for dataset in regression["dataset"].unique():
        subset = regression[
            (regression["dataset"] == dataset)
            & (regression["llm_model"] == "qwen3.5-397b-a17b")
            & (numeric_k == 16)
        ]
        vanilla = subset[subset["method"] == "vanilla_icl"].iloc[0]
        retrieval = subset[subset["method"] == "retrieval_icl"].iloc[0]
        vanilla_rmse = float(vanilla["rmse"])
        rows.append(
            {
                "task_family": "regression",
                "dataset": dataset,
                "metric": "rmse_reduction",
                "improvement_score": (vanilla_rmse - float(retrieval["rmse"])) / vanilla_rmse,
            }
        )

    return pd.DataFrame(rows)


def run_leq_analysis(
    output_dir: Path | None = None,
    k_cls: int = 100,
    k_reg: int = 16,
) -> pd.DataFrame:
    """Run LEQ_A computation on all 14 canonical datasets and correlate with retrieval gains."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    released_results = PROJECT_ROOT / "artifacts" / "results"
    gains_df = load_retrieval_gains(released_results)

    rows = []

    # --- Classification datasets ---
    for ds_info in CLASSIFICATION_DATASETS:
        name = ds_info["name"]
        logger.info(f"Computing LEQ for classification dataset: {name}")
        try:
            dataset = load_openml_dataset(
                dataset_id=ds_info["dataset_id"],
                task_type_override="classification",
            )
            split = train_val_test_split(
                dataset.features,
                dataset.target,
                task_type="classification",
                test_size=0.2,
                val_size=0.2,
                random_state=42,
            )
            leq = compute_leq_a_classification(
                split.X_train, split.y_train,
                split.X_test, split.y_test,
                k=k_cls,
            )

            # Get retrieval gain
            gain_row = gains_df[
                (gains_df["dataset"] == name) & (gains_df["task_family"] == "classification")
            ]
            gain = float(gain_row["improvement_score"].iloc[0]) if len(gain_row) > 0 else np.nan

            rows.append({
                "dataset": name,
                "task_family": "classification",
                "n_classes": leq["n_classes"],
                "leq_purity": leq["mean_purity"],
                "knn_accuracy": leq["knn_accuracy"],
                "majority_baseline": leq["majority_baseline"],
                "locality_benefit": leq["locality_benefit"],
                "entropy_reduction": leq["mean_entropy_reduction"],
                "class_concentration_gain": leq["mean_class_concentration_gain"],
                "retrieval_gain": gain,
                "n_test": leq["n_test"],
            })
            logger.info(
                f"  {name}: knn_acc={leq['knn_accuracy']:.4f}, "
                f"majority={leq['majority_baseline']:.4f}, "
                f"locality_benefit={leq['locality_benefit']:.4f}, "
                f"entropy_red={leq['mean_entropy_reduction']:.4f}, gain={gain:.4f}"
            )
        except Exception as e:
            logger.error(f"  Failed to compute LEQ for {name}: {e}")
            rows.append({
                "dataset": name,
                "task_family": "classification",
                "n_classes": np.nan,
                "leq_purity": np.nan,
                "knn_accuracy": np.nan,
                "majority_baseline": np.nan,
                "locality_benefit": np.nan,
                "entropy_reduction": np.nan,
                "class_concentration_gain": np.nan,
                "retrieval_gain": np.nan,
                "n_test": np.nan,
            })

    # --- Regression datasets ---
    for ds_info in REGRESSION_DATASETS:
        name = ds_info["name"]
        logger.info(f"Computing LEQ for regression dataset: {name}")
        try:
            dataset = load_openml_dataset(
                dataset_id=ds_info["dataset_id"],
                task_type_override="regression",
            )
            split = train_val_test_split(
                dataset.features,
                dataset.target,
                task_type="regression",
                test_size=0.2,
                val_size=0.2,
                random_state=42,
            )
            leq = compute_leq_a_regression(
                split.X_train, split.y_train,
                split.X_test, split.y_test,
                k=k_reg,
            )

            # Get retrieval gain (for regression, use R² improvement which is higher-is-better)
            gain_row = gains_df[
                (gains_df["dataset"] == name)
                & (gains_df["task_family"] == "regression")
            ]
            # Use RMSE improvement_score (already normalized to positive = better)
            gain = float(gain_row["improvement_score"].iloc[0]) if len(gain_row) > 0 else np.nan

            rows.append({
                "dataset": name,
                "task_family": "regression",
                "n_classes": np.nan,
                "leq_purity": np.nan,
                "leq_concentration": np.nan,
                "leq_margin": np.nan,
                "leq_variance_compression": leq["mean_variance_compression"],
                "retrieval_gain": gain,
                "n_test": leq["n_test"],
            })
            logger.info(
                f"  {name}: variance_compression={leq['mean_variance_compression']:.4f}, "
                f"gain={gain:.4f}"
            )
        except Exception as e:
            logger.error(f"  Failed to compute LEQ for {name}: {e}")
            rows.append({
                "dataset": name,
                "task_family": "regression",
                "n_classes": np.nan,
                "leq_purity": np.nan,
                "leq_concentration": np.nan,
                "leq_margin": np.nan,
                "leq_variance_compression": np.nan,
                "retrieval_gain": np.nan,
                "n_test": np.nan,
            })

    summary_df = pd.DataFrame(rows)

    # --- Compute correlations ---
    print("\n" + "=" * 70)
    print("LEQ vs Retrieval Gain Correlation Analysis")
    print("=" * 70)

    # Classification: multiple LEQ variants vs gain
    cls_df = summary_df[summary_df["task_family"] == "classification"].dropna(
        subset=["locality_benefit", "retrieval_gain"]
    )
    if len(cls_df) >= 3:
        metrics_to_test = [
            ("locality_benefit", "Locality Benefit (kNN_acc - majority)"),
            ("entropy_reduction", "Entropy Reduction (global - local)"),
            ("class_concentration_gain", "Class Concentration Gain"),
            ("leq_purity", "Purity (reference, expected weak)"),
            ("knn_accuracy", "kNN Accuracy"),
        ]
        print(f"\n--- Classification (n={len(cls_df)}) ---")
        for col, label in metrics_to_test:
            valid = cls_df.dropna(subset=[col])
            if len(valid) >= 3:
                r, p = stats.spearmanr(valid[col], valid["retrieval_gain"])
                marker = " ***" if p < 0.05 else " **" if p < 0.1 else " *" if p < 0.2 else ""
                print(f"  {label:50s} Spearman r={r:.4f}, p={p:.4f}{marker}")

    # Regression: variance compression vs gain
    reg_df = summary_df[summary_df["task_family"] == "regression"].dropna(
        subset=["leq_variance_compression", "retrieval_gain"]
    )
    if len(reg_df) >= 3:
        r_vc, p_vc = stats.spearmanr(reg_df["leq_variance_compression"], reg_df["retrieval_gain"])
        print(f"\n--- Regression (n={len(reg_df)}) ---")
        print(f"  Variance Compression vs Gain: Spearman r={r_vc:.4f}, p={p_vc:.4f}")

    # Combined: use locality_benefit for classification, variance_compression for regression
    unified = summary_df.copy()
    unified["leq_unified"] = np.where(
        unified["task_family"] == "classification",
        unified["locality_benefit"],
        unified["leq_variance_compression"],
    )
    valid = unified.dropna(subset=["leq_unified", "retrieval_gain"])
    if len(valid) >= 5:
        r_all, p_all = stats.spearmanr(valid["leq_unified"], valid["retrieval_gain"])
        print(f"\n--- All datasets unified (n={len(valid)}) ---")
        print(f"  Unified LEQ vs Gain: Spearman r={r_all:.4f}, p={p_all:.4f}")

    # Save results
    out_path = output_dir / "leq_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    # --- Generate correlation table for easy reading ---
    print("\n--- Per-dataset summary ---")
    print(f"  {'Dataset':35s} {'Task':15s} {'LEQ_primary':>12s} {'kNN_acc':>8s} {'Majority':>8s} {'Gain':>10s}")
    print("  " + "-" * 95)
    for _, row in summary_df.iterrows():
        ds = row["dataset"]
        tf = row["task_family"]
        gain = row["retrieval_gain"]
        if tf == "classification":
            leq_val = row["locality_benefit"]
            leq_name = "loc_benefit"
            knn_acc = row.get("knn_accuracy", np.nan)
            majority = row.get("majority_baseline", np.nan)
            print(f"  {ds:35s} {tf:15s} {leq_val:>12.4f} {knn_acc:>8.4f} {majority:>8.4f} {gain:>+10.4f}")
        else:
            leq_val = row.get("leq_variance_compression", np.nan)
            print(f"  {ds:35s} {tf:15s} {leq_val:>12.4f} {'--':>8s} {'--':>8s} {gain:>+10.4f}")

    return summary_df


def generate_scatter_plot(summary_df: pd.DataFrame, output_dir: Path | None = None):
    """Generate scatter plot: LEQ vs retrieval gain."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot generation")
        return

    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Classification locality_benefit vs gain
    cls_df = summary_df[summary_df["task_family"] == "classification"].dropna(
        subset=["locality_benefit", "retrieval_gain"]
    )
    ax = axes[0]
    ax.scatter(cls_df["locality_benefit"], cls_df["retrieval_gain"], s=80, c="steelblue", edgecolors="black")
    for _, row in cls_df.iterrows():
        ax.annotate(row["dataset"], (row["locality_benefit"], row["retrieval_gain"]),
                    fontsize=7, ha="center", va="bottom", xytext=(0, 5),
                    textcoords="offset points")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Locality Benefit (kNN acc - majority baseline)")
    ax.set_ylabel("Retrieval Gain (accuracy)")
    ax.set_title("Classification: Locality Benefit vs Retrieval Gain")
    if len(cls_df) >= 3:
        r, p = stats.spearmanr(cls_df["locality_benefit"], cls_df["retrieval_gain"])
        ax.text(0.05, 0.95, f"Spearman r={r:.3f}\np={p:.3f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Plot 2: Classification entropy reduction vs gain
    ax = axes[1]
    ax.scatter(cls_df["entropy_reduction"], cls_df["retrieval_gain"], s=80, c="coral", edgecolors="black")
    for _, row in cls_df.iterrows():
        ax.annotate(row["dataset"], (row["entropy_reduction"], row["retrieval_gain"]),
                    fontsize=7, ha="center", va="bottom", xytext=(0, 5),
                    textcoords="offset points")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Entropy Reduction (global - local, normalized)")
    ax.set_ylabel("Retrieval Gain (accuracy)")
    ax.set_title("Classification: Entropy Reduction vs Retrieval Gain")
    if len(cls_df) >= 3:
        r, p = stats.spearmanr(cls_df["entropy_reduction"], cls_df["retrieval_gain"])
        ax.text(0.05, 0.95, f"Spearman r={r:.3f}\np={p:.3f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Plot 3: Regression variance compression vs gain
    reg_df = summary_df[summary_df["task_family"] == "regression"].dropna(
        subset=["leq_variance_compression", "retrieval_gain"]
    )
    ax = axes[2]
    ax.scatter(reg_df["leq_variance_compression"], reg_df["retrieval_gain"],
               s=80, c="forestgreen", edgecolors="black")
    for _, row in reg_df.iterrows():
        ax.annotate(row["dataset"], (row["leq_variance_compression"], row["retrieval_gain"]),
                    fontsize=7, ha="center", va="bottom", xytext=(0, 5),
                    textcoords="offset points")
    ax.set_xlabel("LEQ_A (Variance Compression)")
    ax.set_ylabel("Retrieval Gain (RMSE improvement)")
    ax.set_title("Regression: Variance Compression vs Retrieval Gain")
    if len(reg_df) >= 3:
        r, p = stats.spearmanr(reg_df["leq_variance_compression"], reg_df["retrieval_gain"])
        ax.text(0.05, 0.95, f"Spearman r={r:.3f}\np={p:.3f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    out_path = output_dir / "leq_vs_gain_correlation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nScatter plot saved to: {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = run_leq_analysis()
    generate_scatter_plot(summary)
