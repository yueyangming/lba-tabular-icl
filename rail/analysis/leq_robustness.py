"""LEQ robustness checks: leave-one-out stability and bootstrap CI.

Reads artifacts/results/lba_main.csv (already computed) and runs:
1. Leave-one-dataset-out: re-compute Spearman r dropping each dataset
2. Bootstrap 95% CI for Spearman r
3. Check if result is driven by outliers (letter, credit-g)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, n_boot: int = 10000, seed: int = 42) -> dict:
    """Bootstrap confidence interval for Spearman correlation."""
    rng = np.random.RandomState(seed)
    n = len(x)
    boot_rs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            boot_rs.append(r)
    boot_rs = np.array(boot_rs)
    return {
        "mean": float(np.mean(boot_rs)),
        "std": float(np.std(boot_rs)),
        "ci_lower": float(np.percentile(boot_rs, 2.5)),
        "ci_upper": float(np.percentile(boot_rs, 97.5)),
        "n_boot": len(boot_rs),
    }


def run_robustness():
    """Run all robustness checks."""
    df = pd.read_csv(PROJECT_ROOT / "artifacts" / "results" / "lba_main.csv")

    # Build unified LEQ column
    cls_df = df[df["task_family"] == "classification"].copy()
    reg_df = df[df["task_family"] == "regression"].copy()

    cls_df["leq"] = cls_df["locality_benefit"]
    reg_df["leq"] = reg_df["leq_variance_compression"]

    all_df = pd.concat([cls_df[["dataset", "task_family", "leq", "retrieval_gain"]],
                        reg_df[["dataset", "task_family", "leq", "retrieval_gain"]]])
    all_df = all_df.dropna(subset=["leq", "retrieval_gain"])

    print("=" * 70)
    print("LEQ Robustness Analysis")
    print("=" * 70)

    # =============================================
    # 1. Leave-one-dataset-out (classification only)
    # =============================================
    print("\n--- 1. Leave-One-Dataset-Out (Classification, n=9) ---")
    print(f"  {'Dropped':25s} {'n':>3s} {'Spearman r':>12s} {'p-value':>10s}")
    print(f"  {'-'*55}")

    cls_valid = cls_df.dropna(subset=["leq", "retrieval_gain"])
    r_full, p_full = stats.spearmanr(cls_valid["leq"], cls_valid["retrieval_gain"])
    print(f"  {'(none - full)':25s} {len(cls_valid):>3d} {r_full:>12.4f} {p_full:>10.4f}")

    loo_rs = []
    for drop_ds in cls_valid["dataset"].values:
        subset = cls_valid[cls_valid["dataset"] != drop_ds]
        r, p = stats.spearmanr(subset["leq"], subset["retrieval_gain"])
        loo_rs.append(r)
        print(f"  {drop_ds:25s} {len(subset):>3d} {r:>12.4f} {p:>10.4f}")

    print(f"\n  LOO range: [{min(loo_rs):.4f}, {max(loo_rs):.4f}]")
    print(f"  LOO mean:  {np.mean(loo_rs):.4f}")
    print(f"  Full r:    {r_full:.4f}")

    # =============================================
    # 2. Leave-one-dataset-out (all 14)
    # =============================================
    print("\n--- 2. Leave-One-Dataset-Out (All 14 datasets) ---")
    print(f"  {'Dropped':35s} {'n':>3s} {'Spearman r':>12s} {'p-value':>10s}")
    print(f"  {'-'*65}")

    r_full_all, p_full_all = stats.spearmanr(all_df["leq"], all_df["retrieval_gain"])
    print(f"  {'(none - full)':35s} {len(all_df):>3d} {r_full_all:>12.4f} {p_full_all:>10.4f}")

    loo_rs_all = []
    for drop_ds in all_df["dataset"].values:
        subset = all_df[all_df["dataset"] != drop_ds]
        r, p = stats.spearmanr(subset["leq"], subset["retrieval_gain"])
        loo_rs_all.append(r)
        print(f"  {drop_ds:35s} {len(subset):>3d} {r:>12.4f} {p:>10.4f}")

    print(f"\n  LOO range: [{min(loo_rs_all):.4f}, {max(loo_rs_all):.4f}]")
    print(f"  LOO mean:  {np.mean(loo_rs_all):.4f}")
    print(f"  Full r:    {r_full_all:.4f}")

    # =============================================
    # 3. Bootstrap CI (classification)
    # =============================================
    print("\n--- 3. Bootstrap 95% CI (Classification, n=9) ---")
    x = cls_valid["leq"].values
    y = cls_valid["retrieval_gain"].values
    boot = bootstrap_spearman(x, y)
    print(f"  Full Spearman r: {r_full:.4f}")
    print(f"  Bootstrap mean:  {boot['mean']:.4f} ± {boot['std']:.4f}")
    print(f"  Bootstrap 95% CI: [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")

    # =============================================
    # 4. Bootstrap CI (all 14)
    # =============================================
    print("\n--- 4. Bootstrap 95% CI (All 14 datasets) ---")
    x_all = all_df["leq"].values
    y_all = all_df["retrieval_gain"].values
    boot_all = bootstrap_spearman(x_all, y_all)
    print(f"  Full Spearman r: {r_full_all:.4f}")
    print(f"  Bootstrap mean:  {boot_all['mean']:.4f} ± {boot_all['std']:.4f}")
    print(f"  Bootstrap 95% CI: [{boot_all['ci_lower']:.4f}, {boot_all['ci_upper']:.4f}]")

    # =============================================
    # 5. Pearson for comparison
    # =============================================
    print("\n--- 5. Pearson correlation (for comparison) ---")
    r_pearson_cls, p_pearson_cls = stats.pearsonr(cls_valid["leq"], cls_valid["retrieval_gain"])
    print(f"  Classification: Pearson r={r_pearson_cls:.4f}, p={p_pearson_cls:.4f}")

    # Note: all-dataset Pearson is problematic because gains are in different units
    # But we can rank-transform first
    all_ranked = all_df.copy()
    all_ranked["leq_rank"] = all_ranked["leq"].rank()
    all_ranked["gain_rank"] = all_ranked["retrieval_gain"].rank()
    # Spearman IS rank-correlation, so this is redundant — but shows Pearson on ranks = Spearman
    r_rank, p_rank = stats.pearsonr(all_ranked["leq_rank"], all_ranked["gain_rank"])
    print(f"  All (rank-transformed): Pearson r={r_rank:.4f}, p={p_rank:.4f}  (= Spearman)")

    # =============================================
    # 6. Summary
    # =============================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\n  Classification (n=9):")
    print(f"    Spearman r = {r_full:.4f}, p = {p_full:.4f}")
    print(f"    Bootstrap 95% CI: [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")
    print(f"    LOO stability: [{min(loo_rs):.4f}, {max(loo_rs):.4f}]")
    print(f"\n  All datasets (n=14):")
    print(f"    Spearman r = {r_full_all:.4f}, p = {p_full_all:.4f}")
    print(f"    Bootstrap 95% CI: [{boot_all['ci_lower']:.4f}, {boot_all['ci_upper']:.4f}]")
    print(f"    LOO stability: [{min(loo_rs_all):.4f}, {max(loo_rs_all):.4f}]")

    stable_cls = min(loo_rs) > 0.5
    stable_all = min(loo_rs_all) > 0.5
    ci_excludes_zero_cls = boot["ci_lower"] > 0
    ci_excludes_zero_all = boot_all["ci_lower"] > 0

    print(f"\n  Stability checks:")
    print(f"    Classification LOO all > 0.5: {'✓' if stable_cls else '✗'}")
    print(f"    All-datasets LOO all > 0.5:   {'✓' if stable_all else '✗'}")
    print(f"    Classification CI excludes 0: {'✓' if ci_excludes_zero_cls else '✗'}")
    print(f"    All-datasets CI excludes 0:   {'✓' if ci_excludes_zero_all else '✗'}")


if __name__ == "__main__":
    run_robustness()
