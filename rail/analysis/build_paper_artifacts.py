from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
RESULTS = ARTIFACTS / "results"
FIGURES = ARTIFACTS / "figures"
TABLES = ARTIFACTS / "tables"
DOCS = ROOT / "docs"


CLASSIFICATION_DATASETS = [
    "adult",
    "credit-g",
    "diabetes",
    "phoneme",
    "vehicle",
    "segment",
    "page-blocks",
    "satimage",
    "letter",
]
REGRESSION_DATASETS = [
    "california_housing",
    "wine_quality",
    "concrete_compressive_strength",
    "energy_efficiency",
    "kin8nm",
]


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def _fmt_signed(value: float | int | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):+.{digits}f}"


def _escape_latex(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _latex_table(headers: list[str], rows: list[list[str]], align: str) -> str:
    lines = [
        "\\begin{tabular}{" + align + "}",
        "\\toprule",
        " & ".join(_escape_latex(h) for h in headers) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_escape_latex(cell) for cell in row) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


def _fmt_pvalue(value: float) -> str:
    if value < 0.001:
        return f"{value:.4f}"
    return f"{value:.3f}"


def _classification_main_rows() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "classification_main.csv")
    rows = []
    for dataset in CLASSIFICATION_DATASETS:
        vanilla = df[
            (df["dataset"] == dataset)
            & (df["method"] == "vanilla_icl")
            & (df["llm_model"] == "qwen3.5-397b-a17b")
        ].iloc[0]
        retrieval = df[
            (df["dataset"] == dataset)
            & (df["method"] == "retrieval_icl")
            & (df["llm_model"] == "qwen3.5-397b-a17b")
            & (df["selection_strategy"] == "matched_pair_boundary")
        ].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "type": "classification",
                "vanilla": float(vanilla["accuracy"]),
                "retrieval": float(retrieval["accuracy"]),
                "gain": float(retrieval["accuracy"] - vanilla["accuracy"]),
                "vanilla_f1": float(vanilla["f1_macro"]),
                "retrieval_f1": float(retrieval["f1_macro"]),
            }
        )
    return pd.DataFrame(rows)


def _regression_main_rows() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "regression_main.csv")
    rows = []
    for dataset in REGRESSION_DATASETS:
        vanilla = df[
            (df["dataset"] == dataset)
            & (df["method"] == "vanilla_icl")
            & (df["llm_model"] == "qwen3.5-397b-a17b")
            & (pd.to_numeric(df["k"], errors="coerce") == 16)
        ].iloc[0]
        retrieval = df[
            (df["dataset"] == dataset)
            & (df["method"] == "retrieval_icl")
            & (df["llm_model"] == "qwen3.5-397b-a17b")
            & (pd.to_numeric(df["k"], errors="coerce") == 16)
        ].iloc[0]
        vanilla_rmse = float(vanilla["rmse"])
        retrieval_rmse = float(retrieval["rmse"])
        rows.append(
            {
                "dataset": dataset,
                "type": "regression",
                "vanilla": vanilla_rmse,
                "retrieval": retrieval_rmse,
                "gain": (vanilla_rmse - retrieval_rmse) / vanilla_rmse,
            }
        )
    return pd.DataFrame(rows)


def _main_lba_frame() -> pd.DataFrame:
    leq = pd.read_csv(RESULTS / "lba_main.csv")
    cls = _classification_main_rows()
    reg = _regression_main_rows()
    rows = []
    for row in cls.itertuples(index=False):
        leq_row = leq[leq["dataset"] == row.dataset].iloc[0]
        rows.append(
            {
                "dataset": row.dataset,
                "type": "classification",
                "lba": float(leq_row["locality_benefit"]),
                "vanilla": row.vanilla,
                "retrieval": row.retrieval,
                "gain": row.gain,
                "regime": "multi-class" if row.dataset in {"vehicle", "segment", "page-blocks", "satimage", "letter"} else "binary",
            }
        )
    for row in reg.itertuples(index=False):
        leq_row = leq[leq["dataset"] == row.dataset].iloc[0]
        rows.append(
            {
                "dataset": row.dataset,
                "type": "regression",
                "lba": float(leq_row["leq_variance_compression"]),
                "vanilla": row.vanilla,
                "retrieval": row.retrieval,
                "gain": row.gain,
                "regime": "regression",
            }
        )
    return pd.DataFrame(rows)


def build_figure_lba_vs_gain() -> None:
    df = _main_lba_frame()
    cls = df[df["type"] == "classification"]
    all_rho, all_p = stats.spearmanr(df["lba"], df["gain"])
    cls_rho, cls_p = stats.spearmanr(cls["lba"], cls["gain"])

    fig, ax = plt.subplots(figsize=(4.8, 3.35))
    styles = {
        "binary": {"marker": "o", "facecolor": "#e0f2fe", "edgecolor": "#0369a1", "label": "Binary cls."},
        "multi-class": {"marker": "o", "facecolor": "#0f766e", "edgecolor": "#064e3b", "label": "Multi-class cls."},
        "regression": {"marker": "^", "facecolor": "#f59e0b", "edgecolor": "#92400e", "label": "Regression"},
    }
    for regime, group in df.groupby("regime", sort=False):
        style = styles[regime]
        ax.scatter(
            group["lba"],
            group["gain"],
            s=48,
            marker=style["marker"],
            facecolors=style["facecolor"],
            edgecolors=style["edgecolor"],
            linewidths=1.2,
            label=style["label"],
        )
    label_offsets = {
        "letter": (5, 2),
        "segment": (5, 3),
        "credit-g": (5, -1),
        "adult": (5, -1),
        "page-blocks": (5, 5),
        "california_housing": (5, 3),
    }
    for _, row in df.iterrows():
        if row["dataset"] in {"letter", "segment", "credit-g", "adult", "page-blocks", "california_housing"}:
            ax.annotate(
                row["dataset"],
                (row["lba"], row["gain"]),
                fontsize=7.5,
                xytext=label_offsets[row["dataset"]],
                textcoords="offset points",
            )
    ax.axhline(0, color="#6b7280", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Estimated LBA / LEQ", fontsize=8.5)
    ax.set_ylabel("Retrieval gain", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.text(
        0.02,
        0.98,
        f"classification rho={cls_rho:.2f}, p={_fmt_pvalue(cls_p)}\nall datasets rho={all_rho:.2f}, p={_fmt_pvalue(all_p)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )
    _style_ax(ax)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        loc="lower right",
        fontsize=7.8,
    )
    fig.tight_layout()
    _save_figure(fig, "lba_vs_gain")


def build_figure_query_margin() -> None:
    df = pd.read_csv(RESULTS / "query_margin_buckets.csv")
    keep = ["letter", "diabetes", "credit-g"]
    bucket_order = ["negative_evidence", "ambiguous_evidence", "strong_local_evidence"]
    bucket_labels = {
        "negative_evidence": "Negative",
        "ambiguous_evidence": "Ambiguous",
        "strong_local_evidence": "Strong",
    }
    colors = {
        "negative_evidence": "#b91c1c",
        "ambiguous_evidence": "#f59e0b",
        "strong_local_evidence": "#0f766e",
    }
    fig, axes = plt.subplots(1, len(keep), figsize=(7.1, 2.6), sharey=True)
    for ax, dataset in zip(axes, keep, strict=True):
        sub = df[df["dataset"] == dataset].set_index("bucket")
        values = [float(sub.loc[b, "mean_gain"]) if b in sub.index else np.nan for b in bucket_order]
        counts = [int(sub.loc[b, "query_count"]) if b in sub.index else 0 for b in bucket_order]
        plot_values = [0.0 if pd.isna(value) else value for value in values]
        ax.bar(range(len(bucket_order)), plot_values, color=[colors[b] for b in bucket_order], width=0.68)
        ax.axhline(0, color="#374151", linewidth=0.9)
        ax.set_xticks(range(len(bucket_order)), [bucket_labels[b] for b in bucket_order], rotation=18, ha="right")
        ax.set_ylim(-0.24, 1.08)
        ax.set_title(dataset, fontsize=10)
        ax.tick_params(labelsize=8.5)
        for i, (value, count) in enumerate(zip(values, counts, strict=True)):
            if count == 0:
                ax.text(i, 0.035, "n=0", ha="center", va="bottom", fontsize=8, color="#6b7280")
                continue
            if pd.isna(value):
                continue
            y = value + (0.035 if value >= 0 else -0.025)
            ax.text(i, y, f"n={count}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8.3)
        _style_ax(ax)
    axes[0].set_ylabel("Mean query-level gain", fontsize=9.5)
    fig.tight_layout()
    _save_figure(fig, "query_margin_buckets")


def build_figure_gating() -> None:
    df = pd.read_csv(RESULTS / "gating_policy_summary.csv")
    test = df[df["phase"] == "test"].copy()
    datasets = ["credit-g", "diabetes", "letter"]
    policies = ["never_retrieve", "always_retrieve", "random_gate", "label_free_gate"]
    labels = ["Never", "Always", "Random", "LEQ gate"]
    colors = ["#94a3b8", "#0f766e", "#f59e0b", "#1d4ed8"]

    fig, axes = plt.subplots(1, len(datasets), figsize=(7.2, 2.75), sharey=True)
    for ax, dataset in zip(axes, datasets, strict=True):
        sub = test[test["dataset"] == dataset].set_index("policy")
        values = [float(sub.loc[p, "accuracy"]) for p in policies]
        rates = [float(sub.loc[p, "retrieval_rate"]) for p in policies]
        ax.bar(range(len(policies)), values, color=colors, width=0.68)
        ax.set_xticks(range(len(policies)), labels, rotation=22, ha="right")
        ax.set_ylim(0, 1.16)
        tau = sub.loc["label_free_gate", "selected_threshold"]
        ax.set_title(f"{dataset}\n$\\tau$={tau:.1f}", fontsize=10)
        ax.tick_params(labelsize=8.5)
        for i, (value, rate) in enumerate(zip(values, rates, strict=True)):
            ax.text(i, value + 0.025, f"{value:.2f}\nr={rate:.2f}", ha="center", va="bottom", fontsize=8.1)
        _style_ax(ax)
    axes[0].set_ylabel("Test accuracy", fontsize=9.5)
    fig.tight_layout()
    _save_figure(fig, "validation_gating")


def build_table_main_lba() -> None:
    df = _main_lba_frame()
    order = REGRESSION_DATASETS + ["adult", "credit-g", "diabetes", "phoneme", "vehicle", "segment", "page-blocks", "satimage", "letter"]
    df["order"] = df["dataset"].map({name: i for i, name in enumerate(order)})
    rows = []
    for row in df.sort_values("order").itertuples(index=False):
        if row.type == "classification":
            metric = "Acc"
            vanilla = _fmt(row.vanilla, 3)
            retrieval = _fmt(row.retrieval, 3)
            gain = _fmt_signed(row.gain, 3)
        else:
            metric = "RMSE"
            vanilla = _fmt(row.vanilla, 2 if row.vanilla > 10 else 4)
            retrieval = _fmt(row.retrieval, 2 if row.retrieval > 10 else 4)
            gain = _fmt_signed(row.gain, 3)
        short_type = "Cls" if row.type == "classification" else "Reg"
        rows.append([row.dataset, short_type, metric, _fmt(row.lba, 3), vanilla, retrieval, gain])
    headers = ["Dataset", "Type", "Metric", "LBA/LEQ", "Vanilla", "Retrieval", "Gain"]
    md = _markdown_table(headers, rows)
    tex = _latex_table(headers, rows, "lllrrrr")
    _write(TABLES / "lba_main.md", md)
    _write(TABLES / "lba_main.tex", tex)


def build_table_gating() -> None:
    df = pd.read_csv(RESULTS / "gating_compact.csv")
    order = [
        "adult",
        "credit-g",
        "diabetes",
        "phoneme",
        "vehicle",
        "segment",
        "page-blocks",
        "satimage",
        "letter",
    ]
    indexed = df.set_index("dataset")
    rows = []
    for dataset in order:
        row = indexed.loc[dataset]
        rows.append(
            [
                dataset,
                f"{_fmt(row['never_acc'], 3)}/{_fmt(row['never_f1'], 3)}",
                f"{_fmt(row['always_acc'], 3)}/{_fmt(row['always_f1'], 3)}",
                f"{_fmt(row['gate_acc'], 3)}/{_fmt(row['gate_f1'], 3)}",
                _fmt(row["gate_rate"], 2),
                _fmt(row["gate_minus_always_acc"], 3),
            ]
        )
    headers = ["Dataset", "Never", "Always", "Gate", "Rate", "Delta Gate-Always"]
    _write(TABLES / "validation_gating.md", _markdown_table(headers, rows))
    tex = _latex_table(headers, rows, "lrrrrr").replace("Delta Gate-Always", r"$\Delta$Gate-Always")
    _write(TABLES / "validation_gating.tex", tex)


def build_table_intervention() -> None:
    df = pd.read_csv(RESULTS / "label_intervention.csv")
    metric_rows = df[df["condition"].isin(["vanilla", "retrieval_control", "retrieval_label_shuffle", "matched_histogram_random_support"])]
    metric = metric_rows.set_index(["dataset", "condition"])
    delta = df.set_index(["dataset", "condition"])
    rows = []
    for dataset in ["letter", "diabetes", "credit-g"]:
        ret_shuffle = delta.loc[(dataset, "retrieval_vs_shuffle_paired_ttest")]
        ret_hist = delta.loc[(dataset, "retrieval_vs_matched_histogram_paired_ttest")]
        rows.append(
            [
                dataset,
                _fmt(metric.loc[(dataset, "vanilla")]["accuracy"], 3),
                _fmt(metric.loc[(dataset, "retrieval_control")]["accuracy"], 3),
                _fmt(metric.loc[(dataset, "retrieval_label_shuffle")]["accuracy"], 3),
                _fmt(metric.loc[(dataset, "matched_histogram_random_support")]["accuracy"], 3),
                _fmt_signed(ret_shuffle["retrieval_minus_shuffle_accuracy"], 3),
                _fmt_signed(ret_hist["retrieval_minus_matched_hist_accuracy"], 3),
            ]
        )
    headers = ["Dataset", "Vanilla", "Retrieval", "Shuffle", "Hist-rand", "Ret-Shuf", "Ret-Hist"]
    _write(TABLES / "label_intervention.md", _markdown_table(headers, rows))
    _write(TABLES / "label_intervention.tex", _latex_table(headers, rows, "lrrrrrr"))


def build_index() -> None:
    text = """# Generated Main-result Artifacts

Generated by `rail/analysis/build_paper_artifacts.py`.

## Figures

- `artifacts/figures/lba_vs_gain.pdf`
- `artifacts/figures/query_margin_buckets.pdf`
- `artifacts/figures/validation_gating.pdf`

PNG copies are also generated for quick viewing.

## Tables

- `artifacts/tables/lba_main.tex`
- `artifacts/tables/validation_gating.tex`
- `artifacts/tables/label_intervention.tex`

Markdown copies are also generated.
"""
    _write(DOCS / "GENERATED_ARTIFACTS.md", text)


def main() -> None:
    build_figure_lba_vs_gain()
    build_figure_query_margin()
    build_figure_gating()
    build_table_main_lba()
    build_table_gating()
    build_table_intervention()
    build_index()


if __name__ == "__main__":
    main()
