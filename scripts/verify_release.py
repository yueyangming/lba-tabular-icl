from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"

EXPECTED_FILES = [
    "lba_main.csv",
    "classification_main.csv",
    "regression_main.csv",
    "class_sensitive_lba.csv",
    "query_margin_buckets.csv",
    "query_margin_queries.csv",
    "label_intervention.csv",
    "gating_compact.csv",
    "gating_policy_summary.csv",
    "selector_ablation.csv",
    "modern_baselines.csv",
    "k_sensitivity.csv",
    "aux_estimator_sensitivity.csv",
    "cross_family/california_housing.csv",
    "cross_family/kin8nm.csv",
    "cross_family/letter.csv",
    "cross_family/page_blocks.csv",
    "cross_family/credit_g.csv",
]


def _datasets(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["dataset"] for row in csv.DictReader(handle)}


def main() -> None:
    missing = [name for name in EXPECTED_FILES if not (RESULTS / name).is_file()]
    if missing:
        raise SystemExit("Missing released artifacts: " + ", ".join(missing))

    expected_classification = {
        "adult", "credit-g", "diabetes", "phoneme", "vehicle", "segment",
        "page-blocks", "satimage", "letter",
    }
    expected_regression = {
        "california_housing", "wine_quality", "concrete_compressive_strength",
        "energy_efficiency", "kin8nm",
    }

    classification = _datasets(RESULTS / "classification_main.csv")
    regression = _datasets(RESULTS / "regression_main.csv")
    gating = _datasets(RESULTS / "gating_compact.csv")
    leq = _datasets(RESULTS / "lba_main.csv")

    checks = {
        "classification main table": classification == expected_classification,
        "regression main table": regression == expected_regression,
        "nine-dataset gating": gating == expected_classification,
        "14-dataset LBA/LEQ": leq == expected_classification | expected_regression,
    }
    failed = [name for name, ok in checks.items() if not ok]

    for filename, count, k, selector in [
        ("classification_main.csv", 27, "100", "matched_pair_boundary"),
        ("regression_main.csv", 20, "16", "balanced"),
    ]:
        with (RESULTS / filename).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != count or any(
            row["llm_model"] != "baseline" and (
                row["llm_model"] != "qwen3.5-397b-a17b"
                or row["k"] != k
                or row["selection_strategy"] not in {"none", selector}
            ) for row in rows
        ):
            failed.append(f"primary settings in {filename}")

    with (RESULTS / "modern_baselines.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if {"source_file", "error", "error_type"} & set(reader.fieldnames or []):
            failed.append("modern baseline schema")
        rows = list(reader)
        if sum(row["status"] == "unsupported" for row in rows) != 4:
            failed.append("modern baseline applicability coverage")
    if failed:
        raise SystemExit("Artifact coverage check failed: " + ", ".join(failed))

    print(f"Release artifact check passed: {len(EXPECTED_FILES)} files, 14 datasets.")


if __name__ == "__main__":
    main()
