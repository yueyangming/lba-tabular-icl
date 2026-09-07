# Reproducibility guide

The repository separates offline verification from experiments that need network access, paid APIs, or GPU-capable optional baselines.

## 1. Verify the release package

This check uses only Python's standard library.

```bash
python scripts/verify_release.py
```

It verifies that all required main-result files are present and that the classification, regression, gating, and LBA artifacts cover the intended datasets.

## 2. Install and run tests

```bash
uv sync --frozen
uv run pytest
```

No API credential is required. The tests cover splitting-related utilities, selectors, prediction parsing, gating, and traditional/modern baseline adapters.

## 3. Regenerate primary paper artifacts offline

```bash
uv run lba-build-artifacts
```

Inputs are read from `artifacts/results/`. Generated PDFs, PNGs, Markdown tables, and LaTeX tables are written under `artifacts/figures/` and `artifacts/tables/`.

## 4. Recompute LBA/LEQ from OpenML

This stage downloads the 14 public datasets but does not call an LLM API. Dataset IDs and split settings are in `configs/`.

```bash
uv run python -m rail.analysis.compute_leq
```

Newly computed artifacts should be written under `outputs/analysis/` and compared with the frozen release artifacts. Small differences can arise if an upstream OpenML dataset changes.

## 5. Run the primary LLM comparison

Create a private `.env` from `.env.example` and provide your own API key. The main classification setting uses `k=100` and matched-pair boundary retrieval:

```bash
uv run lba-icl \
  --dataset letter \
  --dataset-id 6 \
  --model qwen3.5-397b-a17b \
  --seed 42 \
  --k 100 \
  --max-examples 100 \
  --thinking-budget 1 \
  --selection-strategy matched_pair_boundary \
  --output outputs/letter_main.csv \
  --request-log outputs/logs/letter_main.jsonl
```

The primary regression setting uses `k=16` and balanced support:

```bash
uv run lba-icl \
  --dataset california_housing \
  --dataset-id 43939 \
  --task-type-override regression \
  --model qwen3.5-397b-a17b \
  --seed 42 \
  --k 16 \
  --max-examples 100 \
  --thinking-budget 1 \
  --selection-strategy balanced \
  --output outputs/california_housing_main.csv \
  --request-log outputs/logs/california_housing_main.jsonl
```

## 6. Main mechanism experiments

Validation-selected gating:

```bash
uv run python -m rail.experiments.run_validation_gating \
  --datasets adult credit-g diabetes phoneme vehicle segment page-blocks satimage letter \
  --model qwen3.5-397b-a17b \
  --seed 42 \
  --k 100 \
  --val-examples 50 \
  --test-examples 100 \
  --summary-output outputs/gating_summary.csv \
  --detail-output outputs/gating_detail.csv \
  --request-log outputs/logs/gating_requests.jsonl \
  --progress-log outputs/logs/gating_progress.jsonl
```

Support-label intervention:

```bash
uv run python -m rail.experiments.run_label_intervention \
  --datasets letter diabetes credit-g \
  --max-examples 50 \
  --model qwen3.5-397b-a17b \
  --seed 42 \
  --summary-output outputs/label_intervention_summary.csv \
  --detail-output outputs/label_intervention_detail.csv \
  --request-log outputs/logs/label_intervention_requests.jsonl
```

Context-size sensitivity:

```bash
uv run python -m rail.experiments.run_classification_k_sensitivity \
  --dataset letter \
  --model qwen3.5-397b-a17b \
  --output outputs/classification_k_sensitivity.csv \
  --figure-output outputs/classification_k_sensitivity.pdf \
  --request-log outputs/logs/classification_k_sensitivity.jsonl
```

### Query-level margin

To regenerate query statistics, first generate local prediction traces. These commands use the hosted API and may incur cost. Run from the repository root:

```bash
for dataset in letter credit-g diabetes page-blocks; do
  seed=42
  count=50
  if [ "$dataset" = "credit-g" ]; then seed=7; fi
  if [ "$dataset" = "diabetes" ]; then count=100; fi
  if [ "$dataset" = "letter" ]; then count=20; fi
  uv run python -m rail.analysis.run_error_analysis \
    --dataset "$dataset" --seed "$seed" --max-examples "$count" \
    --model qwen3.5-397b-a17b --k 100 \
    --strategies matched_pair_boundary \
    --output-dir outputs/error_analysis
done
uv run python -m rail.analysis.query_margin_analysis \
  --flip-root outputs/error_analysis
```

The query-margin analysis uses seed 7 for credit-g and seed 42 for the other datasets, as recorded in the released query CSV. OpenML data may be downloaded if not cached. Trace files contain query features and remain in the ignored `outputs/` directory. Use the released bucket CSV and the offline artifact builder to regenerate the figure without API calls.

## 7. Modern tabular baselines

These optional dependencies can be large and may download model weights:

```bash
uv sync --frozen --extra modern-baselines
uv run python -m rail.experiments.run_modern_tabular_baselines \
  --output outputs/modern_tabular_baselines.csv
```

## Reproducibility limits

- The primary LLM is accessed through a hosted OpenAI-compatible API and is not redistributed.
- Provider-side model revisions, routing, kernels, and decoding implementation can change.
- The paper uses fixed diagnostic slices to bound API cost, rather than large repeated full-dataset runs.
- Frozen result files are therefore the canonical record; fresh API runs validate the pipeline and qualitative regimes but may not reproduce every prediction exactly.
