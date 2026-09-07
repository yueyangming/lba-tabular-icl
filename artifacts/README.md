# Released artifacts

This directory contains only the main results, figures, and tables used to support the accepted paper.

- `results/`: frozen CSV inputs and derived summaries.
- `figures/`: primary paper figures in PDF and/or PNG form.
- `tables/`: primary paper tables in LaTeX form.

Run `python scripts/verify_release.py` to check dataset and file coverage. Run `uv run lba-build-artifacts` to regenerate the LBA, query-margin, gating, and intervention artifacts from the frozen CSV files.

Exploratory runs, API request logs, provider responses, caches, and temporary files are intentionally not part of this release.
