from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a file path if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_results_csv(path: str | Path, rows: Iterable[Mapping]) -> Path:
    """Write a list of result dictionaries to CSV."""
    out_path = ensure_parent_dir(path)
    frame = pd.DataFrame(list(rows))
    frame.to_csv(out_path, index=False)
    return out_path


def append_jsonl(path: str | Path, record: Mapping) -> Path:
    """Append a single record to a JSONL file."""
    out_path = ensure_parent_dir(path)
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path
