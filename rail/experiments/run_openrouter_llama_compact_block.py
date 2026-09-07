from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / '.env')

PYTHON_BIN = ROOT / '.venv' / 'bin' / 'python'
PYTHON = str(PYTHON_BIN if PYTHON_BIN.exists() else Path(sys.executable))
DEFAULT_MODEL = 'meta-llama/llama-3.3-70b-instruct'


@dataclass(frozen=True)
class Job:
    dataset: str
    task_type_override: str | None
    seed: int
    k: int
    n: int
    selection_strategy: str
    slug: str


JOBS = [
    Job('california_housing', 'regression', 42, 16, 100, 'balanced', 'california_n100'),
    Job('kin8nm', 'regression', 42, 16, 100, 'balanced', 'kin8nm_n100'),
    Job('letter', None, 42, 100, 50, 'matched_pair_boundary', 'letter_n50'),
    Job('page-blocks', None, 42, 100, 50, 'matched_pair_boundary', 'pageblocks_n50'),
    Job('credit-g', None, 42, 100, 50, 'matched_pair_boundary', 'creditg_n50'),
]


def run_job(job: Job, *, skip_existing: bool, timeout_seconds: float, sleep_seconds: float) -> None:
    output = ROOT / 'outputs' / f'icl_smoke_openrouter_llama33_{job.slug}.csv'
    request_log = ROOT / 'outputs' / 'logs' / f'icl_requests_openrouter_llama33_{job.slug}.jsonl'
    if skip_existing and output.exists():
        print(f'[skip-existing] {output}')
        return
    cmd = [
        PYTHON,
        '-m',
        'rail.experiments.run_icl_smoke',
        '--dataset', job.dataset,
        '--model', DEFAULT_MODEL,
        '--seed', str(job.seed),
        '--k', str(job.k),
        '--max-examples', str(job.n),
        '--output', str(output),
        '--request-log', str(request_log),
        '--timeout-seconds', str(timeout_seconds),
        '--selection-strategy', job.selection_strategy,
        '--sleep-seconds', str(sleep_seconds),
    ]
    if job.task_type_override:
        cmd.extend(['--task-type-override', job.task_type_override])

    env = os.environ.copy()
    env['QWEN_PROVIDER'] = 'openrouter'
    env['OPENROUTER_DEFAULT_MODEL'] = DEFAULT_MODEL
    env['OPENROUTER_LATEST_MODEL'] = DEFAULT_MODEL
    env['OPENROUTER_DEBUG_MODEL'] = DEFAULT_MODEL

    print('\n[run]', ' '.join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the compact non-Qwen Llama 3.3 70B sanity block via OpenRouter.')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--timeout-seconds', type=float, default=180.0)
    parser.add_argument('--sleep-seconds', type=float, default=0.0)
    args = parser.parse_args()

    for job in JOBS:
        run_job(job, skip_existing=args.skip_existing, timeout_seconds=args.timeout_seconds, sleep_seconds=args.sleep_seconds)


if __name__ == '__main__':
    main()
