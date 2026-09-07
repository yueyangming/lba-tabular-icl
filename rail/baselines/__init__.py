"""Baseline models for RAIL."""

from rail.baselines.traditional_ml import fit_baseline, run_baseline
from rail.baselines.vanilla_icl import VanillaICLRunner, run_vanilla_icl

__all__ = [
    "RetrievalICLRunner",
    "VanillaICLRunner",
    "fit_baseline",
    "run_baseline",
    "run_retrieval_icl",
    "run_vanilla_icl",
]


def __getattr__(name: str):
    if name in {"RetrievalICLRunner", "run_retrieval_icl"}:
        from rail.baselines.retrieval_icl import RetrievalICLRunner, run_retrieval_icl

        return {
            "RetrievalICLRunner": RetrievalICLRunner,
            "run_retrieval_icl": run_retrieval_icl,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
