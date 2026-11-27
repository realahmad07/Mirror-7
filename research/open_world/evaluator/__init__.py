"""EVALUATOR-ONLY package holding sealed gold labels, hidden oracle, and 8-family metrics.

STRICT ISOLATION:
- ZERO modules under ``research/open_world/baselines/`` (or any future learner package)
  may ever import from ``research/open_world/evaluator/``.
"""
from __future__ import annotations

from .hidden_eval import (
    HiddenGoldRecord,
    SealedBenchmarkBundle,
    build_balanced_copa_bundle,
    build_longmemeval_bundle,
    build_offline_smoke_bundles,
    build_squad_v2_bundle,
)
from .metrics import (
    compute_benchmark_metrics,
    compute_brier_and_ece,
    compute_token_f1,
)
from .oracle import EvaluatorOnlyOracle

__all__ = [
    "EvaluatorOnlyOracle",
    "HiddenGoldRecord",
    "SealedBenchmarkBundle",
    "build_balanced_copa_bundle",
    "build_longmemeval_bundle",
    "build_offline_smoke_bundles",
    "build_squad_v2_bundle",
    "compute_benchmark_metrics",
    "compute_brier_and_ece",
    "compute_token_f1",
]
