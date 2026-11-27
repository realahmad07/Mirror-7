"""
Mirror 7 Independent Evaluation and Teacher Disconnection Verification Suite.

The lightweight benchmark definitions are imported eagerly. Model-backed
evaluation and teacher-disconnection helpers are loaded lazily so importing
static prompts does not require PyTorch.
"""
from __future__ import annotations

from .dimensions import EvaluationDimension, ALL_EVAL_DIMENSIONS, DimensionScore
from .benchmark_dataset import EvalBenchmarkExample, load_evaluation_benchmark

__all__ = [
    "EvaluationDimension",
    "ALL_EVAL_DIMENSIONS",
    "DimensionScore",
    "EvalBenchmarkExample",
    "load_evaluation_benchmark",
    "IndependentEvaluator",
    "ComprehensiveEvaluationReport",
    "TeacherDisconnectionVerifier",
    "DisconnectionAuditResult",
]


def __getattr__(name: str):
    if name in {"IndependentEvaluator", "ComprehensiveEvaluationReport"}:
        from .evaluator import IndependentEvaluator, ComprehensiveEvaluationReport

        return {
            "IndependentEvaluator": IndependentEvaluator,
            "ComprehensiveEvaluationReport": ComprehensiveEvaluationReport,
        }[name]
    if name in {"TeacherDisconnectionVerifier", "DisconnectionAuditResult"}:
        from .disconnection_test import TeacherDisconnectionVerifier, DisconnectionAuditResult

        return {
            "TeacherDisconnectionVerifier": TeacherDisconnectionVerifier,
            "DisconnectionAuditResult": DisconnectionAuditResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
