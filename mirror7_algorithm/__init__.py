"""
Mirror 7 — Adaptive Predictive Memory (APM)
============================================

A general-purpose sequential prediction engine that learns contextual
patterns from observation streams, makes probabilistic predictions with
calibrated uncertainty, and adapts through surprise-driven memory management.

Quickstart::

    from mirror7_algorithm import AdaptivePredictiveMemory

    apm = AdaptivePredictiveMemory()
    for symbol in "ABCABCABC":
        result = apm.observe(symbol)

    predictions = apm.predict(n_steps=3)
    print(predictions[0].best_prediction)  # → 'A'
"""

from mirror7_algorithm.mirror7_algorithm import (
    AdaptivePredictiveMemory,
    PredictionResult,
    Pattern,
    MemoryStats,
)

__all__ = [
    "AdaptivePredictiveMemory",
    "PredictionResult",
    "Pattern",
    "MemoryStats",
]

__version__ = "0.1.0"
