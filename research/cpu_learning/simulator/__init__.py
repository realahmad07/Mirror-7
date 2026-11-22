"""Public simulator data structures for Mirror 7 CPU-native learning research.

NOTE: ``hidden_actions`` is intentionally NOT imported here. It is evaluator-only
and must never be imported by candidate learners or baselines (except the evaluator's
hand-coded ceiling oracle).
"""
from __future__ import annotations

from .state_graph import (
    ActionInvocation,
    Entity,
    EvaluationCase,
    EvaluationTaskHiddenLabel,
    EvaluationTaskPublic,
    Fact,
    ObservationMask,
    SparseDelta,
    StateGraph,
    TrainingEpisode,
    canonical_json,
    canonical_json_bytes,
    compressed_bit_length,
    compute_sparse_delta,
    sha256_hex,
)

__all__ = [
    "ActionInvocation",
    "Entity",
    "EvaluationCase",
    "EvaluationTaskHiddenLabel",
    "EvaluationTaskPublic",
    "Fact",
    "ObservationMask",
    "SparseDelta",
    "StateGraph",
    "TrainingEpisode",
    "canonical_json",
    "canonical_json_bytes",
    "compressed_bit_length",
    "compute_sparse_delta",
    "sha256_hex",
]
