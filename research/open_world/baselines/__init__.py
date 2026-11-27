"""Strictly isolated Stage S11.0 non-oracle baseline solvers.

ISOLATION CONTRACT:
- ZERO imports from ``research.open_world.evaluator`` (no ``hidden_eval``, no ``oracle``,
  no ``HiddenGoldRecord``).
- ZERO GPU, ZERO pretrained LLMs, ZERO external AI APIs, ZERO neural-network training.
- ZERO modifications to production ``.dcrystal`` vaults.
"""
from __future__ import annotations

from .solvers import (
    ExistingMirrorSystemBaseline,
    ExistingRuleBlueprintBaseline,
    RetrievalOnlyBaseline,
    SimpleSymbolicBaseline,
)

__all__ = [
    "ExistingMirrorSystemBaseline",
    "ExistingRuleBlueprintBaseline",
    "RetrievalOnlyBaseline",
    "SimpleSymbolicBaseline",
]
