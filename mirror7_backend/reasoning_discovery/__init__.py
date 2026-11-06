"""
Mirror 7 Reasoning Discovery Engine.

Autonomous framework for formulating, synthesizing, testing, and refining
algorithmic problem-solving strategies in a sandboxed environment.
"""

from .schema import (
    ProblemRepresentation,
    ReasoningStrategy,
    ExecutionResult,
    StrategyComparison,
    NoveltyReport,
    TestCase,
)
from .explorer import ProblemExplorer
from .generator import StrategyGenerator
from .sandbox import SafeSandbox
from .comparator import StrategyComparator
from .memory import StrategyMemory
from .refiner import StrategyRefiner
from .novelty import NoveltyEvaluator
from .engine import ReasoningDiscoveryEngine

__all__ = [
    "ProblemRepresentation",
    "ReasoningStrategy",
    "ExecutionResult",
    "StrategyComparison",
    "NoveltyReport",
    "TestCase",
    "ProblemExplorer",
    "StrategyGenerator",
    "SafeSandbox",
    "StrategyComparator",
    "StrategyMemory",
    "StrategyRefiner",
    "NoveltyEvaluator",
    "ReasoningDiscoveryEngine",
]
