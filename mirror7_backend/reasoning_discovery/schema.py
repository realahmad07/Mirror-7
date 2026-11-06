"""
Data contracts for Reasoning Discovery Engine.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TestCase:
    __test__ = False
    input_args: tuple[Any, ...]
    expected_output: Any
    description: str = ""


@dataclass(frozen=True)
class ProblemRepresentation:
    problem_id: str
    objective: str
    knowns: dict[str, Any]
    constraints: tuple[str, ...]
    assumptions: tuple[str, ...]
    domain: str
    test_cases: tuple[TestCase, ...] = ()

    def fingerprint(self) -> str:
        payload = {
            "id": self.problem_id,
            "objective": self.objective,
            "knowns": self.knowns,
            "constraints": list(self.constraints),
            "assumptions": list(self.assumptions),
            "domain": self.domain,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class ReasoningStrategy:
    strategy_id: str
    name: str
    procedure_code: str
    entry_point: str
    rationale: str
    version: int = 1
    parent_strategy_id: str | None = None
    complexity_class: str = "unknown"
    tags: tuple[str, ...] = ()

    def code_hash(self) -> str:
        return hashlib.sha256(self.procedure_code.strip().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExecutionResult:
    passed_tests: int
    total_tests: int
    execution_time_ms: float
    peak_memory_kb: float
    is_correct: bool
    error_message: str | None = None
    test_outputs: tuple[Any, ...] = ()


@dataclass(frozen=True)
class StrategyComparison:
    candidate_id: str
    baseline_id: str
    accuracy_delta: float
    latency_ratio: float  # < 1.0 means candidate is faster
    is_strictly_superior: bool
    summary: str


@dataclass(frozen=True)
class NoveltyReport:
    strategy_id: str
    is_novel: bool
    structural_similarity: float  # 0.0 to 1.0 (1.0 = identical AST)
    algorithmic_differences: tuple[str, ...]
    generalization_score: float  # Accuracy on unseen test cases
    confidence: float
