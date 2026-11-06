"""
Reasoning Discovery Engine Orchestrator.
Coordinates problem exploration, strategy synthesis, sandboxed verification,
comparison, novelty evaluation, and persistent memory retention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .schema import (
    ExecutionResult,
    NoveltyReport,
    ProblemRepresentation,
    ReasoningStrategy,
    StrategyComparison,
    TestCase,
)
from .explorer import ProblemExplorer
from .generator import StrategyGenerator
from .sandbox import SafeSandbox
from .comparator import StrategyComparator
from .memory import StrategyMemory
from .refiner import StrategyRefiner
from .novelty import NoveltyEvaluator


@dataclass
class DiscoveryCycleResult:
    problem: ProblemRepresentation
    evaluated_strategies: list[ReasoningStrategy] = field(default_factory=list)
    execution_results: dict[str, ExecutionResult] = field(default_factory=dict)
    comparisons: list[StrategyComparison] = field(default_factory=list)
    novelty_reports: dict[str, NoveltyReport] = field(default_factory=dict)
    accepted_strategy: ReasoningStrategy | None = None
    best_execution: ExecutionResult | None = None


class ReasoningDiscoveryEngine:
    """Integrated engine for discovering, testing, and refining algorithmic reasoning procedures."""

    def __init__(
        self,
        memory: StrategyMemory | None = None,
        sandbox: SafeSandbox | None = None,
    ):
        self.sandbox = sandbox or SafeSandbox()
        self.memory = memory or StrategyMemory()
        self.explorer = ProblemExplorer()
        self.generator = StrategyGenerator()
        self.comparator = StrategyComparator(sandbox=self.sandbox)
        self.refiner = StrategyRefiner(sandbox=self.sandbox)
        self.novelty = NoveltyEvaluator(sandbox=self.sandbox)

    def run_discovery_cycle(
        self,
        problem: ProblemRepresentation,
        held_out_tests: Sequence[TestCase] | None = None,
    ) -> DiscoveryCycleResult:
        result = DiscoveryCycleResult(problem=problem)

        # 1. Propose candidate strategies
        candidates = self.generator.generate_candidate_strategies(problem)
        result.evaluated_strategies = list(candidates)

        # 2. Test each candidate in the safety sandbox
        valid_candidates: list[tuple[ReasoningStrategy, ExecutionResult]] = []
        for cand in candidates:
            exec_res = self.sandbox.execute(cand, problem.test_cases)
            result.execution_results[cand.strategy_id] = exec_res
            if exec_res.is_correct:
                valid_candidates.append((cand, exec_res))

        if not valid_candidates:
            return result

        # 3. Pairwise comparison against baseline (first candidate is designated baseline)
        baseline_strat, baseline_exec = valid_candidates[0]
        best_candidate = baseline_strat
        best_exec = baseline_exec

        for cand, exec_res in valid_candidates[1:]:
            comp = self.comparator.compare(cand, baseline_strat, problem.test_cases)
            result.comparisons.append(comp)
            if comp.is_strictly_superior:
                best_candidate = cand
                best_exec = exec_res

        # 4. Evaluate novelty and generalization
        known_strats = self.memory.find_by_domain(problem.domain)
        held_out = held_out_tests or problem.test_cases
        novelty_report = self.novelty.evaluate_novelty(
            best_candidate, known_strats, held_out
        )
        result.novelty_reports[best_candidate.strategy_id] = novelty_report

        # 5. Store accepted strategy in memory
        result.accepted_strategy = best_candidate
        result.best_execution = best_exec
        self.memory.store_strategy(
            best_candidate,
            domain=problem.domain,
            execution_result=best_exec,
            limitations=[],
        )

        return result
