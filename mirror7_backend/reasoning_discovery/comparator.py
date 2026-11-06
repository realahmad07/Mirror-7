"""
Strategy Comparator Module.
Compares candidate strategies against established baselines and teacher methods.
"""
from __future__ import annotations

from typing import Sequence
from .schema import ExecutionResult, ReasoningStrategy, StrategyComparison, TestCase
from .sandbox import SafeSandbox


class StrategyComparator:
    """Compares candidate strategies against baselines across correctness, latency, and resource usage."""

    def __init__(self, sandbox: SafeSandbox | None = None):
        self.sandbox = sandbox or SafeSandbox()

    def compare(
        self,
        candidate: ReasoningStrategy,
        baseline: ReasoningStrategy,
        test_cases: Sequence[TestCase],
    ) -> StrategyComparison:
        cand_res = self.sandbox.execute(candidate, test_cases)
        base_res = self.sandbox.execute(baseline, test_cases)

        cand_acc = cand_res.passed_tests / max(1, cand_res.total_tests)
        base_acc = base_res.passed_tests / max(1, base_res.total_tests)
        acc_delta = cand_acc - base_acc

        latency_ratio = 1.0
        if base_res.execution_time_ms > 0:
            latency_ratio = cand_res.execution_time_ms / base_res.execution_time_ms

        # Candidate is strictly superior if it is at least as accurate, and either strictly more accurate OR faster
        is_superior = False
        if cand_res.is_correct and not base_res.is_correct:
            is_superior = True
        elif cand_res.is_correct and base_res.is_correct:
            is_superior = latency_ratio < 0.90  # At least 10% faster

        summary = (
            f"Candidate '{candidate.name}' ({cand_acc*100:.1f}%, {cand_res.execution_time_ms:.2f}ms) vs "
            f"Baseline '{baseline.name}' ({base_acc*100:.1f}%, {base_res.execution_time_ms:.2f}ms). "
            f"Delta Acc: {acc_delta:+.2f}, Speed Ratio: {latency_ratio:.2f}x. "
            f"Superior: {is_superior}."
        )

        return StrategyComparison(
            candidate_id=candidate.strategy_id,
            baseline_id=baseline.strategy_id,
            accuracy_delta=round(acc_delta, 4),
            latency_ratio=round(latency_ratio, 3),
            is_strictly_superior=is_superior,
            summary=summary,
        )
