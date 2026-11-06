"""
Strategy Generation Module.
Proposes inspectable, algorithmic procedure candidates for problem solving.
"""
from __future__ import annotations

import textwrap
from typing import Sequence
from .schema import ProblemRepresentation, ReasoningStrategy


class StrategyGenerator:
    """Generates candidate algorithmic procedures across different algorithmic paradigms."""

    def generate_candidate_strategies(
        self, problem: ProblemRepresentation
    ) -> list[ReasoningStrategy]:
        """Propose candidate strategies based on problem domain and objective."""
        candidates: list[ReasoningStrategy] = []

        # Domain: Numeric / Search / Math
        if "prime" in problem.objective.lower():
            candidates.append(self._generate_prime_trial_division(problem))
            candidates.append(self._generate_prime_sqrt_optimized(problem))
            candidates.append(self._generate_prime_6k_optimized(problem))

        # Domain: Sequence / Search / Sorting
        elif any(k in problem.objective.lower() for k in ["search", "find", "locate"]):
            candidates.append(self._generate_linear_search(problem))
            candidates.append(self._generate_binary_search(problem))

        # Domain: Fibonacci / Sequences
        elif any(k in problem.objective.lower() for k in ["fibonacci", "sequence", "recurrence"]):
            candidates.append(self._generate_fib_recursive(problem))
            candidates.append(self._generate_fib_iterative(problem))
            candidates.append(self._generate_fib_memoized(problem))

        # General default fallback
        else:
            candidates.append(self._generate_generic_greedy(problem))

        return candidates

    def _generate_prime_trial_division(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        def is_prime(n):
            if n <= 1:
                return False
            for i in range(2, n):
                if n % i == 0:
                    return False
            return True
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_naive_trial",
            name="Trial Division (O(N))",
            procedure_code=code,
            entry_point="is_prime",
            rationale="Naive testing of every divisor up to N.",
            complexity_class="O(N)",
            tags=("naive", "brute_force"),
        )

    def _generate_prime_sqrt_optimized(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        import math

        def is_prime(n):
            if n <= 1:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            limit = int(math.isqrt(n))
            for i in range(3, limit + 1, 2):
                if n % i == 0:
                    return False
            return True
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_sqrt_optimized",
            name="Sqrt Bound with Odd Step (O(sqrt(N)))",
            procedure_code=code,
            entry_point="is_prime",
            rationale="Bounds divisor search to sqrt(N) and skips even divisors.",
            complexity_class="O(sqrt(N))",
            tags=("optimized", "number_theory"),
        )

    def _generate_prime_6k_optimized(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        import math

        def is_prime(n):
            if n <= 1:
                return False
            if n <= 3:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            limit = int(math.isqrt(n))
            i = 5
            while i <= limit:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_6k_wheel",
            name="6k +/- 1 Wheel Factorization",
            procedure_code=code,
            entry_point="is_prime",
            rationale="Eliminates multiples of 2 and 3, stepping through 6k +/- 1 up to sqrt(N).",
            complexity_class="O(sqrt(N)/3)",
            tags=("wheel_factorization", "advanced_arithmetic"),
        )

    def _generate_linear_search(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        def binary_search(arr, target):
            for i, val in enumerate(arr):
                if val == target:
                    return i
            return -1
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_linear",
            name="Linear Scan",
            procedure_code=code,
            entry_point="binary_search",
            rationale="Sequential scan over all elements.",
            complexity_class="O(N)",
            tags=("linear", "scan"),
        )

    def _generate_binary_search(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        def binary_search(arr, target):
            low = 0
            high = len(arr) - 1
            while low <= high:
                mid = (low + high) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    low = mid + 1
                else:
                    high = mid - 1
            return -1
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_bisection",
            name="Binary Search",
            procedure_code=code,
            entry_point="binary_search",
            rationale="Logarithmic bisection search on sorted sequence.",
            complexity_class="O(log N)",
            tags=("divide_and_conquer", "bisection"),
        )

    def _generate_fib_recursive(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        def fib(n):
            if n <= 0:
                return 0
            if n == 1:
                return 1
            return fib(n - 1) + fib(n - 2)
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_naive_recursive",
            name="Naive Tree Recursion",
            procedure_code=code,
            entry_point="fib",
            rationale="Direct definition tree recursion.",
            complexity_class="O(2^N)",
            tags=("naive", "tree_recursion"),
        )

    def _generate_fib_iterative(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        def fib(n):
            if n <= 0:
                return 0
            if n == 1:
                return 1
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_iterative_accum",
            name="Iterative State Accumulation",
            procedure_code=code,
            entry_point="fib",
            rationale="O(1) memory linear iteration.",
            complexity_class="O(N)",
            tags=("iterative", "dynamic_programming"),
        )

    def _generate_fib_memoized(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        def fib(n):
            memo = {0: 0, 1: 1}
            def helper(x):
                if x in memo:
                    return memo[x]
                res = helper(x - 1) + helper(x - 2)
                memo[x] = res
                return res
            return helper(n)
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_memoized",
            name="Top-Down Memoization",
            procedure_code=code,
            entry_point="fib",
            rationale="Recursive formulation augmented with a memoization table.",
            complexity_class="O(N)",
            tags=("memoization", "top_down"),
        )

    def _generate_generic_greedy(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        def solve(*args):
            if not args:
                return None
            return sorted(args)[0]
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_greedy",
            name="Greedy Extremum Selection",
            procedure_code=code,
            entry_point="solve",
            rationale="Selects minimum element as heuristic baseline.",
            complexity_class="O(N log N)",
            tags=("greedy",),
        )
