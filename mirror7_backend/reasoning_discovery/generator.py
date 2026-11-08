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

        obj_l = problem.objective.lower()

        # Domain: Interval Merging / Range Consolidation
        if "interval" in obj_l or "merge_interval" in obj_l or ("merge" in obj_l and "overlap" in obj_l):
            candidates.append(self._generate_merge_intervals(problem))

        # Domain: Two-Sum / Pair-Sum Indexing
        elif "two_sum" in obj_l or ("two" in obj_l and "sum" in obj_l) or "target sum" in obj_l:
            candidates.append(self._generate_two_sum_pairs(problem))
            candidates.append(self._generate_two_sum_hashmap(problem))

        # Domain: LCM / Least Common Multiple
        elif "lcm" in obj_l or "least common multiple" in obj_l:
            candidates.append(self._generate_lcm(problem))

        # Domain: GCD / Greatest Common Divisor
        elif "gcd" in obj_l or "greatest common divisor" in obj_l:
            candidates.append(self._generate_gcd(problem))

        # Domain: Palindrome Checking
        elif "palindrome" in obj_l:
            candidates.append(self._generate_palindrome(problem))

        # Domain: Factorial
        elif "factorial" in obj_l:
            candidates.append(self._generate_factorial(problem))

        # Domain: Sorting
        elif "sort" in obj_l and "sorted array" not in obj_l:
            candidates.append(self._generate_mergesort(problem))

        # Domain: Numeric / Search / Math
        elif "prime" in obj_l:
            candidates.append(self._generate_prime_trial_division(problem))
            candidates.append(self._generate_prime_sqrt_optimized(problem))
            candidates.append(self._generate_prime_6k_optimized(problem))

        # Domain: Sequence / Search / Sorting
        elif any(k in obj_l for k in ["search", "find", "locate"]):
            candidates.append(self._generate_linear_search(problem))
            candidates.append(self._generate_binary_search(problem))

        # Domain: Fibonacci / Sequences
        elif any(k in obj_l for k in ["fibonacci", "sequence", "recurrence", "fib"]):
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

    def _generate_merge_intervals(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        def merge_intervals(intervals):
            if not intervals:
                return []
            ordered = sorted([list(iv) for iv in intervals], key=lambda x: (x[0], x[1]))
            merged = [ordered[0]]
            for start, end in ordered[1:]:
                if start <= merged[-1][1]:
                    if end > merged[-1][1]:
                        merged[-1][1] = end
                else:
                    merged.append([start, end])
            return merged
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_sweep_merge_intervals",
            name="Sorted Interval Sweep-Line Merge",
            procedure_code=code,
            entry_point="merge_intervals",
            rationale="Sorts intervals by start point and greedily merges overlapping ranges.",
            complexity_class="O(N log N)",
            tags=("intervals", "sweep_line", "sorting"),
        )

    def _generate_two_sum_pairs(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        def two_sum(nums, target):
            n = len(nums)
            for i in range(n):
                for j in range(i + 1, n):
                    if nums[i] + nums[j] == target:
                        return [i, j]
            return []
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_two_sum_pairs",
            name="Distinct Pair Scan (O(N^2))",
            procedure_code=code,
            entry_point="two_sum",
            rationale="Scans distinct pairs i < j checking exact target sum.",
            complexity_class="O(N^2)",
            tags=("pairs", "brute_force"),
        )

    def _generate_two_sum_hashmap(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        def two_sum(nums, target):
            seen = {}
            for i, val in enumerate(nums):
                need = target - val
                if need in seen:
                    return [seen[need], i]
                seen[val] = i
            return []
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_two_sum_hashmap",
            name="Complement Hash Index (O(N))",
            procedure_code=code,
            entry_point="two_sum",
            rationale="Stores seen values to find exact complements in single linear pass.",
            complexity_class="O(N)",
            tags=("hash_table", "linear"),
        )

    def _generate_lcm(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        import math

        def lcm(a, b):
            if a == 0 or b == 0:
                return 0
            return abs(a * b) // math.gcd(a, b)
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_lcm_euclid",
            name="Euclid GCD Reduction LCM",
            procedure_code=code,
            entry_point="lcm",
            rationale="Computes LCM(a, b) cleanly via |a*b| // gcd(a, b).",
            complexity_class="O(log min(a, b))",
            tags=("number_theory", "euclid"),
        )

    def _generate_gcd(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        def gcd(a, b):
            a = abs(a)
            b = abs(b)
            while b != 0:
                a, b = b, a % b
            return a
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_euclid_gcd",
            name="Euclidean Algorithm",
            procedure_code=code,
            entry_point="gcd",
            rationale="Repeatedly replaces (a, b) with (b, a mod b).",
            complexity_class="O(log min(a, b))",
            tags=("number_theory", "euclid"),
        )

    def _generate_palindrome(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        def is_palindrome(s):
            lo, hi = 0, len(s) - 1
            while lo < hi:
                if s[lo] != s[hi]:
                    return False
                lo += 1
                hi -= 1
            return True
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_two_pointer_palindrome",
            name="Two-Pointer Palindrome Check",
            procedure_code=code,
            entry_point="is_palindrome",
            rationale="Compares mirrored characters inward without allocating reversed copies.",
            complexity_class="O(N)",
            tags=("strings", "two_pointer"),
        )

    def _generate_factorial(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        def factorial(n):
            if n < 0:
                return 0
            acc = 1
            for k in range(2, n + 1):
                acc *= k
            return acc
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_iterative_factorial",
            name="Iterative Product Factorial",
            procedure_code=code,
            entry_point="factorial",
            rationale="Accumulates k=2..n iteratively without recursion depth limits.",
            complexity_class="O(N)",
            tags=("iterative", "combinatorics"),
        )

    def _generate_mergesort(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        def sort_list(arr):
            if len(arr) <= 1:
                return list(arr)
            mid = len(arr) // 2
            left = sort_list(arr[:mid])
            right = sort_list(arr[mid:])
            out = []
            i = j = 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    out.append(left[i])
                    i += 1
                else:
                    out.append(right[j])
                    j += 1
            out.extend(left[i:])
            out.extend(right[j:])
            return out
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_mergesort",
            name="Merge Sort (O(N log N))",
            procedure_code=code,
            entry_point="sort_list",
            rationale="Stable divide-and-conquer sorting.",
            complexity_class="O(N log N)",
            tags=("divide_and_conquer", "sorting"),
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
