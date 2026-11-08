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

        # Domain: Anagram Verification
        elif "anagram" in obj_l:
            candidates.append(self._generate_anagram(problem))

        # Domain: Balanced Parentheses / Bracket Matching
        elif any(k in obj_l for k in ("parenthes", "bracket")):
            candidates.append(self._generate_valid_parentheses(problem))

        # Domain: Sieve of Eratosthenes / Primes up to N (must precede single-integer prime check)
        elif any(k in obj_l for k in ("sieve", "eratosthenes", "primes_up_to")):
            candidates.append(self._generate_sieve(problem))

        # Domain: Shortest Path / Dijkstra
        elif any(k in obj_l for k in ("dijkstra", "shortest_path", "shortest path")):
            candidates.append(self._generate_dijkstra(problem))

        # Domain: 0/1 Knapsack Dynamic Programming
        elif "knapsack" in obj_l:
            candidates.append(self._generate_knapsack_01(problem))

        # Domain: Kadane Maximum Subarray
        elif any(k in obj_l for k in ("kadane", "max_subarray", "maximum subarray")):
            candidates.append(self._generate_kadane(problem))

        # Domain: Levenshtein Edit Distance
        elif any(k in obj_l for k in ("edit_distance", "edit distance", "levenshtein")):
            candidates.append(self._generate_edit_distance(problem))

        # Domain: Longest Increasing Subsequence
        elif any(k in obj_l for k in ("lis_length", "longest increasing subsequence")):
            candidates.append(self._generate_lis(problem))

        # Domain: Topological Sort (Kahn's BFS)
        elif any(k in obj_l for k in ("topo_sort", "topological", "kahn")):
            candidates.append(self._generate_topo_sort(problem))

        # Domain: Binary Modular Exponentiation
        elif any(k in obj_l for k in ("mod_exp", "modular exponentiation", "fast_mod_exp")):
            candidates.append(self._generate_fast_mod_exp(problem))

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

    def _generate_anagram(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        def is_anagram(a, b):
            if len(a) != len(b):
                return False
            counts = {}
            for ch in a:
                counts[ch] = counts.get(ch, 0) + 1
            for ch in b:
                if ch not in counts or counts[ch] == 0:
                    return False
                counts[ch] -= 1
            return True
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_freq_anagram",
            name="Character Frequency Balance (O(N))",
            procedure_code=code,
            entry_point="is_anagram",
            rationale="Counts character frequencies in one pass and decrements to verify bijection.",
            complexity_class="O(N)",
            tags=("strings", "hash_table"),
        )

    def _generate_valid_parentheses(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        def is_valid_parentheses(s):
            pairs = {')': '(', ']': '[', '}': '{'}
            stack = []
            for ch in s:
                if ch in '([{':
                    stack.append(ch)
                elif ch in pairs:
                    if not stack or stack.pop() != pairs[ch]:
                        return False
            return len(stack) == 0
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_stack_parens",
            name="LIFO Stack Bracket Matching (O(N))",
            procedure_code=code,
            entry_point="is_valid_parentheses",
            rationale="Matches closing delimiters against the top of an explicit stack.",
            complexity_class="O(N)",
            tags=("stack", "strings"),
        )

    def _generate_sieve(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        import math

        def sieve_of_eratosthenes(n):
            if n < 2:
                return []
            prime = [True] * (n + 1)
            prime[0] = False
            prime[1] = False
            limit = math.isqrt(n)
            for p in range(2, limit + 1):
                if prime[p]:
                    for m in range(p * p, n + 1, p):
                        prime[m] = False
            return [i for i, ok in enumerate(prime) if ok]
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_eratosthenes_sieve",
            name="Sieve of Eratosthenes (O(N log log N))",
            procedure_code=code,
            entry_point="sieve_of_eratosthenes",
            rationale="Marks composite multiples starting at p^2 up to sqrt(N).",
            complexity_class="O(N log log N)",
            tags=("number_theory", "sieve"),
        )

    def _generate_dijkstra(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        import heapq

        def dijkstra_shortest_path(n, edges, src):
            adj = [[] for _ in range(n)]
            for u, v, w in edges:
                adj[u].append((v, w))
            inf = float('inf')
            dist = [inf] * n
            if 0 <= src < n:
                dist[src] = 0
            pq = [(0, src)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                for v, w in adj[u]:
                    nd = d + w
                    if nd < dist[v]:
                        dist[v] = nd
                        heapq.heappush(pq, (nd, v))
            return [-1 if d == inf else int(d) for d in dist]
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_dijkstra_heap",
            name="Dijkstra Binary Min-Heap (O(E log V))",
            procedure_code=code,
            entry_point="dijkstra_shortest_path",
            rationale="Greedy shortest-path relaxation with lazy priority queue.",
            complexity_class="O(E log V)",
            tags=("graphs", "shortest_path", "heapq"),
        )

    def _generate_knapsack_01(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        def knapsack_01(weights, values, capacity):
            if capacity <= 0 or not weights:
                return 0
            dp = [0] * (capacity + 1)
            for w, v in zip(weights, values):
                if w <= capacity:
                    for c in range(capacity, w - 1, -1):
                        cand = dp[c - w] + v
                        if cand > dp[c]:
                            dp[c] = cand
            return dp[capacity]
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_knapsack_dp",
            name="1D Rolling 0/1 Knapsack DP",
            procedure_code=code,
            entry_point="knapsack_01",
            rationale="Iterates capacity in reverse to enforce at-most-once item selection.",
            complexity_class="O(N * W)",
            tags=("dynamic_programming", "knapsack"),
        )

    def _generate_kadane(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        def kadane_max_subarray(nums):
            if not nums:
                return 0
            cur = nums[0]
            best = nums[0]
            for x in nums[1:]:
                cur = x if x > cur + x else cur + x
                if cur > best:
                    best = cur
            return best
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_kadane",
            name="Kadane Linear Subarray DP (O(N))",
            procedure_code=code,
            entry_point="kadane_max_subarray",
            rationale="Maintains best suffix sum and global maximum in one pass.",
            complexity_class="O(N)",
            tags=("dynamic_programming", "arrays"),
        )

    def _generate_edit_distance(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        def edit_distance(a, b):
            prev = list(range(len(b) + 1))
            for i, ca in enumerate(a, 1):
                curr = [i]
                for j, cb in enumerate(b, 1):
                    cost = 0 if ca == cb else 1
                    curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
                prev = curr
            return prev[-1]
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_edit_dp",
            name="Wagner-Fischer Edit Distance (O(M * N))",
            procedure_code=code,
            entry_point="edit_distance",
            rationale="Computes minimum insert/delete/replace cost with rolling DP rows.",
            complexity_class="O(M * N)",
            tags=("dynamic_programming", "strings"),
        )

    def _generate_lis(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        import bisect

        def lis_length(nums):
            tails = []
            for x in nums:
                idx = bisect.bisect_left(tails, x)
                if idx == len(tails):
                    tails.append(x)
                else:
                    tails[idx] = x
            return len(tails)
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_lis_patience",
            name="Patience Sorting LIS (O(N log N))",
            procedure_code=code,
            entry_point="lis_length",
            rationale="Maintains minimal tail per subsequence length using bisect_left.",
            complexity_class="O(N log N)",
            tags=("dynamic_programming", "bisection"),
        )

    def _generate_topo_sort(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        from collections import deque

        def topo_sort_kahn(n, edges):
            adj = [[] for _ in range(n)]
            indeg = [0] * n
            for u, v in edges:
                adj[u].append(v)
                indeg[v] += 1
            q = deque(i for i in range(n) if indeg[i] == 0)
            order = []
            while q:
                u = q.popleft()
                order.append(u)
                for v in adj[u]:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        q.append(v)
            return order if len(order) == n else []
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_kahn_topo",
            name="Kahn BFS Topological Sort (O(V + E))",
            procedure_code=code,
            entry_point="topo_sort_kahn",
            rationale="Peels zero-indegree frontier and detects cycles cleanly.",
            complexity_class="O(V + E)",
            tags=("graphs", "topological_sort"),
        )

    def _generate_fast_mod_exp(self, prob: ProblemRepresentation) -> ReasoningStrategy:
        code = textwrap.dedent("""
        def fast_mod_exp(a, b, m):
            res = 1 % m
            base = a % m
            exp = b
            while exp > 0:
                if exp & 1:
                    res = (res * base) % m
                base = (base * base) % m
                exp >>= 1
            return res
        """)
        return ReasoningStrategy(
            strategy_id=f"{prob.problem_id}_binary_mod_exp",
            name="Binary Square-and-Multiply (O(log b))",
            procedure_code=code,
            entry_point="fast_mod_exp",
            rationale="Computes a^b mod m in logarithmic multiplications without huge intermediates.",
            complexity_class="O(log b)",
            tags=("number_theory", "modular_arithmetic"),
        )
