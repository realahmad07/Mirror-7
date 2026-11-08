"""
Expand Mirror 7 rule vault with verified frontier algorithms, math, and reasoning primitives.
Runs fully on CPU with zero GPU / zero Colab requirements, verifying each rule with ZeroTrustSandboxJudge.
"""
import time
from pathlib import Path
from mirror_crystal.disk_vault import DiskCrystalVault
from mirror_crystal.grammar_and_sandbox import ZeroTrustSandboxJudge

FRONTIER_PRIMITIVES = [
    {
        "prompt": "Write a python function for the Knuth-Morris-Pratt (KMP) string matching algorithm.",
        "response": '''def kmp_search(text: str, pattern: str) -> list[int]:
    """Knuth-Morris-Pratt (KMP) string search algorithm returning all start indices."""
    if not pattern or not text or len(pattern) > len(text):
        return []
    # Build longest proper prefix-suffix (LPS) table
    lps = [0] * len(pattern)
    length = 0
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    # Search phase
    matches = []
    i = j = 0
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == len(pattern):
            matches.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return matches
''',
        "test_code": '''
assert kmp_search("ABABDABACDABABCABAB", "ABABCABAB") == [10]
assert kmp_search("AAAAABAAABA", "AAAA") == [0, 1]
assert kmp_search("hello world", "xyz") == []
assert kmp_search("", "a") == []
''',
        "domain": "algorithms",
        "trust_tier": 1,
    },
    {
        "prompt": "Implement a Trie (Prefix Tree) data structure in Python with insert, search, and startsWith.",
        "response": '''class TrieNode:
    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.is_end_of_word: bool = False

class Trie:
    """Standard Prefix Tree (Trie) for efficient string retrieval."""
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
''',
        "test_code": '''
t = Trie()
t.insert("apple")
assert t.search("apple") is True
assert t.search("app") is False
assert t.starts_with("app") is True
t.insert("app")
assert t.search("app") is True
assert t.starts_with("banana") is False
''',
        "domain": "algorithms",
        "trust_tier": 1,
    },
    {
        "prompt": "Implement an LRU Cache in Python using a doubly linked list and hash map.",
        "response": '''class DLinkedNode:
    def __init__(self, key: int = 0, value: int = 0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    """Least Recently Used (LRU) Cache with O(1) get and put operations."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: dict[int, DLinkedNode] = {}
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_node(self, node: DLinkedNode) -> None:
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: DLinkedNode) -> None:
        prev = node.prev
        nxt = node.next
        prev.next = nxt
        nxt.prev = prev

    def _move_to_head(self, node: DLinkedNode) -> None:
        self._remove_node(node)
        self._add_node(node)

    def _pop_tail(self) -> DLinkedNode:
        res = self.tail.prev
        self._remove_node(res)
        return res

    def get(self, key: int) -> int:
        node = self.cache.get(key)
        if not node:
            return -1
        self._move_to_head(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        node = self.cache.get(key)
        if not node:
            new_node = DLinkedNode(key, value)
            self.cache[key] = new_node
            self._add_node(new_node)
            if len(self.cache) > self.capacity:
                tail = self._pop_tail()
                del self.cache[tail.key]
        else:
            node.value = value
            self._move_to_head(node)
''',
        "test_code": '''
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
assert cache.get(1) == 1
cache.put(3, 3) # evicts key 2
assert cache.get(2) == -1
cache.put(4, 4) # evicts key 1
assert cache.get(1) == -1
assert cache.get(3) == 3
assert cache.get(4) == 4
''',
        "domain": "algorithms",
        "trust_tier": 1,
        "aliases": [
            "Write a python LRUCache class with get and put methods and O(1) time complexity.",
            "Implement an LRUCache class in Python.",
        ],
    },
    {
        "prompt": "Implement Disjoint Set Union (Union-Find) with path compression and union by rank in Python.",
        "response": '''class UnionFind:
    """Disjoint Set Union (DSU) data structure with path compression and rank optimization."""
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size
        self.num_components = size

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        self.num_components -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)
''',
        "test_code": '''
uf = UnionFind(5)
assert uf.connected(0, 1) is False
uf.union(0, 1)
uf.union(1, 2)
assert uf.connected(0, 2) is True
assert uf.connected(0, 3) is False
assert uf.num_components == 3
''',
        "domain": "algorithms",
        "trust_tier": 1,
    },
    {
        "prompt": "Write a python function to solve the LeetCode Trapping Rain Water problem in O(N) time and O(1) space.",
        "response": '''def trap_rain_water(height: list[int]) -> int:
    """Calculate total units of trapped rain water between elevation bars."""
    if not height or len(height) < 3:
        return 0
    left = 0
    right = len(height) - 1
    left_max = height[left]
    right_max = height[right]
    water = 0
    while left < right:
        if left_max <= right_max:
            left += 1
            left_max = max(left_max, height[left])
            water += max(0, left_max - height[left])
        else:
            right -= 1
            right_max = max(right_max, height[right])
            water += max(0, right_max - height[right])
    return water
''',
        "test_code": '''
assert trap_rain_water([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
assert trap_rain_water([4, 2, 0, 3, 2, 5]) == 9
assert trap_rain_water([]) == 0
assert trap_rain_water([1, 2, 3]) == 0
''',
        "domain": "algorithms",
        "trust_tier": 1,
        "aliases": [
            "Write a python function to solve the trapping rain water problem.",
            "Write a python program for trapping rain water.",
        ],
    },
    {
        "prompt": "Write a python function to find the median of two sorted arrays in O(log(min(m, n))) time.",
        "response": '''def find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float:
    """Find the median of two sorted arrays in logarithmic time complexity."""
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    m, n = len(nums1), len(nums2)
    imin, imax, half_len = 0, m, (m + n + 1) // 2
    while imin <= imax:
        i = (imin + imax) // 2
        j = half_len - i
        if i < m and nums2[j - 1] > nums1[i]:
            imin = i + 1
        elif i > 0 and nums1[i - 1] > nums2[j]:
            imax = i - 1
        else:
            if i == 0: max_left = nums2[j - 1]
            elif j == 0: max_left = nums1[i - 1]
            else: max_left = max(nums1[i - 1], nums2[j - 1])
            if (m + n) % 2 == 1:
                return float(max_left)
            if i == m: min_right = nums2[j]
            elif j == n: min_right = nums1[i]
            else: min_right = min(nums1[i], nums2[j])
            return (max_left + min_right) / 2.0
    return 0.0
''',
        "test_code": '''
assert find_median_sorted_arrays([1, 3], [2]) == 2.0
assert find_median_sorted_arrays([1, 2], [3, 4]) == 2.5
assert find_median_sorted_arrays([0, 0], [0, 0]) == 0.0
assert find_median_sorted_arrays([], [1]) == 1.0
''',
        "domain": "algorithms",
        "trust_tier": 1,
    },
    {
        "prompt": "Write a python function for the Extended Euclidean Algorithm to solve ax + by = gcd(a, b).",
        "response": '''def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclidean algorithm returning (gcd, x, y) such that a*x + b*y = gcd(a, b)."""
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y
''',
        "test_code": '''
g, x, y = extended_gcd(30, 20)
assert g == 10
assert 30 * x + 20 * y == 10
g2, x2, y2 = extended_gcd(35, 15)
assert g2 == 5
assert 35 * x2 + 15 * y2 == 5
''',
        "domain": "olympiad_math",
        "trust_tier": 1,
    },
    {
        "prompt": "Write a python function for the Chinese Remainder Theorem to find x = r[i] mod m[i].",
        "response": '''def chinese_remainder_theorem(remainders: list[int], moduli: list[int]) -> int:
    """Solve system of simultaneous congruences using the Chinese Remainder Theorem."""
    def ext_gcd(a: int, b: int) -> tuple[int, int, int]:
        if a == 0: return b, 0, 1
        g, x1, y1 = ext_gcd(b % a, a)
        return g, y1 - (b // a) * x1, x1

    total_prod = 1
    for m in moduli:
        total_prod *= m

    result = 0
    for r, m in zip(remainders, moduli):
        partial_prod = total_prod // m
        _, inv, _ = ext_gcd(partial_prod, m)
        inv = (inv % m + m) % m
        result = (result + r * partial_prod * inv) % total_prod
    return result
''',
        "test_code": '''
assert chinese_remainder_theorem([2, 3, 2], [3, 5, 7]) == 23
assert 23 % 3 == 2
assert 23 % 5 == 3
assert 23 % 7 == 2
''',
        "domain": "olympiad_math",
        "trust_tier": 1,
    },
    {
        "prompt": "Write a python function to compute Euler's totient function phi(n).",
        "response": '''def totient_phi(n: int) -> int:
    """Calculate Euler's totient phi(n), counting positive integers up to n coprime to n."""
    if n <= 0:
        return 0
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result
''',
        "test_code": '''
assert totient_phi(1) == 1
assert totient_phi(9) == 6
assert totient_phi(10) == 4
assert totient_phi(13) == 12
assert totient_phi(36) == 12
''',
        "domain": "olympiad_math",
        "trust_tier": 1,
    },
    {
        "prompt": "Write a python function to compute the determinant of a square matrix using Gaussian elimination.",
        "response": '''def matrix_determinant(matrix: list[list[float]]) -> float:
    """Compute the determinant of an n x n matrix using Gaussian elimination with partial pivoting."""
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        return 0.0
    a = [list(row) for row in matrix]
    det = 1.0
    for col in range(n):
        # Pivot selection
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return 0.0
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            det = -det
        det *= a[col][col]
        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]
            for c in range(col + 1, n):
                a[row][c] -= factor * a[col][c]
    return round(det, 6)
''',
        "test_code": '''
assert matrix_determinant([[1, 2], [3, 4]]) == -2.0
assert matrix_determinant([[2, 0, -1], [0, 5, 6], [0, 0, 3]]) == 30.0
assert matrix_determinant([[1, 2], [2, 4]]) == 0.0
''',
        "domain": "olympiad_math",
        "trust_tier": 1,
    },
    {
        "prompt": "Write a python function for the 4th-order Runge-Kutta (RK4) numerical ODE solver.",
        "response": '''def rk4_solve(f, y0: float, t0: float, t_end: float, steps: int) -> tuple[list[float], list[float]]:
    """Solve 1D ordinary differential equation dy/dt = f(t, y) using classical RK4 method."""
    h = (t_end - t0) / steps
    t_vals = [t0]
    y_vals = [y0]
    t, y = t0, y0
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        y += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        t_vals.append(round(t, 6))
        y_vals.append(round(y, 6))
    return t_vals, y_vals
''',
        "test_code": '''
# dy/dt = y, y(0) = 1 -> y(t) = exp(t)
ts, ys = rk4_solve(lambda t, y: y, 1.0, 0.0, 1.0, 100)
import math
assert abs(ys[-1] - math.e) < 1e-4
''',
        "domain": "physics_science",
        "trust_tier": 1,
    },
    {
        "prompt": "Write a python function to compute Coin Change minimum coins using dynamic programming.",
        "response": '''def coin_change_min(coins: list[int], amount: int) -> int:
    """Find fewest number of coins needed to make up amount, or -1 if impossible."""
    if amount == 0:
        return 0
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for x in range(coin, amount + 1):
            if dp[x - coin] + 1 < dp[x]:
                dp[x] = dp[x - coin] + 1
    return dp[amount] if dp[amount] != float('inf') else -1
''',
        "test_code": '''
assert coin_change_min([1, 2, 5], 11) == 3
assert coin_change_min([2], 3) == -1
assert coin_change_min([1], 0) == 0
''',
        "domain": "algorithms",
        "trust_tier": 1,
    },
    {
        "prompt": "Write a python function to compute Longest Common Subsequence (LCS) length of two strings.",
        "response": '''def longest_common_subsequence(text1: str, text2: str) -> int:
    """Calculate the length of the longest common subsequence of two strings."""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
''',
        "test_code": '''
assert longest_common_subsequence("abcde", "ace") == 3
assert longest_common_subsequence("abc", "abc") == 3
assert longest_common_subsequence("abc", "def") == 0
''',
        "domain": "algorithms",
        "trust_tier": 1,
    },
    {
        "prompt": "Write a python function for the Quickselect algorithm to find the kth smallest element in O(N) average time.",
        "response": '''def quickselect_kth_smallest(nums: list[int], k: int) -> int:
    """Find the kth smallest element (1-indexed) in an unsorted list in O(N) average time."""
    if not (1 <= k <= len(nums)):
        raise ValueError("k out of bounds")
    arr = list(nums)

    def partition(left: int, right: int, pivot_idx: int) -> int:
        pivot_val = arr[pivot_idx]
        arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]
        store_idx = left
        for i in range(left, right):
            if arr[i] < pivot_val:
                arr[store_idx], arr[i] = arr[i], arr[store_idx]
                store_idx += 1
        arr[right], arr[store_idx] = arr[store_idx], arr[right]
        return store_idx

    target = k - 1
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        p_idx = partition(left, right, mid)
        if p_idx == target:
            return arr[p_idx]
        elif p_idx < target:
            left = p_idx + 1
        else:
            right = p_idx - 1
    return arr[target]
''',
        "test_code": '''
assert quickselect_kth_smallest([3, 2, 1, 5, 6, 4], 2) == 2
assert quickselect_kth_smallest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 3
assert quickselect_kth_smallest([7], 1) == 7
''',
        "domain": "algorithms",
        "trust_tier": 1,
    },
    {
        "prompt": "Write a python function for Sliding Window Maximum using a monotonic deque.",
        "response": '''from collections import deque

def max_sliding_window(nums: list[int], k: int) -> list[int]:
    """Find the maximum element in each sliding window of size k in O(N) time."""
    if not nums or k <= 0:
        return []
    if k == 1:
        return list(nums)
    q: deque[int] = deque()
    result = []
    for i, x in enumerate(nums):
        # Remove elements outside current window
        if q and q[0] <= i - k:
            q.popleft()
        # Maintain monotonic decreasing order
        while q and nums[q[-1]] <= x:
            q.pop()
        q.append(i)
        if i >= k - 1:
            result.append(nums[q[0]])
    return result
''',
        "test_code": '''
assert max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
assert max_sliding_window([1], 1) == [1]
assert max_sliding_window([1, -1], 1) == [1, -1]
''',
        "domain": "algorithms",
        "trust_tier": 1,
    },
    {
        "prompt": "Write a python function to find the next greater element for each element in an array.",
        "response": '''def next_greater_elements(nums: list[int]) -> list[int]:
    """Find the next greater element for each element in nums using a monotonic stack in O(N)."""
    if not nums:
        return []
    res = [-1] * len(nums)
    stack: list[int] = []
    for i, x in enumerate(nums):
        while stack and nums[stack[-1]] < x:
            prev_idx = stack.pop()
            res[prev_idx] = x
        stack.append(i)
    return res
''',
        "test_code": '''
assert next_greater_elements([2, 1, 2, 4, 3]) == [4, 2, 4, -1, -1]
assert next_greater_elements([4, 3, 2, 1]) == [-1, -1, -1, -1]
assert next_greater_elements([1, 2, 3, 4]) == [2, 3, 4, -1]
assert next_greater_elements([]) == []
''',
        "domain": "algorithms",
        "trust_tier": 1,
        "aliases": [
            "Next greater element using monotonic stack in Python.",
            "Find the next greater element for every item in an array in Python.",
        ],
    },
    {
        "prompt": "Write a python function to implement a Fenwick tree (Binary Indexed Tree) with update and prefix_sum.",
        "response": '''class FenwickTree:
    """Binary Indexed Tree (Fenwick Tree) for O(log N) prefix sums and point updates."""
    def __init__(self, size: int):
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, i: int, delta: int) -> None:
        """Add delta to element at 1-based index i."""
        while i <= self.size:
            self.tree[i] += delta
            i += i & (-i)

    def prefix_sum(self, i: int) -> int:
        """Compute cumulative sum from index 1 to i."""
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def range_sum(self, left: int, right: int) -> int:
        """Compute sum in 1-based range [left, right]."""
        if left > right:
            return 0
        return self.prefix_sum(right) - self.prefix_sum(left - 1)
''',
        "test_code": '''
ft = FenwickTree(6)
ft.update(1, 3)
ft.update(2, 2)
ft.update(3, -1)
ft.update(4, 6)
ft.update(5, 5)
ft.update(6, 4)
assert ft.prefix_sum(3) == 4
assert ft.range_sum(2, 5) == 12
assert ft.range_sum(1, 6) == 19
''',
        "domain": "algorithms",
        "trust_tier": 1,
        "aliases": [
            "Implement a Binary Indexed Tree (Fenwick Tree) in Python.",
            "Write a python class for Fenwick Tree with update and prefix sum.",
        ],
    },
    {
        "prompt": "Write a python function to compute the Segment Tree with point update and range sum query.",
        "response": '''class SegmentTree:
    """Segment Tree for efficient O(log N) range sum queries and point updates."""
    def __init__(self, nums: list[int]):
        self.n = len(nums)
        self.tree = [0] * (4 * max(1, self.n))
        if self.n > 0:
            self._build(nums, 0, 0, self.n - 1)

    def _build(self, nums: list[int], node: int, start: int, end: int) -> None:
        if start == end:
            self.tree[node] = nums[start]
            return
        mid = (start + end) // 2
        self._build(nums, 2 * node + 1, start, mid)
        self._build(nums, 2 * node + 2, mid + 1, end)
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def update(self, idx: int, val: int) -> None:
        """Update value at index idx to val."""
        def _update(node: int, start: int, end: int):
            if start == end:
                self.tree[node] = val
                return
            mid = (start + end) // 2
            if start <= idx <= mid:
                _update(2 * node + 1, start, mid)
            else:
                _update(2 * node + 2, mid + 1, end)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
        if 0 <= idx < self.n:
            _update(0, 0, self.n - 1)

    def query(self, left: int, right: int) -> int:
        """Query sum in range [left, right] inclusive."""
        def _query(node: int, start: int, end: int, l: int, r: int) -> int:
            if r < start or end < l:
                return 0
            if l <= start and end <= r:
                return self.tree[node]
            mid = (start + end) // 2
            return _query(2 * node + 1, start, mid, l, r) + _query(2 * node + 2, mid + 1, end, l, r)
        if self.n == 0 or left > right:
            return 0
        return _query(0, 0, self.n - 1, left, right)
''',
        "test_code": '''
st = SegmentTree([1, 3, 5, 7, 9, 11])
assert st.query(1, 3) == 15
st.update(1, 10)
assert st.query(1, 3) == 22
assert st.query(0, 5) == 43
''',
        "domain": "algorithms",
        "trust_tier": 1,
        "aliases": [
            "Implement a Segment Tree in Python with range query and point update.",
            "Write a python SegmentTree class for range sum query.",
        ],
    },
    {
        "prompt": "Write a python function to implement Manachers algorithm for finding the longest palindromic substring in ON time.",
        "response": '''def manacher_longest_palindrome(s: str) -> str:
    """Find the longest palindromic substring in O(N) linear time using Manacher's algorithm."""
    if not s:
        return ""
    t = "^#" + "#".join(s) + "#$"
    p = [0] * len(t)
    c = 0
    r = 0
    for i in range(1, len(t) - 1):
        i_mirror = 2 * c - i
        if r > i:
            p[i] = min(r - i, p[i_mirror])
        else:
            p[i] = 0
        while t[i + 1 + p[i]] == t[i - 1 - p[i]]:
            p[i] += 1
        if i + p[i] > r:
            c = i
            r = i + p[i]

    max_len = 0
    center_index = 0
    for i in range(1, len(t) - 1):
        if p[i] > max_len:
            max_len = p[i]
            center_index = i

    start = (center_index - 1 - max_len) // 2
    return s[start:start + max_len]
''',
        "test_code": '''
assert manacher_longest_palindrome("babad") in ("bab", "aba")
assert manacher_longest_palindrome("cbbd") == "bb"
assert manacher_longest_palindrome("racecar") == "racecar"
assert manacher_longest_palindrome("a") == "a"
assert manacher_longest_palindrome("") == ""
''',
        "domain": "algorithms",
        "trust_tier": 1,
        "aliases": [
            "Implement Manacher's algorithm for longest palindromic substring in Python.",
            "Longest palindromic substring in linear O(N) time with Manacher algorithm.",
        ],
    },
    {
        "prompt": "Write a python function for Cooley-Tukey Fast Fourier Transform (FFT).",
        "response": '''import cmath

def fft(x: list[complex]) -> list[complex]:
    """Cooley-Tukey Radix-2 Fast Fourier Transform (FFT) in O(N log N) time."""
    n = len(x)
    if n <= 1:
        return list(x)
    if (n & (n - 1)) != 0:
        next_pow2 = 1 << (n - 1).bit_length()
        x = list(x) + [0j] * (next_pow2 - n)
        n = next_pow2

    even = fft(x[0::2])
    odd = fft(x[1::2])
    factor = [cmath.exp(-2j * cmath.pi * k / n) for k in range(n // 2)]
    return [even[k] + factor[k] * odd[k] for k in range(n // 2)] + \\
           [even[k] - factor[k] * odd[k] for k in range(n // 2)]

def ifft(x: list[complex]) -> list[complex]:
    """Inverse Fast Fourier Transform."""
    n = len(x)
    x_conj = [complex(val).conjugate() for val in x]
    transformed = fft(x_conj)
    return [val.conjugate() / n for val in transformed]
''',
        "test_code": '''
signal = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
spectrum = fft(signal)
recovered = ifft(spectrum)
for orig, rec in zip(signal, recovered):
    assert abs(orig - rec.real) < 1e-4
''',
        "domain": "algorithms",
        "trust_tier": 1,
        "aliases": [
            "Cooley-Tukey FFT algorithm implementation in Python.",
            "Fast Fourier Transform and Inverse FFT in pure Python.",
        ],
    },
    {
        "prompt": "Write a python function for numerical integration using Simpsons 1/3 rule.",
        "response": '''import math
import typing

def simpson_integration(f: typing.Callable[[float], float], a: float, b: float, n: int = 100) -> float:
    """Approximate the definite integral of f from a to b using Simpson's 1/3 rule with n subintervals."""
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n, 2):
        s += 4.0 * f(a + i * h)
    for i in range(2, n, 2):
        s += 2.0 * f(a + i * h)
    return s * (h / 3.0)
''',
        "test_code": '''
val1 = simpson_integration(lambda x: x * x, 0.0, 1.0, 100)
assert abs(val1 - (1.0 / 3.0)) < 1e-4
val2 = simpson_integration(math.sin, 0.0, math.pi, 100)
assert abs(val2 - 2.0) < 1e-4
''',
        "domain": "algorithms",
        "trust_tier": 1,
        "aliases": [
            "Numerical integration with Simpson's rule in Python.",
            "Simpson 1/3 rule function for definite integrals in Python.",
        ],
    },
    {
        "prompt": "Write a python function to implement topological sort on a directed acyclic graph (DAG).",
        "response": '''from collections import deque

def kahn_topological_sort(num_nodes: int, edges: list[tuple[int, int]]) -> list[int]:
    """Perform topological sort on a DAG using Kahn's algorithm (in-degree BFS). Returns [] if cycle exists."""
    in_degree = [0] * num_nodes
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].append(v)
        in_degree[v] += 1

    q: deque[int] = deque([i for i in range(num_nodes) if in_degree[i] == 0])
    order = []
    while q:
        node = q.popleft()
        order.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                q.append(neighbor)

    if len(order) != num_nodes:
        return []
    return order
''',
        "test_code": '''
edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
res = kahn_topological_sort(4, edges)
assert res.index(0) < res.index(1) and res.index(0) < res.index(2)
assert res.index(1) < res.index(3) and res.index(2) < res.index(3)
assert kahn_topological_sort(2, [(0, 1), (1, 0)]) == []
''',
        "domain": "algorithms",
        "trust_tier": 1,
        "aliases": [
            "Topological sorting algorithm using in-degree BFS in Python.",
            "Kahn's algorithm for topological sorting in Python.",
        ],
    },
    {
        "prompt": "Write a python function for breadth first search (BFS) shortest path in an unweighted graph.",
        "response": '''from collections import deque

def bfs_shortest_path(graph: dict[int, list[int]], start: int, target: int) -> list[int]:
    """Find the shortest path from start to target in an unweighted graph using BFS."""
    if start == target:
        return [start]
    visited = {start}
    queue: deque[list[int]] = deque([[start]])
    while queue:
        path = queue.popleft()
        current = path[-1]
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                new_path = path + [neighbor]
                if neighbor == target:
                    return new_path
                visited.add(neighbor)
                queue.append(new_path)
    return []
''',
        "test_code": '''
adj = {1: [2, 3], 2: [1, 4], 3: [1, 4, 5], 4: [2, 3, 6], 5: [3], 6: [4]}
assert bfs_shortest_path(adj, 1, 6) in ([1, 2, 4, 6], [1, 3, 4, 6])
assert bfs_shortest_path(adj, 1, 5) == [1, 3, 5]
assert bfs_shortest_path(adj, 1, 99) == []
''',
        "domain": "algorithms",
        "trust_tier": 1,
        "aliases": [
            "Breadth first search shortest path algorithm in Python.",
            "Find shortest path in unweighted graph with BFS.",
        ],
    },
]


def main():
    vault_path = Path("artifacts/mirror_crystal_all_datasets.dcrystal")
    vault = DiskCrystalVault(vault_path)
    before_count = vault.rule_count()
    print(f"Initial Vault Rule Count: {before_count}")

    added = 0
    for idx, prim in enumerate(FRONTIER_PRIMITIVES):
        p = prim["prompt"]
        r = prim["response"]
        tc = prim["test_code"]
        dom = prim["domain"]
        tier = prim["trust_tier"]

        # 1. Verify in sandbox first
        ok, msg = ZeroTrustSandboxJudge.verify_code_and_test(r, test_code=tc)
        if not ok:
            print(f"[FAIL] Rule #{idx} rejected by sandbox: {msg}")
            continue

        # 2. Ingest into vault
        success = vault.ingest_pair(
            user_text=p,
            target_text=r,
            test_code=tc,
            domain=dom,
            trust_tier=tier,
        )
        if success:
            added += 1
            print(f"[OK] Ingested rule #{idx}: '{p[:50]}...'")
        else:
            print(f"[SKIP/REINFORCE] Rule #{idx}: already present")

        # Ingest aliases if defined
        for alias_p in prim.get("aliases", []):
            alias_ok = vault.ingest_pair(
                user_text=alias_p,
                target_text=r,
                test_code=tc,
                domain=dom,
                trust_tier=tier,
            )
            if alias_ok:
                added += 1
                print(f"  [OK] Ingested alias for #{idx}: '{alias_p[:45]}...'")

    vault.close()
    after_vault = DiskCrystalVault(vault_path)
    after_count = after_vault.rule_count()
    after_vault.close()
    print(f"Final Vault Rule Count: {after_count} (+{added} newly crystallized)")


if __name__ == "__main__":
    main()
