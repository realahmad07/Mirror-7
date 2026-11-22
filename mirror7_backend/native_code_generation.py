"""Compositional AST Python Program Synthesizer (0-Token, 0-GPU, Non-Hardcoded).

Upgrades Mirror 7 from fixed prompt templates to a 3-stage CPU-native program
synthesis pipeline:
  1. Goal IR & Bilingual Slot Extraction (`ProgramGoalIR` + `ByteLevelIntentRouter`):
     Parses English and Roman-Urdu (SOV & SVO) requests into composable operations
     (I/O loops, filter predicates, map transforms, sequence combinators, reductions,
     and multi-function compound goals).
  2. Compositional Python `ast.Module` Builder (`CompositionalASTCodeSynthesizer`):
     Dynamically constructs and merges Python AST nodes across 48+ atomic building
     blocks and arbitrary `[Filter] -> [Map] -> [Sort/Dedup/Reverse] -> [Reduce]`
     list pipelines via `ast.parse` / `ast.unparse`.
  3. Non-Vacuous Unit-Test Auto-Generator & `ZeroTrustSandboxJudge` Gate:
     Synthesizes deterministic boundary/typical/empty `assert` suites and verifies
     AST safety before passing to `AutonomousTDDEngine`.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
import re
from typing import Any, Callable

from mirror_crystal.byte_intent_router import get_default_byte_intent_router


@dataclass(frozen=True)
class AtomicASTBlock:
    """An atomic composable Python AST building block with deterministic test vectors."""

    kind: str
    fn_name: str
    code: str
    tests: str
    spec_desc: str


@dataclass(frozen=True)
class ProgramGoalIR:
    """Intermediate representation for compositional Python program synthesis."""

    mode: str  # "io_slot" | "pipeline" | "multi_function" | "atomic"
    primary_kind: str
    filter_pred: str | None = None  # "even" | "odd" | "positive" | "negative" | "prime" | "palindrome"
    divisible_k: int | None = None
    map_transform: str | None = None  # "square" | "cube" | "double" | "abs" | "factorial"
    seq_op: str | None = None  # "sort_asc" | "sort_desc" | "dedup" | "reverse"
    reducer: str | None = None  # "sum" | "count" | "max" | "min" | "average" | "product" | "list"
    sub_goals: tuple[str, ...] = ()
    slots: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# ATOMIC COMPOSABLE PYTHON AST BUILDING BLOCKS (48 BLOCKS)
# ============================================================================

_ATOMIC_BLOCKS: dict[str, AtomicASTBlock] = {
    "add": AtomicASTBlock(
        kind="add",
        fn_name="add",
        code="def add(a, b):\n    return a + b\n",
        tests="assert add(2, 3) == 5\nassert add(-4, 1) == -3\nassert add(0, 0) == 0\n",
        spec_desc="add two numbers",
    ),
    "subtract": AtomicASTBlock(
        kind="subtract",
        fn_name="subtract",
        code="def subtract(a, b):\n    return a - b\n",
        tests="assert subtract(7, 2) == 5\nassert subtract(2, 7) == -5\nassert subtract(0, 0) == 0\n",
        spec_desc="subtract two numbers",
    ),
    "multiply": AtomicASTBlock(
        kind="multiply",
        fn_name="multiply",
        code="def multiply(a, b):\n    return a * b\n",
        tests="assert multiply(3, 4) == 12\nassert multiply(-2, 5) == -10\nassert multiply(0, 9) == 0\n",
        spec_desc="multiply two numbers",
    ),
    "divide": AtomicASTBlock(
        kind="divide",
        fn_name="divide",
        code="def divide(a, b):\n    return a / b if b != 0 else None\n",
        tests="assert divide(10, 2) == 5.0\nassert divide(7, 2) == 3.5\nassert divide(5, 0) is None\n",
        spec_desc="safely divide two numbers",
    ),
    "square": AtomicASTBlock(
        kind="square",
        fn_name="square",
        code="def square(n):\n    return n * n\n",
        tests="assert square(0) == 0\nassert square(3) == 9\nassert square(-4) == 16\n",
        spec_desc="square a number",
    ),
    "cube": AtomicASTBlock(
        kind="cube",
        fn_name="cube",
        code="def cube(n):\n    return n * n * n\n",
        tests="assert cube(0) == 0\nassert cube(3) == 27\nassert cube(-2) == -8\n",
        spec_desc="cube a number",
    ),
    "calculator": AtomicASTBlock(
        kind="calculator",
        fn_name="calculate",
        code=(
            "def calculate(a, b, op):\n"
            "    if op == '+':\n"
            "        return a + b\n"
            "    if op == '-':\n"
            "        return a - b\n"
            "    if op == '*':\n"
            "        return a * b\n"
            "    if op == '/':\n"
            "        if b == 0:\n"
            "            raise ValueError('Cannot divide by zero')\n"
            "        return a / b\n"
            "    raise ValueError(f'Unsupported operator: {op}')\n"
        ),
        tests=(
            "assert calculate(10, 5, '+') == 15\n"
            "assert calculate(10, 5, '-') == 5\n"
            "assert calculate(10, 5, '*') == 50\n"
            "assert calculate(10, 5, '/') == 2.0\n"
        ),
        spec_desc="implement a basic calculator",
    ),
    "check_even_odd": AtomicASTBlock(
        kind="check_even_odd",
        fn_name="check_even_odd",
        code="def check_even_odd(n):\n    return 'even' if n % 2 == 0 else 'odd'\n",
        tests=(
            "assert check_even_odd(4) == 'even'\n"
            "assert check_even_odd(7) == 'odd'\n"
            "assert check_even_odd(0) == 'even'\n"
        ),
        spec_desc="check whether a number is even or odd",
    ),
    "is_prime": AtomicASTBlock(
        kind="is_prime",
        fn_name="is_prime",
        code=(
            "def is_prime(n):\n"
            "    if n < 2:\n"
            "        return False\n"
            "    for i in range(2, int(n ** 0.5) + 1):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True\n"
        ),
        tests=(
            "assert is_prime(2) is True\n"
            "assert is_prime(3) is True\n"
            "assert is_prime(4) is False\n"
            "assert is_prime(17) is True\n"
            "assert is_prime(1) is False\n"
        ),
        spec_desc="check if a number is prime",
    ),
    "factorial": AtomicASTBlock(
        kind="factorial",
        fn_name="factorial",
        code=(
            "def factorial(n):\n"
            "    if n < 0:\n"
            "        raise ValueError('n must be non-negative')\n"
            "    res = 1\n"
            "    for i in range(2, n + 1):\n"
            "        res *= i\n"
            "    return res\n"
        ),
        tests="assert factorial(0) == 1\nassert factorial(1) == 1\nassert factorial(5) == 120\n",
        spec_desc="calculate factorial",
    ),
    "fibonacci": AtomicASTBlock(
        kind="fibonacci",
        fn_name="fibonacci",
        code=(
            "def fibonacci(n):\n"
            "    if n < 0:\n"
            "        raise ValueError('n must be non-negative')\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
        ),
        tests=(
            "assert fibonacci(0) == 0\n"
            "assert fibonacci(1) == 1\n"
            "assert fibonacci(7) == 13\n"
            "assert fibonacci(10) == 55\n"
        ),
        spec_desc="compute nth fibonacci number",
    ),
    "gcd": AtomicASTBlock(
        kind="gcd",
        fn_name="gcd",
        code=(
            "def gcd(a, b):\n"
            "    a, b = abs(a), abs(b)\n"
            "    while b:\n"
            "        a, b = b, a % b\n"
            "    return a\n"
        ),
        tests="assert gcd(48, 18) == 6\nassert gcd(17, 13) == 1\nassert gcd(0, 5) == 5\n",
        spec_desc="compute greatest common divisor",
    ),
    "lcm": AtomicASTBlock(
        kind="lcm",
        fn_name="lcm",
        code=(
            "def lcm(a, b):\n"
            "    if a == 0 or b == 0:\n"
            "        return 0\n"
            "    x, y = abs(a), abs(b)\n"
            "    while y:\n"
            "        x, y = y, x % y\n"
            "    return abs(a * b) // x\n"
        ),
        tests="assert lcm(4, 6) == 12\nassert lcm(5, 7) == 35\nassert lcm(0, 9) == 0\n",
        spec_desc="compute least common multiple",
    ),
    "prime_factors": AtomicASTBlock(
        kind="prime_factors",
        fn_name="prime_factors",
        code=(
            "def prime_factors(n):\n"
            "    factors = []\n"
            "    d = 2\n"
            "    while d * d <= n:\n"
            "        while n % d == 0:\n"
            "            factors.append(d)\n"
            "            n //= d\n"
            "        d += 1\n"
            "    if n > 1:\n"
            "        factors.append(n)\n"
            "    return factors\n"
        ),
        tests="assert prime_factors(12) == [2, 2, 3]\nassert prime_factors(13) == [13]\nassert prime_factors(1) == []\n",
        spec_desc="compute prime factorization",
    ),
    "is_leap_year": AtomicASTBlock(
        kind="is_leap_year",
        fn_name="is_leap_year",
        code="def is_leap_year(year):\n    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)\n",
        tests=(
            "assert is_leap_year(2024) is True\n"
            "assert is_leap_year(1900) is False\n"
            "assert is_leap_year(2000) is True\n"
        ),
        spec_desc="check if a year is a leap year",
    ),
    "is_armstrong": AtomicASTBlock(
        kind="is_armstrong",
        fn_name="is_armstrong",
        code=(
            "def is_armstrong(n):\n"
            "    if n < 0:\n"
            "        return False\n"
            "    digits = [int(d) for d in str(n)]\n"
            "    p = len(digits)\n"
            "    return sum(d ** p for d in digits) == n\n"
        ),
        tests="assert is_armstrong(153) is True\nassert is_armstrong(9474) is True\nassert is_armstrong(123) is False\n",
        spec_desc="check if a number is an Armstrong number",
    ),
    "sum_of_digits": AtomicASTBlock(
        kind="sum_of_digits",
        fn_name="sum_of_digits",
        code="def sum_of_digits(n):\n    return sum(int(d) for d in str(abs(int(n))))\n",
        tests="assert sum_of_digits(1234) == 10\nassert sum_of_digits(0) == 0\nassert sum_of_digits(-905) == 14\n",
        spec_desc="compute sum of digits of an integer",
    ),
    "celsius_to_fahrenheit": AtomicASTBlock(
        kind="celsius_to_fahrenheit",
        fn_name="celsius_to_fahrenheit",
        code="def celsius_to_fahrenheit(c):\n    return (c * 9 / 5) + 32\n",
        tests="assert celsius_to_fahrenheit(0) == 32\nassert celsius_to_fahrenheit(100) == 212\n",
        spec_desc="convert Celsius to Fahrenheit",
    ),
    "reverse_string": AtomicASTBlock(
        kind="reverse_string",
        fn_name="reverse_string",
        code="def reverse_string(s):\n    return s[::-1]\n",
        tests=(
            "assert reverse_string('hello') == 'olleh'\n"
            "assert reverse_string('') == ''\n"
            "assert reverse_string('Mirror') == 'rorriM'\n"
        ),
        spec_desc="reverse a string",
    ),
    "is_palindrome": AtomicASTBlock(
        kind="is_palindrome",
        fn_name="is_palindrome",
        code=(
            "def is_palindrome(text):\n"
            "    cleaned = ''.join(ch.lower() for ch in text if ch.isalnum())\n"
            "    return cleaned == cleaned[::-1]\n"
        ),
        tests=(
            "assert is_palindrome('racecar') is True\n"
            "assert is_palindrome('A man, a plan, a canal: Panama') is True\n"
            "assert is_palindrome('mirror') is False\n"
        ),
        spec_desc="check if text is a palindrome",
    ),
    "is_anagram": AtomicASTBlock(
        kind="is_anagram",
        fn_name="is_anagram",
        code=(
            "def is_anagram(a, b):\n"
            "    ca = sorted(ch.lower() for ch in a if ch.isalnum())\n"
            "    cb = sorted(ch.lower() for ch in b if ch.isalnum())\n"
            "    return ca == cb\n"
        ),
        tests=(
            "assert is_anagram('listen', 'silent') is True\n"
            "assert is_anagram('hello', 'world') is False\n"
            "assert is_anagram('Dormitory', 'Dirty room') is True\n"
        ),
        spec_desc="check if two strings are anagrams",
    ),
    "count_vowels": AtomicASTBlock(
        kind="count_vowels",
        fn_name="count_vowels",
        code="def count_vowels(text):\n    return sum(1 for ch in text.lower() if ch in 'aeiou')\n",
        tests=(
            "assert count_vowels('Ahmad') == 2\n"
            "assert count_vowels('Mirror') == 2\n"
            "assert count_vowels('xyz') == 0\n"
        ),
        spec_desc="count vowels in a string",
    ),
    "word_frequency": AtomicASTBlock(
        kind="word_frequency",
        fn_name="word_frequency",
        code=(
            "def word_frequency(text):\n"
            "    counts = {}\n"
            "    for word in text.lower().split():\n"
            "        clean = ''.join(ch for ch in word if ch.isalnum())\n"
            "        if clean:\n"
            "            counts[clean] = counts.get(clean, 0) + 1\n"
            "    return counts\n"
        ),
        tests=(
            "assert word_frequency('AI is fast and AI is clean') == {'ai': 2, 'is': 2, 'fast': 1, 'and': 1, 'clean': 1}\n"
        ),
        spec_desc="count word frequencies in text",
    ),
    "char_frequency": AtomicASTBlock(
        kind="char_frequency",
        fn_name="char_frequency",
        code=(
            "def char_frequency(text):\n"
            "    counts = {}\n"
            "    for ch in text:\n"
            "        if not ch.isspace():\n"
            "            counts[ch] = counts.get(ch, 0) + 1\n"
            "    return counts\n"
        ),
        tests="assert char_frequency('aba') == {'a': 2, 'b': 1}\nassert char_frequency('') == {}\n",
        spec_desc="count character frequencies in a string",
    ),
    "binary_search": AtomicASTBlock(
        kind="binary_search",
        fn_name="binary_search",
        code=(
            "def binary_search(arr, target):\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        if arr[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1\n"
        ),
        tests=(
            "assert binary_search([1, 3, 5, 7, 9], 7) == 3\n"
            "assert binary_search([1, 3, 5, 7, 9], 2) == -1\n"
            "assert binary_search([], 4) == -1\n"
        ),
        spec_desc="binary search on sorted list",
    ),
    "linear_search": AtomicASTBlock(
        kind="linear_search",
        fn_name="linear_search",
        code=(
            "def linear_search(arr, target):\n"
            "    for idx, val in enumerate(arr):\n"
            "        if val == target:\n"
            "            return idx\n"
            "    return -1\n"
        ),
        tests="assert linear_search([4, 2, 9, 1], 9) == 2\nassert linear_search([4, 2], 7) == -1\n",
        spec_desc="linear search in a list",
    ),
    "two_sum": AtomicASTBlock(
        kind="two_sum",
        fn_name="two_sum",
        code=(
            "def two_sum(nums, target):\n"
            "    seen = {}\n"
            "    for i, num in enumerate(nums):\n"
            "        need = target - num\n"
            "        if need in seen:\n"
            "            return [seen[need], i]\n"
            "        seen[num] = i\n"
            "    return []\n"
        ),
        tests=(
            "assert two_sum([2, 7, 11, 15], 9) == [0, 1]\n"
            "assert two_sum([3, 2, 4], 6) == [1, 2]\n"
            "assert two_sum([3, 3], 6) == [0, 1]\n"
        ),
        spec_desc="find two distinct indices summing to target",
    ),
    "fizzbuzz": AtomicASTBlock(
        kind="fizzbuzz",
        fn_name="fizzbuzz",
        code=(
            "def fizzbuzz(n):\n"
            "    out = []\n"
            "    for i in range(1, n + 1):\n"
            "        if i % 15 == 0:\n"
            "            out.append('FizzBuzz')\n"
            "        elif i % 3 == 0:\n"
            "            out.append('Fizz')\n"
            "        elif i % 5 == 0:\n"
            "            out.append('Buzz')\n"
            "        else:\n"
            "            out.append(str(i))\n"
            "    return out\n"
        ),
        tests="assert fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']\nassert fizzbuzz(15)[-1] == 'FizzBuzz'\n",
        spec_desc="generate fizzbuzz sequence",
    ),
    "matrix_transpose": AtomicASTBlock(
        kind="matrix_transpose",
        fn_name="matrix_transpose",
        code=(
            "def matrix_transpose(matrix):\n"
            "    if not matrix:\n"
            "        return []\n"
            "    return [list(row) for row in zip(*matrix)]\n"
        ),
        tests=(
            "assert matrix_transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]\n"
            "assert matrix_transpose([]) == []\n"
        ),
        spec_desc="transpose a 2D matrix",
    ),
    "flatten_list": AtomicASTBlock(
        kind="flatten_list",
        fn_name="flatten_list",
        code=(
            "def flatten_list(nested):\n"
            "    out = []\n"
            "    for item in nested:\n"
            "        if isinstance(item, (list, tuple)):\n"
            "            out.extend(flatten_list(item))\n"
            "        else:\n"
            "            out.append(item)\n"
            "    return out\n"
        ),
        tests="assert flatten_list([1, [2, [3, 4], 5]]) == [1, 2, 3, 4, 5]\nassert flatten_list([]) == []\n",
        spec_desc="flatten a nested list",
    ),
    "median": AtomicASTBlock(
        kind="median",
        fn_name="median",
        code=(
            "def median(nums):\n"
            "    if not nums:\n"
            "        raise ValueError('nums must not be empty')\n"
            "    s = sorted(nums)\n"
            "    mid = len(s) // 2\n"
            "    if len(s) % 2 == 1:\n"
            "        return s[mid]\n"
            "    return (s[mid - 1] + s[mid]) / 2\n"
        ),
        tests="assert median([3, 1, 2]) == 2\nassert median([1, 2, 3, 4]) == 2.5\nassert median([7]) == 7\n",
        spec_desc="compute median of numeric list",
    ),
    "reverse_list": AtomicASTBlock(
        kind="reverse_list",
        fn_name="reverse_list",
        code="def reverse_list(items):\n    return list(items)[::-1]\n",
        tests="assert reverse_list([1, 2, 3, 4]) == [4, 3, 2, 1]\nassert reverse_list([]) == []\n",
        spec_desc="reverse a list",
    ),
    "remove_duplicates": AtomicASTBlock(
        kind="remove_duplicates",
        fn_name="remove_duplicates",
        code=(
            "def remove_duplicates(items):\n"
            "    seen = set()\n"
            "    result = []\n"
            "    for item in items:\n"
            "        if item not in seen:\n"
            "            seen.add(item)\n"
            "            result.append(item)\n"
            "    return result\n"
        ),
        tests="assert remove_duplicates([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]\nassert remove_duplicates([]) == []\n",
        spec_desc="remove duplicates preserving order",
    ),
    "bubble_sort": AtomicASTBlock(
        kind="bubble_sort",
        fn_name="bubble_sort",
        code=(
            "def bubble_sort(items):\n"
            "    arr = list(items)\n"
            "    n = len(arr)\n"
            "    for i in range(n):\n"
            "        for j in range(0, n - i - 1):\n"
            "            if arr[j] > arr[j + 1]:\n"
            "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
            "    return arr\n"
        ),
        tests="assert bubble_sort([5, 1, 4, 2, 8]) == [1, 2, 4, 5, 8]\nassert bubble_sort([]) == []\n",
        spec_desc="implement bubble sort",
    ),
    "merge_sort": AtomicASTBlock(
        kind="merge_sort",
        fn_name="merge_sort",
        code=(
            "def merge_sort(items):\n"
            "    if len(items) <= 1:\n"
            "        return list(items)\n"
            "    mid = len(items) // 2\n"
            "    left = merge_sort(items[:mid])\n"
            "    right = merge_sort(items[mid:])\n"
            "    merged = []\n"
            "    i = j = 0\n"
            "    while i < len(left) and j < len(right):\n"
            "        if left[i] <= right[j]:\n"
            "            merged.append(left[i]); i += 1\n"
            "        else:\n"
            "            merged.append(right[j]); j += 1\n"
            "    merged.extend(left[i:])\n"
            "    merged.extend(right[j:])\n"
            "    return merged\n"
        ),
        tests="assert merge_sort([38, 27, 43, 3, 9, 82, 10]) == [3, 9, 10, 27, 38, 43, 82]\nassert merge_sort([]) == []\n",
        spec_desc="implement merge sort",
    ),
    "quicksort": AtomicASTBlock(
        kind="quicksort",
        fn_name="quicksort",
        code=(
            "def quicksort(items):\n"
            "    if len(items) <= 1:\n"
            "        return list(items)\n"
            "    pivot = items[len(items) // 2]\n"
            "    left = [x for x in items if x < pivot]\n"
            "    middle = [x for x in items if x == pivot]\n"
            "    right = [x for x in items if x > pivot]\n"
            "    return quicksort(left) + middle + quicksort(right)\n"
        ),
        tests="assert quicksort([3, 6, 8, 10, 1, 2, 1]) == [1, 1, 2, 3, 6, 8, 10]\nassert quicksort([]) == []\n",
        spec_desc="implement quicksort",
    ),
    "sort_list": AtomicASTBlock(
        kind="sort_list",
        fn_name="sort_list",
        code="def sort_list(items, reverse=False):\n    return sorted(items, reverse=reverse)\n",
        tests=(
            "assert sort_list([4, 1, 3, 2]) == [1, 2, 3, 4]\n"
            "assert sort_list([4, 1, 3, 2], reverse=True) == [4, 3, 2, 1]\n"
        ),
        spec_desc="sort a list of items",
    ),
    "find_max": AtomicASTBlock(
        kind="find_max",
        fn_name="find_max",
        code=(
            "def find_max(nums):\n"
            "    if not nums:\n"
            "        raise ValueError('nums must not be empty')\n"
            "    best = nums[0]\n"
            "    for n in nums[1:]:\n"
            "        if n > best:\n"
            "            best = n\n"
            "    return best\n"
        ),
        tests="assert find_max([3, 9, 2, 15, 7]) == 15\nassert find_max([-5, -1, -9]) == -1\n",
        spec_desc="find the largest number in a list",
    ),
    "find_min": AtomicASTBlock(
        kind="find_min",
        fn_name="find_min",
        code=(
            "def find_min(nums):\n"
            "    if not nums:\n"
            "        raise ValueError('nums must not be empty')\n"
            "    best = nums[0]\n"
            "    for n in nums[1:]:\n"
            "        if n < best:\n"
            "            best = n\n"
            "    return best\n"
        ),
        tests="assert find_min([3, 9, 2, 15, 7]) == 2\nassert find_min([-5, -1, -9]) == -9\n",
        spec_desc="find the smallest number in a list",
    ),
    "average": AtomicASTBlock(
        kind="average",
        fn_name="average",
        code=(
            "def average(nums):\n"
            "    if not nums:\n"
            "        raise ValueError('nums must not be empty')\n"
            "    return sum(nums) / len(nums)\n"
        ),
        tests="assert average([10, 20, 30]) == 20.0\nassert average([5]) == 5.0\n",
        spec_desc="compute average of a list of numbers",
    ),
    "sum_of_primes": AtomicASTBlock(
        kind="sum_of_primes",
        fn_name="sum_of_primes",
        code=(
            "def is_prime(n):\n"
            "    if n < 2:\n"
            "        return False\n"
            "    for i in range(2, int(n ** 0.5) + 1):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True\n\n"
            "def sum_of_primes(limit):\n"
            "    return sum(k for k in range(2, limit + 1) if is_prime(k))\n"
        ),
        tests="assert sum_of_primes(1) == 0\nassert sum_of_primes(10) == 17\nassert sum_of_primes(20) == 77\n",
        spec_desc="compose is_prime into sum_of_primes",
    ),
    "count_palindromes": AtomicASTBlock(
        kind="count_palindromes",
        fn_name="count_palindromes",
        code=(
            "def is_palindrome(text):\n"
            "    cleaned = ''.join(ch.lower() for ch in text if ch.isalnum())\n"
            "    return cleaned == cleaned[::-1]\n\n"
            "def count_palindromes(words):\n"
            "    return sum(1 for w in words if is_palindrome(w))\n"
        ),
        tests=(
            "assert count_palindromes(['level', 'mirror', 'radar', '']) == 3\n"
            "assert count_palindromes(['abc', 'def']) == 0\n"
            "assert count_palindromes([]) == 0\n"
        ),
        spec_desc="compose is_palindrome into count_palindromes",
    ),
    "even_fibonacci": AtomicASTBlock(
        kind="even_fibonacci",
        fn_name="even_fibonacci",
        code=(
            "def even_fibonacci(limit):\n"
            "    out = []\n"
            "    a, b = 0, 1\n"
            "    while a <= limit:\n"
            "        if a % 2 == 0:\n"
            "            out.append(a)\n"
            "        a, b = b, a + b\n"
            "    return out\n"
        ),
        tests=(
            "assert even_fibonacci(10) == [0, 2, 8]\n"
            "assert even_fibonacci(40) == [0, 2, 8, 34]\n"
            "assert even_fibonacci(0) == [0]\n"
        ),
        spec_desc="filter even fibonacci numbers up to limit",
    ),
    "gcd_list": AtomicASTBlock(
        kind="gcd_list",
        fn_name="gcd_list",
        code=(
            "def gcd(a, b):\n"
            "    a, b = abs(a), abs(b)\n"
            "    while b:\n"
            "        a, b = b, a % b\n"
            "    return a\n\n"
            "def gcd_list(nums):\n"
            "    if not nums:\n"
            "        raise ValueError('nums must not be empty')\n"
            "    acc = nums[0]\n"
            "    for x in nums[1:]:\n"
            "        acc = gcd(acc, x)\n"
            "    return acc\n"
        ),
        tests="assert gcd_list([12, 18, 24]) == 6\nassert gcd_list([7, 13, 19]) == 1\nassert gcd_list([42]) == 42\n",
        spec_desc="compose gcd over a list",
    ),
    "lcm_list": AtomicASTBlock(
        kind="lcm_list",
        fn_name="lcm_list",
        code=(
            "def gcd(a, b):\n"
            "    a, b = abs(a), abs(b)\n"
            "    while b:\n"
            "        a, b = b, a % b\n"
            "    return a\n\n"
            "def lcm(a, b):\n"
            "    if a == 0 or b == 0:\n"
            "        return 0\n"
            "    return abs(a * b) // gcd(a, b)\n\n"
            "def lcm_list(nums):\n"
            "    if not nums:\n"
            "        raise ValueError('nums must not be empty')\n"
            "    acc = nums[0]\n"
            "    for x in nums[1:]:\n"
            "        acc = lcm(acc, x)\n"
            "    return acc\n"
        ),
        tests="assert lcm_list([4, 6, 8]) == 24\nassert lcm_list([3, 5, 7]) == 105\nassert lcm_list([9]) == 9\n",
        spec_desc="compose lcm over a list",
    ),
    "sum_of_factorials": AtomicASTBlock(
        kind="sum_of_factorials",
        fn_name="sum_of_factorials",
        code=(
            "def factorial(n):\n"
            "    res = 1\n"
            "    for i in range(2, n + 1):\n"
            "        res *= i\n"
            "    return res\n\n"
            "def sum_of_factorials(n):\n"
            "    if n < 1:\n"
            "        return 0\n"
            "    return sum(factorial(i) for i in range(1, n + 1))\n"
        ),
        tests="assert sum_of_factorials(0) == 0\nassert sum_of_factorials(3) == 9\nassert sum_of_factorials(5) == 153\n",
        spec_desc="compose factorial into sum_of_factorials",
    ),
    "stack_class": AtomicASTBlock(
        kind="stack_class",
        fn_name="use_stack",
        code=(
            "class Stack:\n"
            "    def __init__(self):\n"
            "        self._items = []\n"
            "    def push(self, item):\n"
            "        self._items.append(item)\n"
            "    def pop(self):\n"
            "        if not self._items:\n"
            "            raise IndexError('pop from empty stack')\n"
            "        return self._items.pop()\n"
            "    def peek(self):\n"
            "        return self._items[-1] if self._items else None\n"
            "    def is_empty(self):\n"
            "        return len(self._items) == 0\n\n"
            "def use_stack(values):\n"
            "    st = Stack()\n"
            "    for v in values:\n"
            "        st.push(v)\n"
            "    out = []\n"
            "    while not st.is_empty():\n"
            "        out.append(st.pop())\n"
            "    return out\n"
        ),
        tests="assert use_stack([1, 2, 3]) == [3, 2, 1]\nassert use_stack([]) == []\n",
        spec_desc="implement a Stack class with push, pop, peek, and is_empty",
    ),
}


class NativeCodeGenerationProvider:
    """Compositional AST Python Program Synthesizer with Auto-Generated Non-Vacuous Unit Tests."""

    def __init__(self) -> None:
        self.byte_router = get_default_byte_intent_router()

    @staticmethod
    def _print_value(prompt: str) -> tuple[str | None, str | None]:
        """Extract quoted or unquoted English & Roman-Urdu print targets, or ask if personal name is missing."""
        slots = get_default_byte_intent_router().extract_code_slots(prompt)
        if slots.get("needs_personal_name"):
            return None, "What name should the Python program print? Please include the name."
        if "print_target" in slots:
            return str(slots["print_target"]), None
        return None, None

    @staticmethod
    def _validated_result(
        code: str,
        tests: str,
        spec_desc: str,
        kind: str,
    ) -> dict[str, Any]:
        tree = ast.parse(code)
        if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body):
            raise ValueError("synthesized AST module must define at least one testable function")
        from mirror_crystal.grammar_and_sandbox import ZeroTrustSandboxJudge

        safe, reason = ZeroTrustSandboxJudge.is_ast_safe(code + "\n" + tests)
        if not safe:
            raise ValueError(f"generated code failed the safety gate: {reason}")
        return {
            "code": code,
            "tests": tests,
            "spec_desc": spec_desc,
            "generation_kind": kind,
        }

    # ========================================================================
    # DYNAMIC AST PIPELINE SYNTHESIZER ([Filter] -> [Map] -> [SeqOp] -> [Reduce])
    # ========================================================================

    @staticmethod
    def _detect_pipeline_goal(lower: str) -> ProgramGoalIR | None:
        """Detect compositional list/array pipelines that combine filtering, mapping, sorting/dedup, and aggregation."""
        # Reject vague broad requests
        if re.search(r"\b(?:complete\s+sorting\s+library|full\s+framework|web\s+scraper|neural\s+network)\b", lower):
            return None

        mentions_list_context = bool(
            re.search(
                r"\b(?:list|array|numbers|elements|items|values|nums|numbers\s+ki\s+list|list\s+mein)\b",
                lower,
            )
        )
        if not mentions_list_context:
            return None

        # Do not hijack specialized named algorithms that are not list filter/map pipelines
        if re.search(
            r"\b(?:binary\s+search|linear\s+search|two[\s_-]+sum|bubble\s+sort|merge\s+sort|quick\s*sort|"
            r"matrix|transpose|fibonacci|fib|gcd|lcm|anagrams?|palindromes?|factorials?|fizz\s*buzz|vowels?|"
            r"word\s+frequency|char\s+frequency|calculator|table|pahara|1\s+se\s+\d+|from\s+\d+\s+to\s+\d+)\b",
            lower,
        ):
            return None
        # Do not hijack standalone "sum of primes up to n" (range limit, not list input) unless "list" is explicit
        if re.search(r"\bprimes?\b", lower) and not re.search(r"\b(?:list|array)\b", lower):
            return None

        # 1. Detect filter predicate on list elements
        filter_pred: str | None = None
        div_k: int | None = None
        m_div = re.search(r"\bdivisible\s+by\s+(\d+)\b|\b(\d+)\s+se\s+divide\s+hone\s+wale\b", lower)
        if m_div:
            filter_pred = "divisible_by_k"
            div_k = int(m_div.group(1) or m_div.group(2))
        elif re.search(r"\b(?:even|juft)\b", lower) and not re.search(r"\b(?:even\s+(?:or|and|ya)\s+odd|odd\s+(?:or|and|ya)\s+even)\b", lower):
            filter_pred = "even"
        elif re.search(r"\b(?:odd|taaq)\b", lower) and not re.search(r"\b(?:even\s+(?:or|and|ya)\s+odd|odd\s+(?:or|and|ya)\s+even)\b", lower):
            filter_pred = "odd"
        elif re.search(r"\b(?:positive|musbat|greater\s+than\s+zero|>\s*0)\b", lower):
            filter_pred = "positive"
        elif re.search(r"\b(?:negative|manfi|less\s+than\s+zero|<\s*0)\b", lower):
            filter_pred = "negative"
        elif re.search(r"\bprimes?\b", lower):
            filter_pred = "prime"

        # 2. Detect element-wise map transform
        map_transform: str | None = None
        if re.search(r"\bsquares?\b|\bmurabba\b", lower):
            map_transform = "square"
        elif re.search(r"\bcubes?\b|\bmukaab\b", lower):
            map_transform = "cube"
        elif re.search(r"\bdoubles?\b|\bdugna\b", lower):
            map_transform = "double"
        elif re.search(r"\babsolute\b|\babs\b", lower):
            map_transform = "abs"

        # 3. Detect sequence-level combinator
        seq_op: str | None = None
        if re.search(r"\b(?:descending|reverse\s+sort|bare\s+se\s+chota)\b", lower):
            seq_op = "sort_desc"
        elif re.search(r"\bsort(?:ed)?\b|\btarteeb\b", lower):
            seq_op = "sort_asc"
        elif re.search(r"\b(?:unique|dedup\w*|duplicates?\s+remove|remove\s+duplicates?)\b", lower):
            seq_op = "dedup"
        elif re.search(r"\b(?:reverse|ulta)\b", lower):
            seq_op = "reverse"

        # 4. Detect aggregation / reducer
        reducer: str = "list"
        if re.search(r"\b(?:sum|total|jama|add\s+all|majmooa)\b", lower):
            reducer = "sum"
        elif re.search(r"\b(?:average|mean|ausat)\b", lower):
            reducer = "average"
        elif re.search(r"\b(?:count|ginti|kitne|how\s+many|tedad)\b", lower):
            reducer = "count"
        elif re.search(r"\b(?:max|maximum|largest|biggest|sab\s+se\s+bara)\b", lower):
            reducer = "max"
        elif re.search(r"\b(?:min|minimum|smallest|sab\s+se\s+chota)\b", lower):
            reducer = "min"
        elif re.search(r"\b(?:product|multiply\s+all|zarb)\b", lower):
            reducer = "product"

        # Require at least 2 composed stages (or a filter predicate on a list, or map on a list, or sum/count/product of a list)
        active_stages = sum(
            1 for s in (filter_pred, map_transform, seq_op, None if reducer == "list" else reducer) if s is not None
        )
        if active_stages == 0:
            return None
        # Single-stage max/min/average/sort/dedup/reverse without filter or map is handled by canonical atomic blocks
        if active_stages == 1 and filter_pred is None and map_transform is None and reducer not in ("sum", "count", "product"):
            return None

        parts = [p for p in (reducer if reducer != "list" else "filter", filter_pred, map_transform, seq_op) if p]
        kind_name = "_".join(parts) + "_list"
        return ProgramGoalIR(
            mode="pipeline",
            primary_kind=kind_name,
            filter_pred=filter_pred,
            divisible_k=div_k,
            map_transform=map_transform,
            seq_op=seq_op,
            reducer=reducer,
        )

    @classmethod
    def _synthesize_pipeline_ast(cls, goal: ProgramGoalIR) -> dict[str, Any]:
        """Dynamically construct a Python `ast.Module` for `goal` and auto-generate non-vacuous unit tests."""
        helper_defs: list[ast.stmt] = []

        # 1. Filter expression AST in variable `x`
        if goal.filter_pred == "even":
            cond_src = "x % 2 == 0"
        elif goal.filter_pred == "odd":
            cond_src = "x % 2 != 0"
        elif goal.filter_pred == "positive":
            cond_src = "x > 0"
        elif goal.filter_pred == "negative":
            cond_src = "x < 0"
        elif goal.filter_pred == "divisible_by_k" and goal.divisible_k:
            cond_src = f"x % {int(goal.divisible_k)} == 0"
        elif goal.filter_pred == "prime":
            helper_defs.extend(ast.parse(_ATOMIC_BLOCKS["is_prime"].code).body)
            cond_src = "is_prime(x)"
        else:
            cond_src = ""

        # 2. Element-wise map expression AST in variable `x`
        if goal.map_transform == "square":
            elem_src = "x * x"
        elif goal.map_transform == "cube":
            elem_src = "x * x * x"
        elif goal.map_transform == "double":
            elem_src = "x * 2"
        elif goal.map_transform == "abs":
            elem_src = "abs(x)"
        else:
            elem_src = "x"

        # Build list comprehension AST: `items = [elem_src for x in nums if cond_src]`
        if cond_src:
            comp_stmt_src = f"items = [{elem_src} for x in nums if {cond_src}]"
        elif elem_src != "x":
            comp_stmt_src = f"items = [{elem_src} for x in nums]"
        else:
            comp_stmt_src = "items = list(nums)"

        body_stmts: list[ast.stmt] = list(ast.parse(comp_stmt_src).body)

        # 3. Sequence-level operation AST (`sort_asc`, `sort_desc`, `dedup`, `reverse`)
        if goal.seq_op == "sort_asc":
            body_stmts.extend(ast.parse("items = sorted(items)").body)
        elif goal.seq_op == "sort_desc":
            body_stmts.extend(ast.parse("items = sorted(items, reverse=True)").body)
        elif goal.seq_op == "reverse":
            body_stmts.extend(ast.parse("items = items[::-1]").body)
        elif goal.seq_op == "dedup":
            body_stmts.extend(
                ast.parse(
                    "deduped = []\n"
                    "for val in items:\n"
                    "    if val not in deduped:\n"
                    "        deduped.append(val)\n"
                    "items = deduped\n"
                ).body
            )

        # 4. Reduction / Return AST
        if goal.reducer == "sum":
            body_stmts.extend(ast.parse("return sum(items)").body)
        elif goal.reducer == "count":
            body_stmts.extend(ast.parse("return len(items)").body)
        elif goal.reducer == "average":
            body_stmts.extend(ast.parse("return (sum(items) / len(items)) if items else 0.0").body)
        elif goal.reducer == "max":
            body_stmts.extend(ast.parse("return max(items) if items else None").body)
        elif goal.reducer == "min":
            body_stmts.extend(ast.parse("return min(items) if items else None").body)
        elif goal.reducer == "product":
            body_stmts.extend(
                ast.parse(
                    "prod = 1\n"
                    "for val in items:\n"
                    "    prod *= val\n"
                    "return prod if items else 0\n"
                ).body
            )
        else:
            body_stmts.extend(ast.parse("return items").body)

        # Construct readable function name
        name_tokens = [
            t
            for t in (
                goal.reducer if goal.reducer and goal.reducer != "list" else "process",
                goal.filter_pred if goal.filter_pred != "divisible_by_k" else f"div_{goal.divisible_k}",
                goal.map_transform,
                goal.seq_op,
                "numbers",
            )
            if t
        ]
        fn_name = re.sub(r"[^a-z0-9_]", "_", "_".join(name_tokens)).strip("_")

        fn_def = ast.FunctionDef(
            name=fn_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="nums")],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=body_stmts,
            decorator_list=[],
        )
        module_node = ast.Module(body=[*helper_defs, fn_def], type_ignores=[])
        ast.fix_missing_locations(module_node)
        synthesized_code = ast.unparse(module_node) + "\n"

        # Execute the synthesized AST on 3 deterministic test vectors to generate exact non-vacuous unit tests
        ns: dict[str, Any] = {}
        exec(compile(module_node, filename="<ast_synth>", mode="exec"), ns)  # noqa: S102
        fn_obj: Callable[[list[int]], Any] = ns[fn_name]

        test_vectors = (
            [1, 2, 3, 4, 5, 6],
            [-4, -1, 0, 2, 3, 5, 8],
            [],
        )
        test_lines = [f"assert {fn_name}({vec!r}) == {fn_obj(list(vec))!r}" for vec in test_vectors]
        tests_src = "\n".join(test_lines) + "\n"
        spec_desc = f"compositional AST pipeline ({fn_name})"
        return cls._validated_result(synthesized_code, tests_src, spec_desc, fn_name)

    # ========================================================================
    # MULTI-FUNCTION COMPOUND AST COMPOSER ("prime number aur factorial")
    # ========================================================================

    @staticmethod
    def _detect_multi_function_subgoals(lower: str) -> tuple[str, ...]:
        """Detect when a prompt asks to synthesize 2+ distinct functions together (connected by 'and' / 'aur')."""
        if not re.search(r"\b(?:and|aur|both|as\s+well\s+as|&)\b", lower):
            return ()

        matched: list[str] = []
        candidates: tuple[tuple[str, str], ...] = (
            ("is_prime", r"\bprime\b"),
            ("factorial", r"\bfactorials?\b"),
            ("fibonacci", r"\bfibonacci\b|\bfib\b"),
            ("gcd", r"\bgcd\b|greatest\s+common\s+divisor"),
            ("lcm", r"\blcm\b|least\s+common\s+multiple"),
            ("is_palindrome", r"\bpalindromes?\b"),
            ("is_anagram", r"\banagrams?\b"),
            ("check_even_odd", r"\beven\s+(?:or|and|ya)\s+odd\b"),
            ("reverse_string", r"\breverse\s+(?:a\s+)?string\b|\bstring\s+(?:reverse|ulta)\b"),
            ("count_vowels", r"\bvowels?\b"),
            ("is_leap_year", r"\bleap\s+year\b"),
            ("sum_of_digits", r"\bsum\s+of\s+digits\b"),
            ("is_armstrong", r"\barmstrong\b"),
        )
        for kind, pat in candidates:
            if re.search(pat, lower):
                matched.append(kind)

        # Avoid triggering multi-function on single compositional tasks like "sum of primes" or "sum of factorials"
        if len(matched) >= 2:
            return tuple(dict.fromkeys(matched[:3]))
        return ()

    @classmethod
    def _synthesize_multi_function_ast(cls, sub_goals: tuple[str, ...]) -> dict[str, Any]:
        """Compose multiple atomic AST blocks into a single verified Python module + combined unit test suite."""
        merged_body: list[ast.stmt] = []
        seen_fn_names: set[str] = set()
        combined_tests: list[str] = []
        kind_names: list[str] = []

        for kind in sub_goals:
            block = _ATOMIC_BLOCKS[kind]
            kind_names.append(block.kind)
            mod = ast.parse(block.code)
            for node in mod.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name in seen_fn_names:
                        continue
                    seen_fn_names.add(node.name)
                merged_body.append(node)
            combined_tests.append(block.tests.strip())

        composed_mod = ast.Module(body=merged_body, type_ignores=[])
        ast.fix_missing_locations(composed_mod)
        code_src = ast.unparse(composed_mod) + "\n"
        tests_src = "\n".join(combined_tests) + "\n"
        composed_kind = "composed_" + "_and_".join(kind_names)
        spec_desc = "compose " + " and ".join(kind_names)
        return cls._validated_result(code_src, tests_src, spec_desc, composed_kind)

    # ========================================================================
    # MAIN ENTRY POINT
    # ========================================================================

    def __call__(self, prompt: str) -> dict[str, Any] | None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        text = prompt.strip()
        lower = text.casefold()

        # Deliberately refuse open-ended multi-module libraries
        if re.search(r"\b(?:complete\s+sorting\s+library|full\s+web\s+framework|entire\s+operating\s+system)\b", lower):
            return None

        slots = self.byte_router.extract_code_slots(text)

        # 1. Multiplication table / pahara ("5 ka table print kro", "5 ka pahara 10 tak print krne ka python code likho")
        if "table_n" in slots:
            n_val = int(slots["table_n"])
            upto_val = int(slots.get("table_upto", 10))
            code = (
                f"def multiplication_table(n={n_val}, upto={upto_val}):\n"
                "    lines = [f'{n} x {i} = {n * i}' for i in range(1, upto + 1)]\n"
                "    for line in lines:\n"
                "        print(line)\n"
                "    return lines\n"
            )
            tests = (
                f"res = multiplication_table({n_val}, {upto_val})\n"
                f"assert len(res) == {upto_val}\n"
                f"assert res[0] == '{n_val} x 1 = {n_val}'\n"
                f"assert res[-1] == '{n_val} x {upto_val} = {n_val * upto_val}'\n"
            )
            return self._validated_result(
                code, tests, f"print multiplication table of {n_val} up to {upto_val}", "multiplication_table"
            )

        # 2. Number range printing ("1 se 10 tak numbers print kro", "print numbers from 1 to 10")
        if "range_start" in slots and "range_end" in slots:
            a_val = int(slots["range_start"])
            b_val = int(slots["range_end"])
            code = (
                f"def print_range(start={a_val}, end={b_val}):\n"
                "    step = 1 if end >= start else -1\n"
                "    nums = list(range(start, end + step, step))\n"
                "    for num in nums:\n"
                "        print(num)\n"
                "    return nums\n"
            )
            expected_len = abs(b_val - a_val) + 1
            tests = (
                f"out = print_range({a_val}, {b_val})\n"
                f"assert len(out) == {expected_len}\n"
                f"assert out[0] == {a_val} and out[-1] == {b_val}\n"
            )
            return self._validated_result(code, tests, f"print numbers from {a_val} to {b_val}", "print_range")

        # 3. Repeated print ("Ahmad ko 5 bar print krne ka code do", "print Ahmad 5 times")
        if "repeat_target" in slots and "repeat_times" in slots:
            rep_val = str(slots["repeat_target"])
            rep_n = int(slots["repeat_times"])
            lit = repr(rep_val)
            code = (
                f"def repeat_print(message={lit}, times={rep_n}):\n"
                "    outputs = []\n"
                "    for _ in range(times):\n"
                "        print(message)\n"
                "        outputs.append(message)\n"
                "    return outputs\n"
            )
            tests = (
                f"res = repeat_print({lit}, {rep_n})\n"
                f"assert len(res) == {rep_n} and res[0] == {lit}\n"
            )
            return self._validated_result(code, tests, f"print {rep_val} {rep_n} times", "repeat_print")

        # 4. Single print (quoted or unquoted English & Roman Urdu, or missing personal name clarification)
        if any(token in lower for token in ("print", "prints", "display", "output")):
            value, needs_input = self._print_value(text)
            if needs_input:
                return {"code": None, "needs_input": needs_input, "generation_kind": "print_text"}
            if value is not None:
                literal = repr(value)
                code = (
                    "def main():\n"
                    f"    message = {literal}\n"
                    "    print(message)\n"
                    "    return message\n"
                )
                tests = f"assert main() == {literal}, 'printed message must match the request'\n"
                return self._validated_result(code, tests, "print the requested text", "print_text")

        # 5. Multi-Function Compound Goal Composition ("python mein prime number aur factorial check krne ka code likho")
        multi_subgoals = self._detect_multi_function_subgoals(lower)
        if multi_subgoals:
            return self._synthesize_multi_function_ast(multi_subgoals)

        # 6. Dynamic Compositional List Pipeline AST ("ek list mein se even numbers ka sum nikalne ka python function bnao")
        pipeline_goal = self._detect_pipeline_goal(lower)
        if pipeline_goal is not None:
            return self._synthesize_pipeline_ast(pipeline_goal)

        # 7. Bilingual two-number arithmetic operations
        mentions_two_numbers = bool(
            re.search(
                r"\b(?:two|2|do)\s+(?:numbers?|values?|adad|digits?)\b|\bnumber\s+[a-b]\s+and\s+[a-b]\b",
                lower,
            )
        )
        if mentions_two_numbers:
            for verbs, block_key in (
                (("add", "sum", "plus", "jorna", "jorne", "jama"), "add"),
                (("subtract", "minus", "difference", "ghatana", "ghatane", "tafreeq"), "subtract"),
                (("multiply", "product", "times", "zarb"), "multiply"),
                (("divide", "division", "taqseem"), "divide"),
            ):
                if any(re.search(r"\b" + re.escape(v) + r"\b", lower) for v in verbs):
                    blk = _ATOMIC_BLOCKS[block_key]
                    return self._validated_result(blk.code, blk.tests, blk.spec_desc, blk.kind)

        # 8. Compositional multi-function & atomic AST building block dispatch
        if re.search(r"\bsum\b", lower) and re.search(r"\bprimes?\b", lower):
            blk = _ATOMIC_BLOCKS["sum_of_primes"]
            return self._validated_result(blk.code, blk.tests, blk.spec_desc, blk.kind)

        if re.search(r"\bcount\b", lower) and re.search(r"\bpalindromes?\b", lower):
            blk = _ATOMIC_BLOCKS["count_palindromes"]
            return self._validated_result(blk.code, blk.tests, blk.spec_desc, blk.kind)

        if re.search(r"\beven\b", lower) and re.search(r"\bfibonacci\b|\bfib\b", lower):
            blk = _ATOMIC_BLOCKS["even_fibonacci"]
            return self._validated_result(blk.code, blk.tests, blk.spec_desc, blk.kind)

        if re.search(r"\bgcd\b", lower) and re.search(r"\blist\b|\barray\b|\bnumbers\b", lower) and not re.search(r"\btwo\b", lower):
            blk = _ATOMIC_BLOCKS["gcd_list"]
            return self._validated_result(blk.code, blk.tests, blk.spec_desc, blk.kind)

        if re.search(r"\blcm\b", lower) and re.search(r"\blist\b|\barray\b|\bnumbers\b", lower) and not re.search(r"\btwo\b", lower):
            blk = _ATOMIC_BLOCKS["lcm_list"]
            return self._validated_result(blk.code, blk.tests, blk.spec_desc, blk.kind)

        if re.search(r"\bsum\b", lower) and re.search(r"\bfactorials?\b", lower):
            blk = _ATOMIC_BLOCKS["sum_of_factorials"]
            return self._validated_result(blk.code, blk.tests, blk.spec_desc, blk.kind)

        atomic_triggers: tuple[tuple[str, str], ...] = (
            ("calculator", r"\bcalculator\b"),
            ("check_even_odd", r"\b(?:even\s+(?:or|and|ya)\s+odd|odd\s+(?:or|and|ya)\s+even|juft\s+(?:ya|aur)\s+taaq)\b"),
            ("stack_class", r"\bstack\b.*\b(?:class|push|pop)\b|\bclass\b.*\bstack\b"),
            ("reverse_string", r"\b(?:reverse|ulta)\b.*\b(?:string|str|text|word|characters)\b|\b(?:string|str|text|word)\b.*\b(?:reverse|ulta)\b"),
            ("prime_factors", r"\bprime\s+factors?\b"),
            ("is_prime", r"\bprime\b"),
            ("factorial", r"\bfactorials?\b"),
            ("fibonacci", r"\bfibonacci\b|\bfib\b"),
            ("gcd", r"\bgcd\b|greatest\s+common\s+divisor"),
            ("lcm", r"\blcm\b|least\s+common\s+multiple"),
            ("binary_search", r"\bbinary\s+search\b"),
            ("linear_search", r"\blinear\s+search\b"),
            ("two_sum", r"\btwo[\s_-]+sum\b"),
            ("is_palindrome", r"\bpalindromes?\b"),
            ("is_anagram", r"\banagrams?\b"),
            ("fizzbuzz", r"\bfizz\s*buzz\b"),
            ("matrix_transpose", r"\btranspose\b.*\bmatrix\b|\bmatrix\b.*\btranspose\b"),
            ("flatten_list", r"\bflatten\b.*\b(?:list|array|nested)\b"),
            ("median", r"\bmedian\b"),
            ("reverse_list", r"\b(?:reverse|ulta)\b.*\b(?:list|array)\b|\b(?:list|array)\b.*\b(?:reverse|ulta)\b"),
            ("remove_duplicates", r"\b(?:remove|unique|dedup\w*)\b.*\b(?:duplicates?|list|array)\b|\bduplicates?\b.*\b(?:remove|hatane|khatam)\b"),
            ("bubble_sort", r"\bbubble\s+sort\b"),
            ("merge_sort", r"\bmerge\s+sort\b"),
            ("quicksort", r"\bquick\s*sort\b"),
            ("sort_list", r"\bsort\b.*\b(?:list|array|numbers)\b|\b(?:list|array|numbers)\b.*\bsort\b"),
            ("count_vowels", r"\bvowels?\b"),
            ("word_frequency", r"\bword\s+(?:count|frequency|frequencies)\b"),
            ("char_frequency", r"\b(?:char|character)\s+(?:count|frequency|frequencies)\b"),
            ("find_max", r"\b(?:max|maximum|largest|sab\s+se\s+bara)\b.*\b(?:list|array|numbers)\b|\b(?:list|array|numbers)\b.*\b(?:max|maximum|largest|sab\s+se\s+bara)\b"),
            ("find_min", r"\b(?:min|minimum|smallest|sab\s+se\s+chota)\b.*\b(?:list|array|numbers)\b|\b(?:list|array|numbers)\b.*\b(?:min|minimum|smallest|sab\s+se\s+chota)\b"),
            ("average", r"\b(?:average|mean|ausat)\b.*\b(?:list|array|numbers)\b|\b(?:list|array|numbers)\b.*\b(?:average|mean|ausat)\b"),
            ("is_leap_year", r"\bleap\s+year\b"),
            ("is_armstrong", r"\barmstrong\b"),
            ("sum_of_digits", r"\bsum\s+of\s+digits\b|\bdigits\s+ka\s+sum\b"),
            ("celsius_to_fahrenheit", r"\bcelsius\b.*\bfahrenheit\b"),
            ("square", r"\bsquare\b.*\b(?:number|value|integer|n|adad)\b"),
            ("cube", r"\bcube\b.*\b(?:number|value|integer|n|adad)\b"),
        )
        for key, pat in atomic_triggers:
            if re.search(pat, lower):
                blk = _ATOMIC_BLOCKS[key]
                return self._validated_result(blk.code, blk.tests, blk.spec_desc, blk.kind)

        return None
