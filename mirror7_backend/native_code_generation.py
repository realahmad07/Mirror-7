"""Small deterministic Python code builder for beginner-sized, explicit requests.

This is deliberately a narrow native capability, not a general natural-language
programming model. It builds only recognized templates and returns their tests.
"""
from __future__ import annotations

import ast
import re
from typing import Any


class NativeCodeGenerationProvider:
    """Construct a few explicit Python tasks and provide non-vacuous checks."""

    @staticmethod
    def _print_value(prompt: str) -> tuple[str | None, str | None]:
        # A quoted string after an explicit print/display instruction is unambiguous.
        quoted = re.search(
            r"""\b(?:print|prints|display|output)\b.*?(['"])(.*?)\1""",
            prompt,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if quoted:
            return quoted.group(2), None

        # "my name" must be supplied in the request; Mirror will not guess it.
        name = re.search(
            r"\bmy\s+name(?:\s+is|\s*:)?\s+(.+?)"
            r"(?=\s+(?:in|using|with)\s+python\b|[?.!,]|$)",
            prompt,
            flags=re.IGNORECASE,
        )
        if name:
            value = name.group(1).strip().strip("\"'")
            if value and value.casefold() not in {"in python", "using python"}:
                return value, None
        if re.search(r"\bmy\s+name\b", prompt, flags=re.IGNORECASE):
            return None, "What name should the Python program print? Please include the name."
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
            raise ValueError("native code template must define a testable function")
        # Apply Mirror's existing static safety gate before returning generated code.
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

    def __call__(self, prompt: str) -> dict[str, Any] | None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        text = prompt.strip()
        lower = text.casefold()

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
                return self._validated_result(
                    code, tests, "print the requested text", "print_text"
                )

        operation_specs = (
            (("add", "sum", "plus"), "add", "a + b", (
                "assert add(2, 3) == 5\n"
                "assert add(-4, 1) == -3\n"
                "assert add(0, 0) == 0\n"
            ), "add two numbers"),
            (("subtract", "minus", "difference"), "subtract", "a - b", (
                "assert subtract(7, 2) == 5\n"
                "assert subtract(2, 7) == -5\n"
                "assert subtract(0, 0) == 0\n"
            ), "subtract two numbers"),
            (("multiply", "product", "times"), "multiply", "a * b", (
                "assert multiply(3, 4) == 12\n"
                "assert multiply(-2, 5) == -10\n"
                "assert multiply(0, 9) == 0\n"
            ), "multiply two numbers"),
        )
        mentions_two_numbers = bool(re.search(
            r"\b(?:two|2)\s+(?:numbers|values)\b|\bnumber\s+[a-b]\s+and\s+[a-b]\b",
            lower,
        ))
        if mentions_two_numbers:
            for verbs, function_name, expression, tests, spec in operation_specs:
                if any(re.search(r"\b" + re.escape(verb) + r"\b", lower) for verb in verbs):
                    code = (
                        f"def {function_name}(a, b):\n"
                        f"    return {expression}\n"
                    )
                    return self._validated_result(code, tests, spec, function_name)

        square_request = (
            re.search(r"\bsquare\b", lower)
            and re.search(r"\b(?:number|value|integer|n)\b", lower)
        )
        if square_request:
            code = "def square(n):\n    return n * n\n"
            tests = (
                "assert square(0) == 0\n"
                "assert square(3) == 9\n"
                "assert square(-4) == 16\n"
            )
            return self._validated_result(code, tests, "square a number", "square")

        # Reverse string / text
        if re.search(r"\breverse\b", lower) and re.search(r"\b(?:string|str|text|word|characters)\b", lower):
            code = "def reverse_string(s):\n    return s[::-1]\n"
            tests = (
                "assert reverse_string('hello') == 'olleh'\n"
                "assert reverse_string('') == ''\n"
                "assert reverse_string('Mirror') == 'rorriM'\n"
            )
            return self._validated_result(code, tests, "reverse a string", "reverse_string")

        # Prime number check
        if re.search(r"\bprime\b", lower) and any(w in lower for w in ("check", "checks", "test", "is", "whether", "function")):
            code = (
                "def is_prime(n):\n"
                "    if n < 2:\n"
                "        return False\n"
                "    for i in range(2, int(n ** 0.5) + 1):\n"
                "        if n % i == 0:\n"
                "            return False\n"
                "    return True\n"
            )
            tests = (
                "assert is_prime(2) is True\n"
                "assert is_prime(3) is True\n"
                "assert is_prime(4) is False\n"
                "assert is_prime(17) is True\n"
                "assert is_prime(1) is False\n"
            )
            return self._validated_result(code, tests, "check if a number is prime", "is_prime")

        # Factorial
        if re.search(r"\bfactorial\b", lower):
            code = (
                "def factorial(n):\n"
                "    if n < 0:\n"
                "        raise ValueError('n must be non-negative')\n"
                "    res = 1\n"
                "    for i in range(2, n + 1):\n"
                "        res *= i\n"
                "    return res\n"
            )
            tests = (
                "assert factorial(0) == 1\n"
                "assert factorial(1) == 1\n"
                "assert factorial(5) == 120\n"
            )
            return self._validated_result(code, tests, "calculate factorial", "factorial")

        # Fibonacci
        if re.search(r"\bfibonacci\b|\bfib\b", lower):
            code = (
                "def fibonacci(n):\n"
                "    if n < 0:\n"
                "        raise ValueError('n must be non-negative')\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a\n"
            )
            tests = (
                "assert fibonacci(0) == 0\n"
                "assert fibonacci(1) == 1\n"
                "assert fibonacci(7) == 13\n"
                "assert fibonacci(10) == 55\n"
            )
            return self._validated_result(code, tests, "compute nth fibonacci number", "fibonacci")

        # GCD (Greatest Common Divisor)
        if re.search(r"\bgcd\b|greatest common divisor", lower):
            code = (
                "def gcd(a, b):\n"
                "    a, b = abs(a), abs(b)\n"
                "    while b:\n"
                "        a, b = b, a % b\n"
                "    return a\n"
            )
            tests = (
                "assert gcd(48, 18) == 6\n"
                "assert gcd(17, 13) == 1\n"
                "assert gcd(0, 5) == 5\n"
            )
            return self._validated_result(code, tests, "compute greatest common divisor", "gcd")

        # LCM (Least Common Multiple)
        if re.search(r"\blcm\b|least common multiple", lower):
            code = (
                "def lcm(a, b):\n"
                "    if a == 0 or b == 0:\n"
                "        return 0\n"
                "    x, y = abs(a), abs(b)\n"
                "    while y:\n"
                "        x, y = y, x % y\n"
                "    return abs(a * b) // x\n"
            )
            tests = (
                "assert lcm(4, 6) == 12\n"
                "assert lcm(5, 7) == 35\n"
                "assert lcm(0, 9) == 0\n"
            )
            return self._validated_result(code, tests, "compute least common multiple", "lcm")

        # Binary search
        if re.search(r"binary\s+search", lower):
            code = (
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
            )
            tests = (
                "assert binary_search([1, 3, 5, 7, 9], 7) == 3\n"
                "assert binary_search([1, 3, 5, 7, 9], 2) == -1\n"
                "assert binary_search([], 4) == -1\n"
            )
            return self._validated_result(code, tests, "binary search on sorted list", "binary_search")

        # Two sum
        if re.search(r"two[\s_-]+sum", lower):
            code = (
                "def two_sum(nums, target):\n"
                "    seen = {}\n"
                "    for i, num in enumerate(nums):\n"
                "        need = target - num\n"
                "        if need in seen:\n"
                "            return [seen[need], i]\n"
                "        seen[num] = i\n"
                "    return []\n"
            )
            tests = (
                "assert two_sum([2, 7, 11, 15], 9) == [0, 1]\n"
                "assert two_sum([3, 2, 4], 6) == [1, 2]\n"
                "assert two_sum([3, 3], 6) == [0, 1]\n"
            )
            return self._validated_result(code, tests, "find two distinct indices summing to target", "two_sum")

        # Palindrome check
        if re.search(r"\bpalindrome\b", lower):
            code = (
                "def is_palindrome(text):\n"
                "    cleaned = ''.join(ch.lower() for ch in text if ch.isalnum())\n"
                "    return cleaned == cleaned[::-1]\n"
            )
            tests = (
                "assert is_palindrome('racecar') is True\n"
                "assert is_palindrome('A man, a plan, a canal: Panama') is True\n"
                "assert is_palindrome('mirror') is False\n"
            )
            return self._validated_result(code, tests, "check if text is a palindrome", "is_palindrome")

        # Anagram check
        if re.search(r"\banagram\b", lower):
            code = (
                "def is_anagram(a, b):\n"
                "    ca = sorted(ch.lower() for ch in a if ch.isalnum())\n"
                "    cb = sorted(ch.lower() for ch in b if ch.isalnum())\n"
                "    return ca == cb\n"
            )
            tests = (
                "assert is_anagram('listen', 'silent') is True\n"
                "assert is_anagram('hello', 'world') is False\n"
                "assert is_anagram('Dormitory', 'Dirty room') is True\n"
            )
            return self._validated_result(code, tests, "check if two strings are anagrams", "is_anagram")

        # Prime factors
        if re.search(r"prime\s+factors?", lower):
            code = (
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
            )
            tests = (
                "assert prime_factors(12) == [2, 2, 3]\n"
                "assert prime_factors(13) == [13]\n"
                "assert prime_factors(1) == []\n"
            )
            return self._validated_result(code, tests, "compute prime factorization", "prime_factors")

        # FizzBuzz
        if re.search(r"\bfizz\s*buzz\b", lower):
            code = (
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
            )
            tests = (
                "assert fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']\n"
                "assert fizzbuzz(15)[-1] == 'FizzBuzz'\n"
            )
            return self._validated_result(code, tests, "generate fizzbuzz sequence", "fizzbuzz")

        # Matrix transpose
        if re.search(r"transpose", lower) and re.search(r"matrix", lower):
            code = (
                "def matrix_transpose(matrix):\n"
                "    if not matrix:\n"
                "        return []\n"
                "    return [list(row) for row in zip(*matrix)]\n"
            )
            tests = (
                "assert matrix_transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]\n"
                "assert matrix_transpose([]) == []\n"
            )
            return self._validated_result(code, tests, "transpose a 2D matrix", "matrix_transpose")

        # Median
        if re.search(r"\bmedian\b", lower):
            code = (
                "def median(nums):\n"
                "    if not nums:\n"
                "        raise ValueError('nums must not be empty')\n"
                "    s = sorted(nums)\n"
                "    mid = len(s) // 2\n"
                "    if len(s) % 2 == 1:\n"
                "        return s[mid]\n"
                "    return (s[mid - 1] + s[mid]) / 2\n"
            )
            tests = (
                "assert median([3, 1, 2]) == 2\n"
                "assert median([1, 2, 3, 4]) == 2.5\n"
                "assert median([7]) == 7\n"
            )
            return self._validated_result(code, tests, "compute median of numeric list", "median")

        return None
