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
        if re.search(r"\b(?:my\s+name|mera\s+naam)\b", prompt, flags=re.IGNORECASE):
            return None, "What name should the Python program print? Please include the name."

        # Roman-Urdu / Hinglish: prioritize explicit "jo <X> print" before "mein <X> ko print"
        urdu_jo = re.search(
            r"\bjo\s+([A-Za-z0-9_][A-Za-z0-9_ .\-]{0,60}?)(?:\s+ko)?\s+print\s*(?:kry|kare|kre|kro|karo|karna|krna|karne|krne|ho|hoga)?\b",
            prompt,
            flags=re.IGNORECASE,
        )
        if urdu_jo:
            value = urdu_jo.group(1).strip().strip("\"'")
            if value and value.casefold() not in {"code", "program", "python", "script", "function", "text", "output", "numbers", "number", "table"}:
                return value, None

        urdu_mein = re.search(
            r"\b(?:mein|me|mn)\s+([A-Za-z0-9_][A-Za-z0-9_ .\-]{0,60}?)(?:\s+ko)?\s+print\s*(?:kry|kare|kre|kro|karo|karna|krna|karne|krne|ho|hoga)?\b",
            prompt,
            flags=re.IGNORECASE,
        )
        if urdu_mein:
            value = urdu_mein.group(1).strip().strip("\"'")
            if value and value.casefold() not in {"code", "program", "python", "script", "function", "text", "output", "numbers", "number", "table"}:
                return value, None

        # English / mixed: "print Ahmad", "prints Ahmad in python", "display Hello World"
        plain_after = re.search(
            r"\b(?:print|prints|display)\s+(?!the\b|a\b|an\b|my\b|your\b|text\b|message\b|output\b|something\b|name\b|numbers?\b|all\b|even\b|odd\b|table\b)"
            r"([A-Za-z0-9_][A-Za-z0-9_ .\-]{0,60}?)"
            r"(?=\s+(?:in|using|with|via|by|on|for)\s+python\b|\s+(?:\d+\s+(?:times|bar|baar|martaba)|kro|karo|kry|kare|kre)\b|[?.!,]|$)",
            prompt,
            flags=re.IGNORECASE,
        )
        if plain_after:
            value = plain_after.group(1).strip().strip("\"'")
            if value and value.casefold() not in {"python", "code", "program", "script", "function", "numbers", "table"}:
                return value, None
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

        # Multiplication table ("5 ka table print kro", "multiplication table of 7")
        m_table = re.search(
            r"(?:\b(\d+)\s+ka\s+(?:table|pahara)\b|\b(?:multiplication\s+)?table\s+(?:of|for)\s+(\d+)\b)",
            lower,
        )
        if m_table:
            n_val = int(m_table.group(1) or m_table.group(2))
            code = (
                f"def multiplication_table(n={n_val}, upto=10):\n"
                "    lines = [f'{n} x {i} = {n * i}' for i in range(1, upto + 1)]\n"
                "    for line in lines:\n"
                "        print(line)\n"
                "    return lines\n"
            )
            tests = (
                f"res = multiplication_table({n_val}, 10)\n"
                f"assert len(res) == 10\n"
                f"assert res[0] == '{n_val} x 1 = {n_val}'\n"
                f"assert res[-1] == '{n_val} x 10 = {n_val * 10}'\n"
            )
            return self._validated_result(code, tests, f"print multiplication table of {n_val}", "multiplication_table")

        # Number range printing ("1 se 10 tak numbers print kro", "print numbers from 1 to 10")
        m_range = re.search(
            r"\b(?:from\s+)?(\d+)\s+(?:to|se)\s+(\d+)(?:\s+tak)?\b.*\b(?:print|numbers?|counting|ginti)\b"
            r"|\b(?:print|display)\s+(?:numbers?\s+)?(?:from\s+)?(\d+)\s+(?:to|se)\s+(\d+)\b",
            lower,
        )
        if m_range:
            a_val = int(m_range.group(1) or m_range.group(3))
            b_val = int(m_range.group(2) or m_range.group(4))
            if abs(b_val - a_val) <= 1000:
                step = 1 if b_val >= a_val else -1
                code = (
                    f"def print_range(start={a_val}, end={b_val}):\n"
                    f"    step = 1 if end >= start else -1\n"
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

        if any(token in lower for token in ("print", "prints", "display", "output")):
            # Check repeated print ("print Ahmad 5 times", "Ahmad ko 5 bar print kro", "python mein Ahmad ko 5 bar print kro")
            m_rep = (
                re.search(
                    r"\b(?:jo|mein|me|mn)\s+['\"]?([A-Za-z0-9_][A-Za-z0-9_ .\-]{0,40}?)['\"]?(?:\s+ko)?\s+(\d+)\s+(?:times|bar|baar|martaba)\s+print\b",
                    text,
                    flags=re.IGNORECASE,
                )
                or re.search(
                    r"\b(?:print|prints|display)\s+['\"]?([A-Za-z0-9_][A-Za-z0-9_ .\-]{0,40}?)['\"]?\s+(\d+)\s+(?:times|bar|baar|martaba)\b",
                    text,
                    flags=re.IGNORECASE,
                )
                or re.search(
                    r"^\s*['\"]?([A-Za-z0-9_][A-Za-z0-9_ .\-]{0,40}?)['\"]?(?:\s+ko)?\s+(\d+)\s+(?:times|bar|baar|martaba)\s+print\b",
                    text,
                    flags=re.IGNORECASE,
                )
            )
            if m_rep:
                rep_val = (m_rep.group(1) or "").strip().strip("\"'")
                rep_n = int(m_rep.group(2))
                if rep_val and 1 <= rep_n <= 500:
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
            (("add", "sum", "plus", "jorna", "jama"), "add", "a + b", (
                "assert add(2, 3) == 5\n"
                "assert add(-4, 1) == -3\n"
                "assert add(0, 0) == 0\n"
            ), "add two numbers"),
            (("subtract", "minus", "difference", "ghatana", "tafreeq"), "subtract", "a - b", (
                "assert subtract(7, 2) == 5\n"
                "assert subtract(2, 7) == -5\n"
                "assert subtract(0, 0) == 0\n"
            ), "subtract two numbers"),
            (("multiply", "product", "times", "zarb"), "multiply", "a * b", (
                "assert multiply(3, 4) == 12\n"
                "assert multiply(-2, 5) == -10\n"
                "assert multiply(0, 9) == 0\n"
            ), "multiply two numbers"),
            (("divide", "division", "taqseem"), "divide", "a / b if b != 0 else None", (
                "assert divide(10, 2) == 5.0\n"
                "assert divide(7, 2) == 3.5\n"
                "assert divide(5, 0) is None\n"
            ), "safely divide two numbers"),
        )
        mentions_two_numbers = bool(re.search(
            r"\b(?:two|2|do)\s+(?:numbers?|values?|adad|digits?)\b|\bnumber\s+[a-b]\s+and\s+[a-b]\b",
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

        # Simple Calculator
        if re.search(r"\bcalculator\b", lower):
            code = (
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
            )
            tests = (
                "assert calculate(10, 5, '+') == 15\n"
                "assert calculate(10, 5, '-') == 5\n"
                "assert calculate(10, 5, '*') == 50\n"
                "assert calculate(10, 5, '/') == 2.0\n"
            )
            return self._validated_result(code, tests, "implement a basic calculator", "calculator")

        # Even or Odd check
        if re.search(r"\b(?:even\s+(?:or|and|ya)\s+odd|odd\s+(?:or|and|ya)\s+even|juft\s+(?:ya|aur)\s+taaq)\b", lower):
            code = (
                "def check_even_odd(n):\n"
                "    return 'even' if n % 2 == 0 else 'odd'\n"
            )
            tests = (
                "assert check_even_odd(4) == 'even'\n"
                "assert check_even_odd(7) == 'odd'\n"
                "assert check_even_odd(0) == 'even'\n"
            )
            return self._validated_result(code, tests, "check whether a number is even or odd", "check_even_odd")

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

        # ---------------- Compositional Multi-Function Code Builders ----------------
        # Composes Mirror's verified building blocks into compound algorithms.

        # 1. Sum of primes up to n (composes is_prime + sum)
        if re.search(r"\bsum\b", lower) and re.search(r"\bprimes?\b", lower):
            code = (
                "def is_prime(n):\n"
                "    if n < 2:\n"
                "        return False\n"
                "    for i in range(2, int(n ** 0.5) + 1):\n"
                "        if n % i == 0:\n"
                "            return False\n"
                "    return True\n\n"
                "def sum_of_primes(limit):\n"
                "    return sum(k for k in range(2, limit + 1) if is_prime(k))\n"
            )
            tests = (
                "assert sum_of_primes(1) == 0\n"
                "assert sum_of_primes(10) == 17\n"
                "assert sum_of_primes(20) == 77\n"
            )
            return self._validated_result(code, tests, "compose is_prime into sum_of_primes", "sum_of_primes")

        # 2. Count palindromes in a list (composes is_palindrome + count)
        if re.search(r"\bcount\b", lower) and re.search(r"\bpalindromes?\b", lower):
            code = (
                "def is_palindrome(text):\n"
                "    cleaned = ''.join(ch.lower() for ch in text if ch.isalnum())\n"
                "    return cleaned == cleaned[::-1]\n\n"
                "def count_palindromes(words):\n"
                "    return sum(1 for w in words if is_palindrome(w))\n"
            )
            tests = (
                "assert count_palindromes(['level', 'mirror', 'radar', '']) == 3\n"
                "assert count_palindromes(['abc', 'def']) == 0\n"
                "assert count_palindromes([]) == 0\n"
            )
            return self._validated_result(code, tests, "compose is_palindrome into count_palindromes", "count_palindromes")

        # 3. Even Fibonacci numbers up to limit (composes fibonacci generator + even filter)
        if re.search(r"\beven\b", lower) and re.search(r"\bfibonacci\b|\bfib\b", lower):
            code = (
                "def even_fibonacci(limit):\n"
                "    out = []\n"
                "    a, b = 0, 1\n"
                "    while a <= limit:\n"
                "        if a % 2 == 0:\n"
                "            out.append(a)\n"
                "        a, b = b, a + b\n"
                "    return out\n"
            )
            tests = (
                "assert even_fibonacci(10) == [0, 2, 8]\n"
                "assert even_fibonacci(40) == [0, 2, 8, 34]\n"
                "assert even_fibonacci(0) == [0]\n"
            )
            return self._validated_result(code, tests, "filter even fibonacci numbers up to limit", "even_fibonacci")

        # 4. GCD of a list of numbers (composes gcd + fold)
        if re.search(r"\bgcd\b", lower) and re.search(r"\blist\b|\barray\b|\bnumbers\b", lower) and not re.search(r"\btwo\b", lower):
            code = (
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
            )
            tests = (
                "assert gcd_list([12, 18, 24]) == 6\n"
                "assert gcd_list([7, 13, 19]) == 1\n"
                "assert gcd_list([42]) == 42\n"
            )
            return self._validated_result(code, tests, "compose gcd over a list", "gcd_list")

        # 5. LCM of a list of numbers (composes lcm + fold)
        if re.search(r"\blcm\b", lower) and re.search(r"\blist\b|\barray\b|\bnumbers\b", lower) and not re.search(r"\btwo\b", lower):
            code = (
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
            )
            tests = (
                "assert lcm_list([4, 6, 8]) == 24\n"
                "assert lcm_list([3, 5, 7]) == 105\n"
                "assert lcm_list([9]) == 9\n"
            )
            return self._validated_result(code, tests, "compose lcm over a list", "lcm_list")

        # 6. Sum of factorials 1! + 2! + ... + n! (composes factorial + sum)
        if re.search(r"\bsum\b", lower) and re.search(r"\bfactorials?\b", lower):
            code = (
                "def factorial(n):\n"
                "    res = 1\n"
                "    for i in range(2, n + 1):\n"
                "        res *= i\n"
                "    return res\n\n"
                "def sum_of_factorials(n):\n"
                "    if n < 1:\n"
                "        return 0\n"
                "    return sum(factorial(i) for i in range(1, n + 1))\n"
            )
            tests = (
                "assert sum_of_factorials(0) == 0\n"
                "assert sum_of_factorials(3) == 9\n"
                "assert sum_of_factorials(5) == 153\n"
            )
            return self._validated_result(code, tests, "compose factorial into sum_of_factorials", "sum_of_factorials")

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

        # Reverse a list ("python mein list reverse krne ka code btao", "reverse a list")
        if re.search(r"\b(?:reverse|ulta)\b", lower) and re.search(r"\b(?:list|array)\b", lower):
            code = (
                "def reverse_list(items):\n"
                "    return list(items)[::-1]\n"
            )
            tests = (
                "assert reverse_list([1, 2, 3, 4]) == [4, 3, 2, 1]\n"
                "assert reverse_list([]) == []\n"
            )
            return self._validated_result(code, tests, "reverse a list", "reverse_list")

        # Remove duplicates from list
        if re.search(r"\b(?:remove|unique|dedup\w*)\b", lower) and re.search(r"\b(?:duplicate|duplicates|list|array)\b", lower):
            code = (
                "def remove_duplicates(items):\n"
                "    seen = set()\n"
                "    result = []\n"
                "    for item in items:\n"
                "        if item not in seen:\n"
                "            seen.add(item)\n"
                "            result.append(item)\n"
                "    return result\n"
            )
            tests = (
                "assert remove_duplicates([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]\n"
                "assert remove_duplicates([]) == []\n"
            )
            return self._validated_result(code, tests, "remove duplicates preserving order", "remove_duplicates")

        # Bubble / Merge / Quick sort or sort a list (deliberately skips broad requests like "complete sorting library")
        if not re.search(r"\blibrary\b", lower):
            if re.search(r"\bbubble\s+sort\b", lower):
                code = (
                    "def bubble_sort(items):\n"
                    "    arr = list(items)\n"
                    "    n = len(arr)\n"
                    "    for i in range(n):\n"
                    "        for j in range(0, n - i - 1):\n"
                    "            if arr[j] > arr[j + 1]:\n"
                    "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
                    "    return arr\n"
                )
                tests = (
                    "assert bubble_sort([5, 1, 4, 2, 8]) == [1, 2, 4, 5, 8]\n"
                    "assert bubble_sort([]) == []\n"
                )
                return self._validated_result(code, tests, "implement bubble sort", "bubble_sort")

            if re.search(r"\bmerge\s+sort\b", lower):
                code = (
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
                )
                tests = (
                    "assert merge_sort([38, 27, 43, 3, 9, 82, 10]) == [3, 9, 10, 27, 38, 43, 82]\n"
                    "assert merge_sort([]) == []\n"
                )
                return self._validated_result(code, tests, "implement merge sort", "merge_sort")

            if re.search(r"\bquick\s*sort\b", lower):
                code = (
                    "def quicksort(items):\n"
                    "    if len(items) <= 1:\n"
                    "        return list(items)\n"
                    "    pivot = items[len(items) // 2]\n"
                    "    left = [x for x in items if x < pivot]\n"
                    "    middle = [x for x in items if x == pivot]\n"
                    "    right = [x for x in items if x > pivot]\n"
                    "    return quicksort(left) + middle + quicksort(right)\n"
                )
                tests = (
                    "assert quicksort([3, 6, 8, 10, 1, 2, 1]) == [1, 1, 2, 3, 6, 8, 10]\n"
                    "assert quicksort([]) == []\n"
                )
                return self._validated_result(code, tests, "implement quicksort", "quicksort")

            if re.search(r"\bsort\b", lower) and re.search(r"\b(?:list|array|numbers)\b", lower):
                code = (
                    "def sort_list(items, reverse=False):\n"
                    "    return sorted(items, reverse=reverse)\n"
                )
                tests = (
                    "assert sort_list([4, 1, 3, 2]) == [1, 2, 3, 4]\n"
                    "assert sort_list([4, 1, 3, 2], reverse=True) == [4, 3, 2, 1]\n"
                )
                return self._validated_result(code, tests, "sort a list of items", "sort_list")

        # Count vowels
        if re.search(r"\bvowels?\b", lower):
            code = (
                "def count_vowels(text):\n"
                "    return sum(1 for ch in text.lower() if ch in 'aeiou')\n"
            )
            tests = (
                "assert count_vowels('Ahmad') == 2\n"
                "assert count_vowels('Mirror') == 2\n"
                "assert count_vowels('xyz') == 0\n"
            )
            return self._validated_result(code, tests, "count vowels in a string", "count_vowels")

        # Word count / frequency
        if re.search(r"\bword\s+(?:count|frequency|frequencies)\b", lower):
            code = (
                "def word_frequency(text):\n"
                "    counts = {}\n"
                "    for word in text.lower().split():\n"
                "        clean = ''.join(ch for ch in word if ch.isalnum())\n"
                "        if clean:\n"
                "            counts[clean] = counts.get(clean, 0) + 1\n"
                "    return counts\n"
            )
            tests = (
                "assert word_frequency('AI is fast and AI is clean') == {'ai': 2, 'is': 2, 'fast': 1, 'and': 1, 'clean': 1}\n"
            )
            return self._validated_result(code, tests, "count word frequencies in text", "word_frequency")

        # Find max / min / average / sum of a list
        if re.search(r"\b(?:max|maximum|largest|sab\s+se\s+bara)\b", lower) and re.search(r"\b(?:list|array|numbers)\b", lower):
            code = (
                "def find_max(nums):\n"
                "    if not nums:\n"
                "        raise ValueError('nums must not be empty')\n"
                "    best = nums[0]\n"
                "    for n in nums[1:]:\n"
                "        if n > best:\n"
                "            best = n\n"
                "    return best\n"
            )
            tests = (
                "assert find_max([3, 9, 2, 15, 7]) == 15\n"
                "assert find_max([-5, -1, -9]) == -1\n"
            )
            return self._validated_result(code, tests, "find the largest number in a list", "find_max")

        if re.search(r"\b(?:min|minimum|smallest|sab\s+se\s+chota)\b", lower) and re.search(r"\b(?:list|array|numbers)\b", lower):
            code = (
                "def find_min(nums):\n"
                "    if not nums:\n"
                "        raise ValueError('nums must not be empty')\n"
                "    best = nums[0]\n"
                "    for n in nums[1:]:\n"
                "        if n < best:\n"
                "            best = n\n"
                "    return best\n"
            )
            tests = (
                "assert find_min([3, 9, 2, 15, 7]) == 2\n"
                "assert find_min([-5, -1, -9]) == -9\n"
            )
            return self._validated_result(code, tests, "find the smallest number in a list", "find_min")

        if re.search(r"\b(?:average|mean|ausat)\b", lower) and re.search(r"\b(?:list|array|numbers)\b", lower):
            code = (
                "def average(nums):\n"
                "    if not nums:\n"
                "        raise ValueError('nums must not be empty')\n"
                "    return sum(nums) / len(nums)\n"
            )
            tests = (
                "assert average([10, 20, 30]) == 20.0\n"
                "assert average([5]) == 5.0\n"
            )
            return self._validated_result(code, tests, "compute average of a list of numbers", "average")

        if re.search(r"\bleap\s+year\b", lower):
            code = (
                "def is_leap_year(year):\n"
                "    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)\n"
            )
            tests = (
                "assert is_leap_year(2024) is True\n"
                "assert is_leap_year(1900) is False\n"
                "assert is_leap_year(2000) is True\n"
            )
            return self._validated_result(code, tests, "check whether a year is a leap year", "is_leap_year")

        return None
