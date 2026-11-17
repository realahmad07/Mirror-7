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

        return None
