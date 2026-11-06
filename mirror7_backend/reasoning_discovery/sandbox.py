"""
Security Sandbox for Executing and Testing Synthesized Reasoning Strategies.
"""
from __future__ import annotations

import ast
import math
import time
from typing import Any, Callable, Sequence
from .schema import ExecutionResult, ReasoningStrategy, TestCase

FORBIDDEN_BUILTINS = {
    "open", "eval", "exec", "__import__", "compile", "globals", "locals",
    "getattr", "setattr", "delattr", "input", "print", "breakpoint", "exit", "quit"
}

FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "socket", "urllib", "requests", "http",
    "shutil", "pathlib", "importlib", "threading", "multiprocessing", "ctypes"
}

def _safe_import(name, *args, **kwargs):
    if name == "math":
        return math
    raise SecurityViolation(f"Forbidden import: {name}")


SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "math": math,
    "__import__": _safe_import,
}


class SecurityViolation(Exception):
    pass


class ASTSecurityScanner(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] in FORBIDDEN_MODULES:
                raise SecurityViolation(f"Forbidden import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.split(".")[0] in FORBIDDEN_MODULES:
            raise SecurityViolation(f"Forbidden import from module: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_BUILTINS:
                raise SecurityViolation(f"Forbidden function call: {node.func.id}()")
        self.generic_visit(node)


class SafeSandbox:
    """Restricted sandbox for evaluating algorithm code with strict isolation."""

    def __init__(self, timeout_seconds: float = 1.0):
        self.timeout_seconds = timeout_seconds

    def scan_security(self, code: str) -> None:
        """Parse and scan code AST for security policy violations."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in strategy code: {e}") from e

        scanner = ASTSecurityScanner()
        scanner.visit(tree)

    def execute(
        self,
        strategy: ReasoningStrategy,
        test_cases: Sequence[TestCase],
    ) -> ExecutionResult:
        # 1. Static security scan
        try:
            self.scan_security(strategy.procedure_code)
        except SecurityViolation as sv:
            return ExecutionResult(
                passed_tests=0,
                total_tests=len(test_cases),
                execution_time_ms=0.0,
                peak_memory_kb=0.0,
                is_correct=False,
                error_message=f"Security violation: {sv}",
            )
        except Exception as e:
            return ExecutionResult(
                passed_tests=0,
                total_tests=len(test_cases),
                execution_time_ms=0.0,
                peak_memory_kb=0.0,
                is_correct=False,
                error_message=f"Static analysis failed: {e}",
            )

        # 2. Compile in restricted environment
        safe_env = {"__builtins__": SAFE_BUILTINS}
        try:
            compiled = compile(strategy.procedure_code, "<strategy_sandbox>", "exec")
            exec(compiled, safe_env)
        except Exception as e:
            return ExecutionResult(
                passed_tests=0,
                total_tests=len(test_cases),
                execution_time_ms=0.0,
                peak_memory_kb=0.0,
                is_correct=False,
                error_message=f"Compilation/Initialization error: {e}",
            )

        entry_fn = safe_env.get(strategy.entry_point)
        if not entry_fn or not callable(entry_fn):
            return ExecutionResult(
                passed_tests=0,
                total_tests=len(test_cases),
                execution_time_ms=0.0,
                peak_memory_kb=0.0,
                is_correct=False,
                error_message=f"Entry point '{strategy.entry_point}' not found or not callable",
            )

        # 3. Execute test cases with timeout enforcement
        passed = 0
        outputs: list[Any] = []
        t0 = time.perf_counter()

        for tc in test_cases:
            if (time.perf_counter() - t0) > self.timeout_seconds:
                return ExecutionResult(
                    passed_tests=passed,
                    total_tests=len(test_cases),
                    execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                    peak_memory_kb=0.0,
                    is_correct=False,
                    error_message=f"Execution timed out after {self.timeout_seconds}s",
                    test_outputs=tuple(outputs),
                )

            try:
                out = entry_fn(*tc.input_args)
                outputs.append(out)
                if out == tc.expected_output:
                    passed += 1
            except Exception as e:
                outputs.append(None)
                return ExecutionResult(
                    passed_tests=passed,
                    total_tests=len(test_cases),
                    execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                    peak_memory_kb=0.0,
                    is_correct=False,
                    error_message=f"Runtime error on test case: {e}",
                    test_outputs=tuple(outputs),
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        is_correct = passed == len(test_cases) and len(test_cases) > 0
        return ExecutionResult(
            passed_tests=passed,
            total_tests=len(test_cases),
            execution_time_ms=round(elapsed_ms, 3),
            peak_memory_kb=0.0,
            is_correct=is_correct,
            error_message=None if is_correct else f"Failed {len(test_cases)-passed} test cases",
            test_outputs=tuple(outputs),
        )
