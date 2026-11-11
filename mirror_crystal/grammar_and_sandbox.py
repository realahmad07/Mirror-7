"""
ByteMarkovGrammar (0-parameter variable-order UTF-8 byte fluency table) &
ZeroTrustSandboxJudge (CPU AST & unit-test reality filter for self-correcting rules).
"""
from __future__ import annotations

import ast
import bisect
import builtins
import cmath
import collections
import copy
import csv
import dataclasses
import decimal
import enum
import fractions
import functools
import heapq
import io
import itertools
import json
import math
import os
import operator
import queue
import random
import re
import statistics
import string
import typing
from collections import defaultdict
from contextlib import redirect_stdout
from typing import Any

_ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "math",
        "cmath",
        "decimal",
        "operator",
        "queue",
        "enum",
        "heapq",
        "collections",
        "bisect",
        "itertools",
        "functools",
        "typing",
        "fractions",
        "statistics",
        "re",
        "json",
        "csv",
        "string",
        "random",
        "copy",
        "dataclasses",
    }
)

_FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
        "exit",
        "quit",
        "_mc_step_tick",
    }
)


class _BoundedStepInjector(ast.NodeTransformer):
    """AST loop & call step-counter guard preventing `while True:` or runaway recursion hangs on CPU."""

    @staticmethod
    def _make_tick_stmt() -> ast.stmt:
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id="_mc_step_tick", ctx=ast.Load()),
                args=[],
                keywords=[],
            )
        )

    def _inject_body(self, body: list[ast.stmt], preserve_docstring: bool = False) -> list[ast.stmt]:
        tick = self._make_tick_stmt()
        if (
            preserve_docstring
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            return [body[0], tick] + body[1:]
        return [tick] + body

    def visit_While(self, node: ast.While) -> ast.AST:
        self.generic_visit(node)
        node.body = self._inject_body(node.body)
        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        self.generic_visit(node)
        node.body = self._inject_body(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._inject_body(node.body, preserve_docstring=True)
        return node


class ByteMarkovGrammar:
    """Compact variable-order (orders 2 & 4) raw UTF-8 byte transition table (0 float weights)."""

    def __init__(self, max_contexts: int = 16384):
        self.max_contexts = int(max_contexts)
        self._order4: dict[bytes, dict[int, int]] = {}
        self._order2: dict[bytes, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.total_bytes_observed: int = 0

    def observe_bytes(self, text: str, max_sample_bytes: int = 512) -> None:
        raw = text.encode("utf-8", errors="ignore")[:max_sample_bytes]
        n = len(raw)
        if n < 3:
            return
        self.total_bytes_observed += n
        for i in range(2, n):
            b_next = raw[i]
            ctx2 = raw[i - 2 : i]
            self._order2[ctx2][b_next] += 1
            if i >= 4:
                ctx4 = raw[i - 4 : i]
                tbl = self._order4.get(ctx4)
                if tbl is None:
                    if len(self._order4) < self.max_contexts:
                        self._order4[ctx4] = {b_next: 1}
                else:
                    tbl[b_next] = tbl.get(b_next, 0) + 1

    def byte_fluency_score(self, text: str) -> float:
        """Return empirical byte transition support in [0.0, 1.0] without neural perplexity."""
        raw = text.encode("utf-8", errors="ignore")[:256]
        if len(raw) < 4 or self.total_bytes_observed == 0:
            return 0.5
        hits = 0
        checks = 0
        for i in range(4, len(raw)):
            checks += 1
            b_next = raw[i]
            ctx4 = raw[i - 4 : i]
            if ctx4 in self._order4 and b_next in self._order4[ctx4]:
                hits += 1
            elif b_next in self._order2.get(raw[i - 2 : i], {}):
                hits += 1
        return hits / max(1, checks)

    def to_state_dict(self) -> dict[str, Any]:
        return {"total_bytes_observed": self.total_bytes_observed}

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> ByteMarkovGrammar:
        obj = cls()
        obj.total_bytes_observed = int(data.get("total_bytes_observed", 0))
        return obj


class ZeroTrustSandboxJudge:
    """Static AST auditor plus bounded OS child-process verifier for code & math checks."""

    @staticmethod
    def is_ast_safe(code_str: str) -> tuple[bool, str]:
        try:
            tree = ast.parse(code_str)
        except SyntaxError as exc:
            return False, f"SyntaxError: {exc}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in _ALLOWED_IMPORTS:
                        return False, f"Blocked import '{alias.name}'"
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] not in _ALLOWED_IMPORTS:
                    return False, f"Blocked from-import '{node.module}'"
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    return False, f"Blocked dunder attribute '{node.attr}'"
            elif isinstance(node, ast.Name):
                if node.id == "_mc_step_tick" or (node.id.startswith("__") and node.id.endswith("__")):
                    return False, f"Blocked dunder/reserved name '{node.id}'"
            elif isinstance(node, ast.FunctionDef) and node.name == "_mc_step_tick":
                return False, "Blocked reserved function name '_mc_step_tick'"
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in _FORBIDDEN_NAMES:
                    return False, f"Blocked call '{node.func.id}()'"
        return True, "OK"

    @classmethod
    def extract_python_code(cls, response_text: str) -> str:
        # Match explicitly tagged python blocks
        fenced = re.findall(r"```(?:python|py)\b[^\n]*\n(.*?)```", response_text, flags=re.DOTALL)
        if fenced:
            return fenced[0].strip()
        # Match untagged code blocks only if they begin with python syntax
        untagged = re.findall(r"```[ \t]*\n(.*?)```", response_text, flags=re.DOTALL)
        for block in untagged:
            stripped = block.strip()
            if any(stripped.startswith(k) for k in ("def ", "class ", "import ", "from ", "#")):
                return stripped
        cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
        if any(cleaned.startswith(k) for k in ("def ", "class ", "import ", "from ")):
            return cleaned
        return ""

    @staticmethod
    def _static_loop_limit_violation(code_str: str, max_loop_steps: int) -> str | None:
        """Reject statically provable oversized or non-terminating loops before spawning a child."""
        try:
            tree = ast.parse(code_str)
        except (SyntaxError, RecursionError, MemoryError):
            return None  # The ordinary AST safety check reports malformed input.

        def const_int(node: ast.AST) -> int | None:
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
                return node.value
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = const_int(node.operand)
                if value is not None:
                    return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Pow)):
                left, right = const_int(node.left), const_int(node.right)
                if left is None or right is None:
                    return None
                if isinstance(node.op, ast.Add):
                    value = left + right
                elif isinstance(node.op, ast.Sub):
                    value = left - right
                elif isinstance(node.op, ast.Mult):
                    value = left * right
                else:
                    if right < 0 or right > 32:
                        return None
                    value = left ** right
                return value if value.bit_length() <= 256 else None
            return None

        def has_exit_from_loop(body: list[ast.stmt]) -> bool:
            pending: list[ast.AST] = list(body)
            while pending:
                current = pending.pop()
                if isinstance(current, (ast.Break, ast.Return, ast.Raise)):
                    return True
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                    continue
                pending.extend(ast.iter_child_nodes(current))
            return False

        def scan_block(body: list[ast.stmt]) -> str | None:
            for index, statement in enumerate(body):
                if isinstance(statement, ast.For) and isinstance(statement.iter, ast.Call):
                    call = statement.iter
                    if isinstance(call.func, ast.Name) and call.func.id == "range":
                        values = [const_int(arg) for arg in call.args]
                        if values and all(value is not None for value in values):
                            try:
                                span = len(range(*values))
                            except (OverflowError, ValueError):
                                span = max_loop_steps + 1
                            if span > max_loop_steps:
                                return f"RangeSpanLimitExceeded ({span} > {max_loop_steps})"

                if isinstance(statement, ast.While) and not has_exit_from_loop(statement.body):
                    test = statement.test
                    if isinstance(test, ast.Constant) and test.value is True:
                        return "LoopStepLimitExceeded (statically proven while True loop)"

                    # Prove the common monotonic integer loop case, e.g.
                    # x = 1; while x > 0: x += 1. Python integers do not overflow.
                    if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                            and len(test.ops) == 1 and len(test.comparators) == 1):
                        variable = test.left.id
                        bound = const_int(test.comparators[0])
                        op = test.ops[0]
                        initial = None
                        for prior in reversed(body[:index]):
                            targets: list[ast.expr] = []
                            value_node: ast.expr | None = None
                            if isinstance(prior, ast.Assign):
                                targets = list(prior.targets)
                                value_node = prior.value
                            elif isinstance(prior, ast.AnnAssign):
                                targets = [prior.target]
                                value_node = prior.value
                            if value_node is not None and any(
                                isinstance(target, ast.Name) and target.id == variable for target in targets
                            ):
                                initial = const_int(value_node)
                                break

                        delta = None
                        for item in statement.body:
                            if (isinstance(item, ast.AugAssign) and isinstance(item.target, ast.Name)
                                    and item.target.id == variable):
                                amount = const_int(item.value)
                                if amount is not None and isinstance(item.op, (ast.Add, ast.Sub)):
                                    delta = amount if isinstance(item.op, ast.Add) else -amount
                                    break
                        if initial is not None and bound is not None and delta is not None:
                            increasing_forever = (
                                delta > 0 and (
                                    (isinstance(op, ast.Gt) and initial > bound) or
                                    (isinstance(op, ast.GtE) and initial >= bound)
                                )
                            )
                            decreasing_forever = (
                                delta < 0 and (
                                    (isinstance(op, ast.Lt) and initial < bound) or
                                    (isinstance(op, ast.LtE) and initial <= bound)
                                )
                            )
                            if increasing_forever or decreasing_forever:
                                return "LoopStepLimitExceeded (statically proven monotonic non-terminating loop)"

                # Recurse into nested statement suites (function bodies, branches, etc.).
                for _, value in ast.iter_fields(statement):
                    if isinstance(value, list):
                        nested = [item for item in value if isinstance(item, ast.stmt)]
                        if nested:
                            issue = scan_block(nested)
                            if issue:
                                return issue
                    elif isinstance(value, ast.stmt):
                        issue = scan_block([value])
                        if issue:
                            return issue
            return None

        return scan_block(tree.body)

    @classmethod
    def verify_code_and_test(
        cls,
        response_text: str,
        test_code: str = "",
        max_loop_steps: int = 50_000,
    ) -> tuple[bool, str]:
        """AST-audit in the parent and execute in Docker/Podman by default.

        MIRROR_SANDBOX_MODE=process is an explicit, lower-isolation development/test
        override. Production deployments must leave the mode at container.
        """
        code = cls.extract_python_code(response_text)
        if not code and not test_code:
            return True, "non_code"
        full_snippet = (code + "\n" + test_code).strip()
        safe, reason = cls.is_ast_safe(full_snippet)
        if not safe:
            return False, reason
        loop_violation = cls._static_loop_limit_violation(full_snippet, max_loop_steps)
        if loop_violation:
            return False, loop_violation

        mode = os.environ.get("MIRROR_SANDBOX_MODE", "container").strip().lower()
        if mode == "container":
            from .container_sandbox import run_python_in_container
            result = run_python_in_container(full_snippet, max_loop_steps=max_loop_steps)
            if not result.ok:
                return False, result.error or "container execution failed"
            return True, result.output or "PASS"
        if mode == "process":
            from .os_process_sandbox import run_python_isolated
            result = run_python_isolated(full_snippet, max_loop_steps=max_loop_steps)
            if not result.ok:
                return False, result.error or "process execution failed"
            return True, result.output or "PASS"
        return False, f"unsupported MIRROR_SANDBOX_MODE '{mode}'; refusing to execute"
