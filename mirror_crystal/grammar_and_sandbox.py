"""
ByteMarkovGrammar (0-parameter variable-order UTF-8 byte fluency table) &
ZeroTrustSandboxJudge (CPU AST & unit-test reality filter for self-correcting rules).
"""
from __future__ import annotations

import ast
import bisect
import collections
import fractions
import functools
import heapq
import io
import itertools
import math
import re
import statistics
import typing
from collections import defaultdict
from contextlib import redirect_stdout
from typing import Any

_ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {"math", "heapq", "collections", "bisect", "itertools", "functools", "typing", "fractions", "statistics"}
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
    """Static AST auditor + isolated CPU execution verifier for code & math reality checks."""

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
        fenced = re.findall(r"```(?:python)?\s*(.*?)```", response_text, flags=re.DOTALL)
        if fenced:
            return fenced[0].strip()
        cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
        if any(cleaned.startswith(k) for k in ("def ", "class ", "import ", "from ")):
            return cleaned
        return ""

    @classmethod
    def verify_code_and_test(
        cls,
        response_text: str,
        test_code: str = "",
        max_loop_steps: int = 50_000,
    ) -> tuple[bool, str]:
        code = cls.extract_python_code(response_text)
        if not code and not test_code:
            return True, "non_code"
        full_snippet = (code + "\n" + test_code).strip()
        safe, reason = cls.is_ast_safe(full_snippet)
        if not safe:
            return False, reason

        step_cap = max(100, int(max_loop_steps))
        steps_box = [0]

        def _mc_step_tick() -> None:
            s = steps_box[0] + 1
            if s > step_cap:
                raise RuntimeError(f"LoopStepLimitExceeded ({step_cap} steps)")
            steps_box[0] = s

        def _safe_range(*args: int) -> range:
            r = range(*args)
            if len(r) > step_cap:
                raise RuntimeError(f"RangeSpanLimitExceeded ({len(r)} > {step_cap})")
            return r

        tree = ast.parse(full_snippet)
        guarded_tree = _BoundedStepInjector().visit(tree)
        ast.fix_missing_locations(guarded_tree)
        compiled_code = compile(guarded_tree, "<mirror_sandbox>", "exec")

        mod_map = {
            "math": math,
            "heapq": heapq,
            "collections": collections,
            "bisect": bisect,
            "itertools": itertools,
            "functools": functools,
            "typing": typing,
            "fractions": fractions,
            "statistics": statistics,
        }
        safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "divmod": divmod,
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
            "pow": pow,
            "print": print,
            "range": _safe_range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "ZeroDivisionError": ZeroDivisionError,
            "IndexError": IndexError,
            "KeyError": KeyError,
            "RuntimeError": RuntimeError,
            "AssertionError": AssertionError,
            "__import__": lambda name, *a, **k: mod_map[name.split(".")[0]],
        }
        ns: dict[str, Any] = {"__builtins__": safe_builtins, "_mc_step_tick": _mc_step_tick}
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(compiled_code, ns)
            return True, buf.getvalue().strip() or "PASS"
        except Exception as exc:
            return False, f"ExecError: {exc}"
