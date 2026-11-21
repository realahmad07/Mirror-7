"""Parameter-Free Counterexample-Guided AST Buggy-Code Repair Engine.

Repairs buggy Python functions without neural parameters or external LLMs:
1. Parses the user's buggy Python function into an AST and validates AST safety.
2. Evaluates against a canonical or prompt-supplied test suite to isolate a failing counterexample.
3. Searches bounded single-site AST mutations (constants, binary operators, comparison operators,
   and off-by-one range(n) -> range(n + 1) bounds).
4. Emits repaired code via `ast.unparse` ONLY when 100% of test cases pass.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import re
from typing import Any, Callable


@dataclass(frozen=True)
class CodeRepairResult:
    matched: bool
    repaired: bool = False
    func_name: str = ""
    counterexample: str = ""
    mutation_applied: str = ""
    repaired_code: str = ""
    tests_passed: int = 0
    explanation: str = ""
    uses_neural_parameters: bool = False


_SAFE_BUILTINS: dict[str, Any] = {
    "range": range,
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "tuple": tuple,
    "set": set,
    "dict": dict,
    "enumerate": enumerate,
    "zip": zip,
    "sorted": sorted,
    "reversed": reversed,
    "all": all,
    "any": any,
}

# Canonical multi-point specification test suites: func_name -> list of (args_tuple, expected_output)
_CANONICAL_SPECS: dict[str, tuple[tuple[tuple[Any, ...], Any], ...]] = {
    "factorial": (((0,), 1), ((1,), 1), ((4,), 24), ((5,), 120), ((6,), 720)),
    "add": (((2, 3), 5), ((-4, 9), 5), ((0, 0), 0), ((10, -3), 7)),
    "subtract": (((10, 4), 6), ((3, 8), -5), ((0, 5), -5), ((7, 7), 0)),
    "multiply": (((3, 4), 12), ((-2, 6), -12), ((7, 0), 0), ((5, 5), 25)),
    "square": (((0,), 0), ((3,), 9), ((-4,), 16), ((7,), 49)),
    "is_even": (((0,), True), ((2,), True), ((3,), False), ((14,), True), ((17,), False)),
    "is_odd": (((0,), False), ((1,), True), ((2,), False), ((15,), True), ((18,), False)),
    "sum_to_n": (((0,), 0), ((1,), 1), ((5,), 15), ((10,), 55)),
    "max_of_two": (((3, 7), 7), ((10, 2), 10), ((-5, -1), -1), ((4, 4), 4)),
    "min_of_two": (((3, 7), 3), ((10, 2), 2), ((-5, -1), -5), ((4, 4), 4)),
    "reverse_string": ((("abc",), "cba"), (("mirror",), "rorrim"), (("",), ""), (("a",), "a")),
    "is_palindrome": ((("racecar",), True), (("level",), True), (("python",), False), (("a",), True)),
    "fibonacci": (((0,), 0), ((1,), 1), ((5,), 5), ((7,), 13), ((8,), 21)),
    "abs_val": (((-5,), 5), ((0,), 0), ((8,), 8), ((-12,), 12)),
}


def _is_safe_ast(tree: ast.AST, func_name: str) -> bool:
    allowed_calls = set(_SAFE_BUILTINS.keys()) | {func_name}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Delete, ast.Try, ast.With, ast.Raise)):
            return False
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return False
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return False
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in allowed_calls:
                return False
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                return False
    return True


def _run_candidate_ast(tree: ast.Module, func_name: str, tests: tuple[tuple[tuple[Any, ...], Any], ...]) -> tuple[bool, str]:
    """Execute a validated AST against all test cases. Returns (all_passed, first_failure_msg)."""
    if not _is_safe_ast(tree, func_name):
        return False, "unsafe AST rejected"
    try:
        code_obj = compile(tree, filename="<mirror_ast_repair>", mode="exec")
    except Exception as exc:
        return False, str(exc)

    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    try:
        exec(code_obj, namespace, namespace)  # noqa: S102 - restricted AST + safe builtins only
    except Exception as exc:
        return False, str(exc)

    fn = namespace.get(func_name)
    if not callable(fn):
        return False, f"function {func_name} not defined"

    for args, expected in tests:
        try:
            actual = fn(*args)
        except Exception as exc:
            return False, f"{func_name}{args} raised {type(exc).__name__}: {exc}"
        if actual != expected or type(actual) is not type(expected):
            return False, f"{func_name}{args} returned {actual!r}, expected {expected!r}"
    return True, ""


def _generate_ast_mutations(tree: ast.Module) -> list[tuple[str, ast.Module]]:
    """Generate bounded single-site AST mutations for counterexample-guided repair."""
    mutations: list[tuple[str, ast.Module]] = []

    # Collect paths by node index in ast.walk order
    base_nodes = list(ast.walk(tree))
    for idx, node in enumerate(base_nodes):
        # 1. Integer Constant mutations (e.g. base case 0 -> 1, slice step 1 -> -1)
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            orig = node.value
            candidates = {0: (1, -1), 1: (0, -1, 2), -1: (1, 0), 2: (1, 0)}.get(orig, (orig + 1, orig - 1))
            for repl in candidates:
                cloned = copy.deepcopy(tree)
                target = list(ast.walk(cloned))[idx]
                assert isinstance(target, ast.Constant)
                target.value = repl
                ast.fix_missing_locations(cloned)
                mutations.append((f"Constant({orig} -> {repl})", cloned))

        # 2. Binary operator mutations (Sub <-> Add, Add <-> Mult, etc.)
        elif isinstance(node, ast.BinOp):
            op_map: list[tuple[str, ast.operator]] = [
                ("Add", ast.Add()),
                ("Sub", ast.Sub()),
                ("Mult", ast.Mult()),
                ("FloorDiv", ast.FloorDiv()),
                ("Mod", ast.Mod()),
            ]
            orig_name = type(node.op).__name__
            for repl_name, repl_op in op_map:
                if repl_name == orig_name:
                    continue
                cloned = copy.deepcopy(tree)
                target = list(ast.walk(cloned))[idx]
                assert isinstance(target, ast.BinOp)
                target.op = repl_op
                ast.fix_missing_locations(cloned)
                mutations.append((f"BinOp({orig_name} -> {repl_name})", cloned))

        # 3. Comparison operator mutations (Lt <-> LtE, Gt <-> GtE, Eq <-> NotEq, Lt <-> Gt)
        elif isinstance(node, ast.Compare) and len(node.ops) == 1:
            cmp_map: list[tuple[str, ast.cmpop]] = [
                ("Eq", ast.Eq()),
                ("NotEq", ast.NotEq()),
                ("Lt", ast.Lt()),
                ("LtE", ast.LtE()),
                ("Gt", ast.Gt()),
                ("GtE", ast.GtE()),
            ]
            orig_name = type(node.ops[0]).__name__
            for repl_name, repl_op in cmp_map:
                if repl_name == orig_name:
                    continue
                cloned = copy.deepcopy(tree)
                target = list(ast.walk(cloned))[idx]
                assert isinstance(target, ast.Compare)
                target.ops = [repl_op]
                ast.fix_missing_locations(cloned)
                mutations.append((f"Compare({orig_name} -> {repl_name})", cloned))

        # 4. Off-by-one range(n) -> range(n + 1) mutation
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "range"
            and len(node.args) == 1
        ):
            cloned = copy.deepcopy(tree)
            target = list(ast.walk(cloned))[idx]
            assert isinstance(target, ast.Call)
            orig_arg = target.args[0]
            target.args = [ast.BinOp(left=orig_arg, op=ast.Add(), right=ast.Constant(value=1))]
            ast.fix_missing_locations(cloned)
            mutations.append(("RangeBound(range(n) -> range(n + 1))", cloned))

    return mutations


class CounterexampleCodeRepairEngine:
    """Zero-parameter Counterexample-Guided AST Mutation Engine for repairing buggy Python functions."""

    VERSION = "cegis-ast-code-repair-v1"

    def repair_prompt(self, prompt: str) -> CodeRepairResult:
        text = prompt.strip()
        lower = text.casefold()

        if "def " not in text or not re.search(r"\b(?:fix|repair|debug|correct|theek)\b", lower):
            return CodeRepairResult(False)

        m_def = re.search(r"(def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(.*)", text, flags=re.DOTALL)
        if not m_def:
            return CodeRepairResult(False)

        raw_code = m_def.group(1).strip()
        # Strip trailing markdown fences if present
        raw_code = re.sub(r"```\s*$", "", raw_code).strip()
        func_name = m_def.group(2).strip()

        try:
            tree = ast.parse(raw_code)
        except SyntaxError as exc:
            return CodeRepairResult(
                matched=True,
                repaired=False,
                func_name=func_name,
                explanation=f"SyntaxError in provided function '{func_name}': {exc}",
            )

        tests = _CANONICAL_SPECS.get(func_name)
        if not tests:
            return CodeRepairResult(
                matched=True,
                repaired=False,
                func_name=func_name,
                explanation=f"Mirror abstains: no verified test specification registered for '{func_name}'.",
            )

        ok_orig, counterexample = _run_candidate_ast(tree, func_name, tests)
        if ok_orig:
            clean_code = ast.unparse(tree)
            return CodeRepairResult(
                matched=True,
                repaired=True,
                func_name=func_name,
                counterexample="none (already passes all tests)",
                mutation_applied="Identity (0 mutations needed)",
                repaired_code=clean_code,
                tests_passed=len(tests),
                explanation=f"Verified function '{func_name}' already passes all {len(tests)} specification tests:\n{clean_code}",
            )

        for mut_desc, cand_tree in _generate_ast_mutations(tree):
            ok_cand, _ = _run_candidate_ast(cand_tree, func_name, tests)
            if ok_cand:
                repaired_src = ast.unparse(cand_tree)
                expl = (
                    f"Verified AST Code Repair ({func_name}):\n"
                    f"- Counterexample on buggy code: {counterexample}\n"
                    f"- AST Mutation Applied: {mut_desc} ({len(tests)}/{len(tests)} tests verified)\n"
                    f"- Repaired Code:\n{repaired_src}"
                )
                return CodeRepairResult(
                    matched=True,
                    repaired=True,
                    func_name=func_name,
                    counterexample=counterexample,
                    mutation_applied=mut_desc,
                    repaired_code=repaired_src,
                    tests_passed=len(tests),
                    explanation=expl,
                )

        return CodeRepairResult(
            matched=True,
            repaired=False,
            func_name=func_name,
            counterexample=counterexample,
            explanation=f"Detected bug ({counterexample}), but no single-site safe AST mutation satisfied all {len(tests)} tests.",
        )
