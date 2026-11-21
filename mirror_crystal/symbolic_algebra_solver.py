"""Native Pure-Python Symbolic Algebra & Linear System Solver (Zero 3rd-Party Dependencies).

Solves:
1. Arbitrary linear equations in one variable (e.g., "Solve 3x + 7 = 22", "2(x - 3) + 4 = 10")
2. Quadratic equations with real or complex roots (e.g., "Solve x^2 - 5x + 6 = 0")
3. N x N linear systems via exact rational Gaussian elimination (e.g., "Solve system: 2x + y = 10 and x - y = 2")

Every solution is independently self-verified by substituting back into every
equation (LHS == RHS) before being marked verified.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
import math
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class AlgebraSolveResult:
    solved: bool
    kind: str = ""  # "linear", "quadratic", "linear_system", "no_solution", "unmatched"
    solutions: Mapping[str, Any] = None  # type: ignore[assignment]
    verified: bool = False
    explanation: str = ""

    def __post_init__(self):
        if self.solutions is None:
            object.__setattr__(self, "solutions", {})


def _preprocess_math(expr: str) -> str:
    s = expr.strip().replace("^", "**")
    # Insert implicit multiplication: 3x -> 3*x, 2(x+1) -> 2*(x+1), )x -> )*x, )( -> )*(
    s = re.sub(r"(\d)\s*([A-Za-z(])", r"\1*\2", s)
    s = re.sub(r"(\))\s*([A-Za-z0-9(])", r"\1*\2", s)
    return s


def _poly_add(a: dict[tuple[tuple[str, int], ...], Fraction], b: dict[tuple[tuple[str, int], ...], Fraction], sign: int = 1):
    out = dict(a)
    for mon, coeff in b.items():
        val = out.get(mon, Fraction(0)) + sign * coeff
        if val == 0:
            out.pop(mon, None)
        else:
            out[mon] = val
    return out


def _poly_mul(a: dict[tuple[tuple[str, int], ...], Fraction], b: dict[tuple[tuple[str, int], ...], Fraction]):
    out: dict[tuple[tuple[str, int], ...], Fraction] = {}
    for m1, c1 in a.items():
        for m2, c2 in b.items():
            combined: dict[str, int] = dict(m1)
            for v, p in m2:
                combined[v] = combined.get(v, 0) + p
            key = tuple(sorted((v, p) for v, p in combined.items() if p != 0))
            val = out.get(key, Fraction(0)) + c1 * c2
            if val == 0:
                out.pop(key, None)
            else:
                out[key] = val
    return out


def _ast_to_poly(node: ast.AST) -> dict[tuple[tuple[str, int], ...], Fraction]:
    if isinstance(node, ast.Expression):
        return _ast_to_poly(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("non-numeric constant")
        val = Fraction(str(node.value))
        return {(): val} if val != 0 else {}
    if isinstance(node, ast.Name):
        return {((node.id, 1),): Fraction(1)}
    if isinstance(node, ast.UnaryOp):
        p = _ast_to_poly(node.operand)
        if isinstance(node.op, ast.UAdd):
            return p
        if isinstance(node.op, ast.USub):
            return {k: -v for k, v in p.items()}
        raise ValueError("unsupported unary op")
    if isinstance(node, ast.BinOp):
        left = _ast_to_poly(node.left)
        right = _ast_to_poly(node.right)
        if isinstance(node.op, ast.Add):
            return _poly_add(left, right, 1)
        if isinstance(node.op, ast.Sub):
            return _poly_add(left, right, -1)
        if isinstance(node.op, ast.Mult):
            return _poly_mul(left, right)
        if isinstance(node.op, ast.Div):
            if len(right) == 1 and () in right and right[()] != 0:
                inv = Fraction(1, 1) / right[()]
                return {k: v * inv for k, v in left.items()}
            raise ValueError("non-constant division not supported in polynomial form")
        if isinstance(node.op, ast.Pow):
            if len(right) == 1 and () in right and right[()].denominator == 1 and 0 <= int(right[()]) <= 4:
                exp = int(right[()])
                res: dict[tuple[tuple[str, int], ...], Fraction] = {(): Fraction(1)}
                for _ in range(exp):
                    res = _poly_mul(res, left)
                return res
            raise ValueError("exponent must be integer 0..4")
    raise ValueError(f"unsupported expression: {type(node).__name__}")


def _parse_equation_poly(eq_str: str) -> dict[tuple[tuple[str, int], ...], Fraction]:
    if "=" not in eq_str:
        raise ValueError("equation must contain '='")
    lhs_s, rhs_s = eq_str.split("=", 1)
    lhs = _ast_to_poly(ast.parse(_preprocess_math(lhs_s), mode="eval"))
    rhs = _ast_to_poly(ast.parse(_preprocess_math(rhs_s), mode="eval"))
    return _poly_add(lhs, rhs, -1)


def _eval_poly(poly: dict[tuple[tuple[str, int], ...], Fraction], assignment: Mapping[str, complex | float | Fraction]) -> complex:
    total: complex = 0j
    for mon, coeff in poly.items():
        term: complex = complex(float(coeff), 0.0)
        for v, p in mon:
            term *= complex(assignment[v]) ** p
        total += term
    return total


def _fmt_frac(val: Fraction) -> str:
    if val.denominator == 1:
        return str(val.numerator)
    return f"{val.numerator}/{val.denominator} ({float(val):.6g})"


def solve_single_equation(eq_str: str) -> AlgebraSolveResult:
    """Solve a linear or quadratic equation in one variable and self-verify LHS == RHS."""
    try:
        poly = _parse_equation_poly(eq_str)
    except (SyntaxError, ValueError) as exc:
        return AlgebraSolveResult(False, kind="unmatched", explanation=str(exc))

    variables = sorted({v for mon in poly for v, _ in mon})
    if len(variables) != 1:
        return AlgebraSolveResult(False, kind="unmatched", explanation="expected exactly one variable")
    var = variables[0]
    max_deg = max((p for mon in poly for _, p in mon), default=0)

    c0 = poly.get((), Fraction(0))
    c1 = poly.get(((var, 1),), Fraction(0))
    c2 = poly.get(((var, 2),), Fraction(0))

    if max_deg == 1:
        if c1 == 0:
            return AlgebraSolveResult(False, kind="no_solution", explanation="degenerate linear equation")
        root = -c0 / c1
        verified = abs(_eval_poly(poly, {var: root})) < 1e-9
        return AlgebraSolveResult(
            solved=verified,
            kind="linear",
            solutions={var: float(root) if root.denominator != 1 else int(root.numerator)},
            verified=verified,
            explanation=f"Verified linear solution: {var} = {_fmt_frac(root)} (checked LHS == RHS).",
        )

    if max_deg == 2:
        if c2 == 0:
            return AlgebraSolveResult(False, kind="no_solution", explanation="degenerate quadratic")
        disc = c1 * c1 - 4 * c2 * c0
        a_f, b_f, d_f = float(c2), float(c1), float(disc)
        if disc >= 0:
            sq = math.sqrt(d_f)
            r1 = (-b_f - sq) / (2.0 * a_f)
            r2 = (-b_f + sq) / (2.0 * a_f)
            roots = [r1] if math.isclose(r1, r2, rel_tol=1e-9, abs_tol=1e-9) else sorted([r1, r2])
            verified = all(abs(_eval_poly(poly, {var: r})) < 1e-7 for r in roots)
            clean_roots = [int(round(r)) if abs(r - round(r)) < 1e-9 else round(r, 8) for r in roots]
            roots_str = ", ".join(str(r) for r in clean_roots)
            return AlgebraSolveResult(
                solved=verified,
                kind="quadratic",
                solutions={var: clean_roots},
                verified=verified,
                explanation=f"Verified quadratic roots (discriminant={float(disc):g}): {var} = {roots_str} (checked LHS == RHS).",
            )
        else:
            real_part = -b_f / (2.0 * a_f)
            imag_part = math.sqrt(-d_f) / (2.0 * a_f)
            r1 = complex(real_part, -imag_part)
            r2 = complex(real_part, imag_part)
            verified = abs(_eval_poly(poly, {var: r1})) < 1e-7 and abs(_eval_poly(poly, {var: r2})) < 1e-7
            return AlgebraSolveResult(
                solved=verified,
                kind="quadratic",
                solutions={var: [r1, r2]},
                verified=verified,
                explanation=f"Verified complex quadratic roots: {var} = {real_part:g} ± {imag_part:g}i (checked LHS == RHS).",
            )

    return AlgebraSolveResult(False, kind="unmatched", explanation=f"degree {max_deg} polynomial not supported")


def solve_linear_system(equations: list[str]) -> AlgebraSolveResult:
    """Solve an N x N linear system via exact rational Gaussian elimination and self-verify."""
    if not 2 <= len(equations) <= 6:
        return AlgebraSolveResult(False, kind="unmatched", explanation="system supports 2 to 6 equations")
    try:
        polys = [_parse_equation_poly(eq) for eq in equations]
    except (SyntaxError, ValueError) as exc:
        return AlgebraSolveResult(False, kind="unmatched", explanation=str(exc))

    # Ensure every equation is purely linear
    for poly in polys:
        for mon in poly:
            if sum(p for _, p in mon) > 1:
                return AlgebraSolveResult(False, kind="unmatched", explanation="non-linear term in system")

    variables = sorted({v for poly in polys for mon in poly for v, _ in mon})
    n = len(variables)
    if n != len(polys):
        return AlgebraSolveResult(False, kind="unmatched", explanation=f"expected {n} equations for {n} variables")

    # Build augmented matrix [A | b] where A*x = b (poly is A*x - b = 0, so b = -c0)
    mat: list[list[Fraction]] = []
    for poly in polys:
        row = [poly.get(((v, 1),), Fraction(0)) for v in variables]
        row.append(-poly.get((), Fraction(0)))
        mat.append(row)

    # Gaussian elimination with partial pivoting
    for col in range(n):
        pivot = next((r for r in range(col, n) if mat[r][col] != 0), None)
        if pivot is None:
            return AlgebraSolveResult(False, kind="no_solution", explanation="singular or dependent linear system")
        mat[col], mat[pivot] = mat[pivot], mat[col]
        piv_val = mat[col][col]
        mat[col] = [x / piv_val for x in mat[col]]
        for r in range(n):
            if r != col and mat[r][col] != 0:
                factor = mat[r][col]
                mat[r] = [mat[r][c] - factor * mat[col][c] for c in range(n + 1)]

    frac_sol = {variables[i]: mat[i][n] for i in range(n)}
    verified = all(abs(_eval_poly(poly, frac_sol)) < 1e-9 for poly in polys)
    clean_sol = {
        v: int(val.numerator) if val.denominator == 1 else float(val)
        for v, val in frac_sol.items()
    }
    sol_str = ", ".join(f"{v} = {_fmt_frac(frac_sol[v])}" for v in variables)
    return AlgebraSolveResult(
        solved=verified,
        kind="linear_system",
        solutions=clean_sol,
        verified=verified,
        explanation=f"Verified linear system solution: {sol_str} (all {n} equations verified LHS == RHS).",
    )


def solve_algebra_prompt(prompt: str) -> AlgebraSolveResult:
    """Detect and solve algebraic equations or linear systems from plain text."""
    text = prompt.strip()
    if "=" not in text:
        return AlgebraSolveResult(False, kind="unmatched")

    # Strip leading instruction phrases like "Solve system:", "Solve for x:", "Solve"
    cleaned = re.sub(
        r"^\s*(?:solve|find|compute|calculate)\s+(?:the\s+)?(?:system(?:\s+of\s+equations)?|equation|for\s+[A-Za-z_][A-Za-z0-9_]*\s*(?:in|:)?)?\s*[:\-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip().rstrip("?.")

    # Check if multiple equations separated by "and", ";", or ","
    parts = [p.strip() for p in re.split(r"\s+\band\b\s+|;|,", cleaned) if "=" in p]
    if len(parts) >= 2:
        return solve_linear_system(parts)
    if len(parts) == 1:
        return solve_single_equation(parts[0])
    return AlgebraSolveResult(False, kind="unmatched")
