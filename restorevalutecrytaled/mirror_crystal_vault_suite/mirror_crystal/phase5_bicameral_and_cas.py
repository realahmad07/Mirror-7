"""
Mirror Crystal Phase-5 Cognitive Organs & Phase-6 Bottleneck Guards (100% Pure Python 3.11+ Stdlib, 0 GPU):
  1. SymbolicCASOrgan (+ CASComplexityExplosionGuard & DimensionalMismatchGuard):
     - Exact rational (Fraction) differentiation, polynomial/elementary integration, linear & quadratic
       equations, N x N linear systems (Gauss-Jordan), and dimensional unit conversions with independent proof checks.
  2. PackedTernaryProjection & TernaryFluidComposer (+ FactFidelityGuard & CreativeFactualBleedGuard):
     - Bicameral Left+Right Brain fusion: 1.58-bit ternary {-1, 0, +1} 2-bit packed integer projection (4 weights/byte,
       0 float multiplies) + online collocation phrase lattice for grounded rhetorical styling & creative synthesis.
  3. ActiveCuriosityPlanner (+ GenuineAmbiguityFilter & CuriosityLoopGuard):
     - Information-gain diagnostic clarifier over genuinely distinct LSH candidate branches with 1-turn loop cap.
  4. DocumentAutoCrystallizer (+ DocBoilerplateAndSafetyGuard):
     - Unstructured Markdown, textbook, plain-text, and Python (.py AST + doctest) auto-crystallizer.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import re
from typing import Any

from .grammar_and_sandbox import ZeroTrustSandboxJudge

# ============================================================================
# 1. PHASE-5 #1 + PHASE-6 B6-3: SymbolicCASOrgan + CASComplexityExplosionGuard
# ============================================================================

_IMPLICIT_MUL_NUM_VAR = re.compile(r"(\d)\s*([a-zA-Z(])")
_IMPLICIT_MUL_PAREN = re.compile(r"(\))\s*([a-zA-Z0-9(])")

_DIFF_RE = re.compile(
    r"^\s*(?:compute\s+|find\s+|what\s+is\s+)?(?:the\s+)?"
    r"(?:differentiate|derivative\s+of|d/d([a-zA-Z]))\s+(.+?)"
    r"(?:\s+(?:with\s+respect\s+to|wrt)\s+([a-zA-Z]))?\s*\??\s*$",
    re.I,
)
_INTEG_DEF_RE = re.compile(
    r"^\s*(?:compute\s+|evaluate\s+|find\s+)?(?:the\s+)?"
    r"(?:definite\s+)?(?:integrate|integral\s+of)\s+(.+?)\s+from\s+"
    r"([+-]?\d+(?:/\d+|\.\d+)?)\s+to\s+([+-]?\d+(?:/\d+|\.\d+)?)"
    r"(?:\s+(?:with\s+respect\s+to|wrt|d)\s*([a-zA-Z]))?\s*\??\s*$",
    re.I,
)
_INTEG_INDEF_RE = re.compile(
    r"^\s*(?:compute\s+|find\s+|what\s+is\s+)?(?:the\s+)?"
    r"(?:indefinite\s+)?(?:integrate|integral\s+of|antiderivative\s+of)\s+(.+?)"
    r"(?:\s+(?:with\s+respect\s+to|wrt)\s+([a-zA-Z]))?\s*\??\s*$",
    re.I,
)
_SOLVE_RE = re.compile(
    r"^\s*(?:solve(?:\s+for\s+[a-zA-Z,\s]+)?|find\s+roots?\s+of)\s*[:\-]?\s*(.+?)\s*\??\s*$",
    re.I,
)
_CONVERT_RE = re.compile(
    r"^\s*(?:convert\s+|what\s+is\s+)?([+-]?\d+(?:\.\d+)?)\s*"
    r"([a-zA-Z°/]+)\s+(?:to|in|into)\s+([a-zA-Z°/]+)\s*\??\s*$",
    re.I,
)

# Canonical unit table: alias -> (dimension, factor_to_base_SI_or_anchor)
# Temperature handled separately due to affine offset.
_UNIT_TABLE: dict[str, tuple[str, Fraction]] = {
    # Length (base: meter)
    "mm": ("length", Fraction(1, 1000)),
    "millimeter": ("length", Fraction(1, 1000)),
    "millimeters": ("length", Fraction(1, 1000)),
    "cm": ("length", Fraction(1, 100)),
    "centimeter": ("length", Fraction(1, 100)),
    "centimeters": ("length", Fraction(1, 100)),
    "m": ("length", Fraction(1, 1)),
    "meter": ("length", Fraction(1, 1)),
    "meters": ("length", Fraction(1, 1)),
    "km": ("length", Fraction(1000, 1)),
    "kilometer": ("length", Fraction(1000, 1)),
    "kilometers": ("length", Fraction(1000, 1)),
    "in": ("length", Fraction(254, 10000)),
    "inch": ("length", Fraction(254, 10000)),
    "inches": ("length", Fraction(254, 10000)),
    "ft": ("length", Fraction(3048, 10000)),
    "foot": ("length", Fraction(3048, 10000)),
    "feet": ("length", Fraction(3048, 10000)),
    "yd": ("length", Fraction(9144, 10000)),
    "yard": ("length", Fraction(9144, 10000)),
    "yards": ("length", Fraction(9144, 10000)),
    "mi": ("length", Fraction(1609344, 1000)),
    "mile": ("length", Fraction(1609344, 1000)),
    "miles": ("length", Fraction(1609344, 1000)),
    # Mass (base: kilogram)
    "mg": ("mass", Fraction(1, 1_000_000)),
    "g": ("mass", Fraction(1, 1000)),
    "gram": ("mass", Fraction(1, 1000)),
    "grams": ("mass", Fraction(1, 1000)),
    "kg": ("mass", Fraction(1, 1)),
    "kilogram": ("mass", Fraction(1, 1)),
    "kilograms": ("mass", Fraction(1, 1)),
    "lb": ("mass", Fraction(45359237, 100_000_000)),
    "lbs": ("mass", Fraction(45359237, 100_000_000)),
    "pound": ("mass", Fraction(45359237, 100_000_000)),
    "pounds": ("mass", Fraction(45359237, 100_000_000)),
    "oz": ("mass", Fraction(45359237, 1_600_000_000)),
    # Time (base: second)
    "ms": ("time", Fraction(1, 1000)),
    "millisecond": ("time", Fraction(1, 1000)),
    "milliseconds": ("time", Fraction(1, 1000)),
    "s": ("time", Fraction(1, 1)),
    "sec": ("time", Fraction(1, 1)),
    "secs": ("time", Fraction(1, 1)),
    "second": ("time", Fraction(1, 1)),
    "seconds": ("time", Fraction(1, 1)),
    "min": ("time", Fraction(60, 1)),
    "mins": ("time", Fraction(60, 1)),
    "minute": ("time", Fraction(60, 1)),
    "minutes": ("time", Fraction(60, 1)),
    "h": ("time", Fraction(3600, 1)),
    "hr": ("time", Fraction(3600, 1)),
    "hrs": ("time", Fraction(3600, 1)),
    "hour": ("time", Fraction(3600, 1)),
    "hours": ("time", Fraction(3600, 1)),
    "day": ("time", Fraction(86400, 1)),
    "days": ("time", Fraction(86400, 1)),
    # Speed (base: m/s)
    "m/s": ("speed", Fraction(1, 1)),
    "km/h": ("speed", Fraction(5, 18)),
    "kmph": ("speed", Fraction(5, 18)),
    "kph": ("speed", Fraction(5, 18)),
    "mph": ("speed", Fraction(1609344, 3_600_000)),
    # Digital storage (base: byte)
    "b": ("data", Fraction(1, 1)),
    "byte": ("data", Fraction(1, 1)),
    "bytes": ("data", Fraction(1, 1)),
    "kb": ("data", Fraction(1000, 1)),
    "mb": ("data", Fraction(1_000_000, 1)),
    "gb": ("data", Fraction(1_000_000_000, 1)),
    "tb": ("data", Fraction(1_000_000_000_000, 1)),
    "kib": ("data", Fraction(1024, 1)),
    "mib": ("data", Fraction(1024**2, 1)),
    "gib": ("data", Fraction(1024**3, 1)),
    "tib": ("data", Fraction(1024**4, 1)),
    # Energy (base: Joule)
    "j": ("energy", Fraction(1, 1)),
    "joule": ("energy", Fraction(1, 1)),
    "joules": ("energy", Fraction(1, 1)),
    "kj": ("energy", Fraction(1000, 1)),
    "cal": ("energy", Fraction(4184, 1000)),
    "kcal": ("energy", Fraction(4184, 1)),
    "wh": ("energy", Fraction(3600, 1)),
    "kwh": ("energy", Fraction(3_600_000, 1)),
}

_TEMP_ALIASES: dict[str, str] = {
    "c": "c",
    "°c": "c",
    "celsius": "c",
    "f": "f",
    "°f": "f",
    "fahrenheit": "f",
    "k": "k",
    "kelvin": "k",
}


class CASComplexityExplosionError(ValueError):
    """Raised when an adversarial symbolic expression exceeds safe CPU/RAM degree or AST bounds."""


@dataclass(frozen=True)
class CASResult:
    answer: str
    mode: str
    verified: bool
    trace: str


class SymbolicCASOrgan:
    """
    Pure-Python Exact Symbolic Algebra, Calculus, Linear Systems & Unit Organ.
    Guarded by CASComplexityExplosionGuard (max_ast_nodes=128, max_poly_degree=64, max_frac_bits=2048).
    """

    MAX_AST_NODES = 128
    MAX_POLY_DEGREE = 64
    MAX_FRAC_BITS = 2048
    _SAFE_FUNCS = frozenset({"sin", "cos", "tan", "exp", "ln", "log", "sqrt"})

    @classmethod
    def _guard_fraction(cls, f: Fraction) -> Fraction:
        if f.numerator.bit_length() > cls.MAX_FRAC_BITS or f.denominator.bit_length() > cls.MAX_FRAC_BITS:
            raise CASComplexityExplosionError("Rational coefficient exceeded 2048-bit complexity guard.")
        return f

    @classmethod
    def _prep_expr_str(cls, raw: str) -> str:
        s = raw.strip().replace("^", "**").replace("−", "-")
        # Insert explicit * for 3x or 2(x+1), without breaking sin(x), cos(x), ln(x), sqrt(x), exp(x)
        s = _IMPLICIT_MUL_NUM_VAR.sub(r"\1*\2", s)
        s = _IMPLICIT_MUL_PAREN.sub(r"\1*\2", s)
        return s

    @classmethod
    def _parse_safe_ast(cls, expr_str: str) -> ast.AST:
        s = cls._prep_expr_str(expr_str)
        if len(s) > 512:
            raise CASComplexityExplosionError("Expression length exceeds 512-byte CAS safety cap.")
        tree = ast.parse(s, mode="eval")
        nodes = list(ast.walk(tree))
        if len(nodes) > cls.MAX_AST_NODES:
            raise CASComplexityExplosionError(f"AST node count {len(nodes)} exceeds cap {cls.MAX_AST_NODES}.")
        for nd in nodes:
            if isinstance(nd, ast.Call):
                if not isinstance(nd.func, ast.Name) or nd.func.id not in cls._SAFE_FUNCS or len(nd.args) != 1:
                    raise ValueError("Unsupported symbolic function call.")
            elif isinstance(nd, ast.BinOp):
                if not isinstance(nd.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                    raise ValueError("Unsupported binary operator.")
                if isinstance(nd.op, ast.Pow) and isinstance(nd.right, ast.Constant) and isinstance(nd.right.value, int):
                    if abs(nd.right.value) > cls.MAX_POLY_DEGREE:
                        raise CASComplexityExplosionError(
                            f"Exponent {nd.right.value} exceeds max degree {cls.MAX_POLY_DEGREE}."
                        )
            elif isinstance(nd, ast.UnaryOp):
                if not isinstance(nd.op, (ast.UAdd, ast.USub)):
                    raise ValueError("Unsupported unary operator.")
            elif not isinstance(
                nd,
                (
                    ast.Expression,
                    ast.Name,
                    ast.Load,
                    ast.Constant,
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.Div,
                    ast.Pow,
                    ast.UAdd,
                    ast.USub,
                ),
            ):
                raise ValueError(f"Disallowed AST node: {type(nd).__name__}")
        return tree.body

    @classmethod
    def _ast_to_poly(cls, node: ast.AST, var: str) -> dict[int, Fraction] | None:
        """Convert AST into exact univariate polynomial {degree: Fraction}, or None if non-polynomial."""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            c = cls._guard_fraction(Fraction(str(node.value)))
            return {0: c} if c != 0 else {}
        if isinstance(node, ast.Name):
            if node.id == var:
                return {1: Fraction(1)}
            return None
        if isinstance(node, ast.UnaryOp):
            sub = cls._ast_to_poly(node.operand, var)
            if sub is None:
                return None
            if isinstance(node.op, ast.UAdd):
                return sub
            return {d: -c for d, c in sub.items()}
        if isinstance(node, ast.BinOp):
            left = cls._ast_to_poly(node.left, var)
            right = cls._ast_to_poly(node.right, var)
            if left is None or right is None:
                return None
            if isinstance(node.op, (ast.Add, ast.Sub)):
                sign = 1 if isinstance(node.op, ast.Add) else -1
                out: dict[int, Fraction] = defaultdict(Fraction, left)
                for d, c in right.items():
                    out[d] = cls._guard_fraction(out[d] + sign * c)
                return {d: c for d, c in out.items() if c != 0}
            if isinstance(node.op, ast.Mult):
                out_m: dict[int, Fraction] = defaultdict(Fraction)
                for d1, c1 in left.items():
                    for d2, c2 in right.items():
                        if d1 + d2 > cls.MAX_POLY_DEGREE:
                            raise CASComplexityExplosionError("Polynomial degree exceeded 64.")
                        out_m[d1 + d2] = cls._guard_fraction(out_m[d1 + d2] + c1 * c2)
                return {d: c for d, c in out_m.items() if c != 0}
            if isinstance(node.op, ast.Div):
                if len(right) == 1 and 0 in right and right[0] != 0:
                    denom = right[0]
                    return {d: cls._guard_fraction(c / denom) for d, c in left.items() if c != 0}
                return None
            if isinstance(node.op, ast.Pow):
                if not right:
                    return {0: Fraction(1)}
                if len(right) == 1 and 0 in right and right[0].denominator == 1:
                    exp_int = int(right[0].numerator)
                    if exp_int < 0 or exp_int > cls.MAX_POLY_DEGREE:
                        if exp_int > cls.MAX_POLY_DEGREE:
                            raise CASComplexityExplosionError("Polynomial exponent exceeded 64.")
                        return None
                    acc: dict[int, Fraction] = {0: Fraction(1)}
                    for _ in range(exp_int):
                        nxt: dict[int, Fraction] = defaultdict(Fraction)
                        for d1, c1 in acc.items():
                            for d2, c2 in left.items():
                                if d1 + d2 > cls.MAX_POLY_DEGREE:
                                    raise CASComplexityExplosionError("Polynomial degree exceeded 64.")
                                nxt[d1 + d2] = cls._guard_fraction(nxt[d1 + d2] + c1 * c2)
                        acc = {d: c for d, c in nxt.items() if c != 0}
                    return acc
                return None
        return None

    @staticmethod
    def _format_frac(c: Fraction) -> str:
        if c.denominator == 1:
            return str(c.numerator)
        return f"({c.numerator}/{c.denominator})"

    @classmethod
    def _format_poly(cls, poly: dict[int, Fraction], var: str) -> str:
        if not poly:
            return "0"
        parts: list[str] = []
        for idx, deg in enumerate(sorted(poly.keys(), reverse=True)):
            c = poly[deg]
            if c == 0:
                continue
            neg = c < 0
            abs_c = abs(c)
            if deg == 0:
                term = str(abs_c.numerator) if abs_c.denominator == 1 else f"{abs_c.numerator}/{abs_c.denominator}"
            else:
                vpart = var if deg == 1 else f"{var}^{deg}"
                if abs_c == 1:
                    term = vpart
                elif abs_c.denominator == 1:
                    term = f"{abs_c.numerator}*{vpart}"
                else:
                    term = f"({abs_c.numerator}/{abs_c.denominator})*{vpart}"
            if idx == 0:
                parts.append(f"-{term}" if neg else term)
            else:
                parts.append(f"- {term}" if neg else f"+ {term}")
        return " ".join(parts) if parts else "0"

    @classmethod
    def _diff_ast(cls, node: ast.AST, var: str) -> ast.AST:
        """Symbolic AST differentiation with product, quotient, power, and elementary chain rules."""
        if isinstance(node, ast.Constant):
            return ast.Constant(value=0)
        if isinstance(node, ast.Name):
            return ast.Constant(value=1 if node.id == var else 0)
        if isinstance(node, ast.UnaryOp):
            return ast.UnaryOp(op=node.op, operand=cls._diff_ast(node.operand, var))
        if isinstance(node, ast.BinOp):
            u, v = node.left, node.right
            du, dv = cls._diff_ast(u, var), cls._diff_ast(v, var)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return ast.BinOp(left=du, op=node.op, right=dv)
            if isinstance(node.op, ast.Mult):
                return ast.BinOp(
                    left=ast.BinOp(left=du, op=ast.Mult(), right=v),
                    op=ast.Add(),
                    right=ast.BinOp(left=u, op=ast.Mult(), right=dv),
                )
            if isinstance(node.op, ast.Div):
                num = ast.BinOp(
                    left=ast.BinOp(left=du, op=ast.Mult(), right=v),
                    op=ast.Sub(),
                    right=ast.BinOp(left=u, op=ast.Mult(), right=dv),
                )
                den = ast.BinOp(left=v, op=ast.Pow(), right=ast.Constant(value=2))
                return ast.BinOp(left=num, op=ast.Div(), right=den)
            if isinstance(node.op, ast.Pow):
                if isinstance(v, ast.Constant) and isinstance(v.value, (int, float)):
                    n_val = v.value
                    if n_val == 0:
                        return ast.Constant(value=0)
                    base_pow = u if n_val == 2 else ast.BinOp(left=u, op=ast.Pow(), right=ast.Constant(value=n_val - 1))
                    return ast.BinOp(
                        left=ast.BinOp(left=ast.Constant(value=n_val), op=ast.Mult(), right=base_pow),
                        op=ast.Mult(),
                        right=du,
                    )
                # General f(x)^g(x) = f^g * (g' * ln(f) + g * f' / f)
                ln_u = ast.Call(func=ast.Name(id="ln", ctx=ast.Load()), args=[u], keywords=[])
                term1 = ast.BinOp(left=dv, op=ast.Mult(), right=ln_u)
                term2 = ast.BinOp(left=v, op=ast.Mult(), right=ast.BinOp(left=du, op=ast.Div(), right=u))
                return ast.BinOp(left=node, op=ast.Mult(), right=ast.BinOp(left=term1, op=ast.Add(), right=term2))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = node.func.id
            arg = node.args[0]
            darg = cls._diff_ast(arg, var)
            if fn == "sin":
                outer: ast.AST = ast.Call(func=ast.Name(id="cos", ctx=ast.Load()), args=[arg], keywords=[])
            elif fn == "cos":
                outer = ast.UnaryOp(
                    op=ast.USub(),
                    operand=ast.Call(func=ast.Name(id="sin", ctx=ast.Load()), args=[arg], keywords=[]),
                )
            elif fn == "tan":
                cos_a = ast.Call(func=ast.Name(id="cos", ctx=ast.Load()), args=[arg], keywords=[])
                outer = ast.BinOp(
                    left=ast.Constant(value=1),
                    op=ast.Div(),
                    right=ast.BinOp(left=cos_a, op=ast.Pow(), right=ast.Constant(value=2)),
                )
            elif fn == "exp":
                outer = ast.Call(func=ast.Name(id="exp", ctx=ast.Load()), args=[arg], keywords=[])
            elif fn in ("ln", "log"):
                outer = ast.BinOp(left=ast.Constant(value=1), op=ast.Div(), right=arg)
            elif fn == "sqrt":
                outer = ast.BinOp(
                    left=ast.Constant(value=1),
                    op=ast.Div(),
                    right=ast.BinOp(
                        left=ast.Constant(value=2),
                        op=ast.Mult(),
                        right=ast.Call(func=ast.Name(id="sqrt", ctx=ast.Load()), args=[arg], keywords=[]),
                    ),
                )
            else:
                raise ValueError(f"Unsupported function {fn}")
            return ast.BinOp(left=outer, op=ast.Mult(), right=darg)
        raise ValueError("Cannot differentiate node")

    @classmethod
    def _simplify_ast(cls, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.UnaryOp):
            opd = cls._simplify_ast(node.operand)
            if isinstance(opd, ast.Constant) and isinstance(opd.value, (int, float)):
                return ast.Constant(value=+opd.value if isinstance(node.op, ast.UAdd) else -opd.value)
            if isinstance(node.op, ast.UAdd):
                return opd
            return ast.UnaryOp(op=node.op, operand=opd)
        if isinstance(node, ast.Call):
            return ast.Call(func=node.func, args=[cls._simplify_ast(a) for a in node.args], keywords=[])
        if isinstance(node, ast.BinOp):
            l = cls._simplify_ast(node.left)
            r = cls._simplify_ast(node.right)
            lv = l.value if isinstance(l, ast.Constant) and isinstance(l.value, (int, float)) else None
            rv = r.value if isinstance(r, ast.Constant) and isinstance(r.value, (int, float)) else None
            if isinstance(node.op, ast.Add):
                if lv == 0:
                    return r
                if rv == 0:
                    return l
                if lv is not None and rv is not None:
                    return ast.Constant(value=lv + rv)
            elif isinstance(node.op, ast.Sub):
                if rv == 0:
                    return l
                if lv == 0:
                    return cls._simplify_ast(ast.UnaryOp(op=ast.USub(), operand=r))
                if lv is not None and rv is not None:
                    return ast.Constant(value=lv - rv)
            elif isinstance(node.op, ast.Mult):
                if lv == 0 or rv == 0:
                    return ast.Constant(value=0)
                if lv == 1:
                    return r
                if rv == 1:
                    return l
                if lv == -1:
                    return cls._simplify_ast(ast.UnaryOp(op=ast.USub(), operand=r))
                if rv == -1:
                    return cls._simplify_ast(ast.UnaryOp(op=ast.USub(), operand=l))
                if lv is not None and rv is not None:
                    return ast.Constant(value=lv * rv)
            elif isinstance(node.op, ast.Div):
                if lv == 0 and rv != 0:
                    return ast.Constant(value=0)
                if rv == 1:
                    return l
            elif isinstance(node.op, ast.Pow):
                if rv == 0:
                    return ast.Constant(value=1)
                if rv == 1:
                    return l
            return ast.BinOp(left=l, op=node.op, right=r)
        return node

    @classmethod
    def _eval_numeric(cls, node: ast.AST, env: dict[str, float]) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return math.pi
            if node.id == "e":
                return math.e
            return float(env[node.id])
        if isinstance(node, ast.UnaryOp):
            v = cls._eval_numeric(node.operand, env)
            return +v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp):
            l = cls._eval_numeric(node.left, env)
            r = cls._eval_numeric(node.right, env)
            if isinstance(node.op, ast.Add):
                return l + r
            if isinstance(node.op, ast.Sub):
                return l - r
            if isinstance(node.op, ast.Mult):
                return l * r
            if isinstance(node.op, ast.Div):
                return l / r
            if isinstance(node.op, ast.Pow):
                return float(l**r)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            a = cls._eval_numeric(node.args[0], env)
            fn = node.func.id
            if fn == "sin":
                return math.sin(a)
            if fn == "cos":
                return math.cos(a)
            if fn == "tan":
                return math.tan(a)
            if fn == "exp":
                return math.exp(a)
            if fn in ("ln", "log"):
                return math.log(a)
            if fn == "sqrt":
                return math.sqrt(a)
        raise ValueError("Numeric eval failed")

    @classmethod
    def _solve_differentiation(cls, expr_str: str, var: str) -> CASResult:
        node = cls._parse_safe_ast(expr_str)
        poly = cls._ast_to_poly(node, var)
        if poly is not None:
            d_poly = {deg - 1: cls._guard_fraction(c * deg) for deg, c in poly.items() if deg >= 1 and c != 0}
            formatted = cls._format_poly(d_poly, var)
            d_node = cls._parse_safe_ast(formatted)
        else:
            raw_d = cls._simplify_ast(cls._diff_ast(node, var))
            formatted = ast.unparse(raw_d).replace("**", "^")
            d_node = raw_d

        # Independent finite-difference numeric verification at safe sample points
        verified = False
        h = 1e-6
        for x0 in (1.37, 0.73, 2.11):
            try:
                f_plus = cls._eval_numeric(node, {var: x0 + h})
                f_minus = cls._eval_numeric(node, {var: x0 - h})
                fd = (f_plus - f_minus) / (2.0 * h)
                analytic = cls._eval_numeric(d_node, {var: x0})
                if abs(fd - analytic) <= max(1e-4, 1e-4 * abs(analytic)):
                    verified = True
                    break
            except Exception:
                continue
        return CASResult(
            answer=f"d/d{var}({expr_str.strip()}) = {formatted}",
            mode="symbolic_cas_exact",
            verified=verified,
            trace=f"SymbolicCASOrgan exact AST differentiation wrt '{var}' (finite-diff verified={verified})",
        )

    @classmethod
    def _eval_poly_frac(cls, poly: dict[int, Fraction], x: Fraction) -> Fraction:
        total = Fraction(0)
        for deg, c in poly.items():
            total = cls._guard_fraction(total + c * (x**deg))
        return total

    @classmethod
    def _solve_integration(
        cls, expr_str: str, var: str, bounds: tuple[Fraction, Fraction] | None = None
    ) -> CASResult | None:
        node = cls._parse_safe_ast(expr_str)
        poly = cls._ast_to_poly(node, var)
        if poly is None:
            return None
        anti_poly = {deg + 1: cls._guard_fraction(c / (deg + 1)) for deg, c in poly.items() if c != 0}
        anti_str = cls._format_poly(anti_poly, var)
        if bounds is None:
            # Verify by differentiating anti_poly == poly
            re_diff = {deg - 1: c * deg for deg, c in anti_poly.items() if deg >= 1}
            verified = re_diff == {d: c for d, c in poly.items() if c != 0}
            return CASResult(
                answer=f"Integral of {expr_str.strip()} d{var} = {anti_str} + C",
                mode="symbolic_cas_exact",
                verified=verified,
                trace=f"SymbolicCASOrgan exact rational antiderivative (FTC verified={verified})",
            )
        a, b = bounds
        exact_val = cls._guard_fraction(cls._eval_poly_frac(anti_poly, b) - cls._eval_poly_frac(anti_poly, a))
        val_str = str(exact_val.numerator) if exact_val.denominator == 1 else f"{exact_val.numerator}/{exact_val.denominator}"
        # Independent Simpson's rule numerical check
        fa = cls._eval_numeric(node, {var: float(a)})
        fb = cls._eval_numeric(node, {var: float(b)})
        fm = cls._eval_numeric(node, {var: float(a + b) / 2.0})
        # Multi-panel Simpson (16 subintervals) for high degree accuracy
        n_sub = 32
        dx = float(b - a) / n_sub
        simp = fa + fb
        for i in range(1, n_sub):
            xi = float(a) + i * dx
            simp += (4.0 if i % 2 == 1 else 2.0) * cls._eval_numeric(node, {var: xi})
        simp *= dx / 3.0
        _ = fm
        verified = abs(simp - float(exact_val)) <= max(1e-4, 1e-4 * abs(float(exact_val)))
        return CASResult(
            answer=f"Definite integral of {expr_str.strip()} from {a} to {b} = {val_str}",
            mode="symbolic_cas_exact",
            verified=verified,
            trace=f"SymbolicCASOrgan exact rational definite integral = {val_str} (Simpson verified={verified})",
        )

    @classmethod
    def _extract_linear_form(cls, node: ast.AST) -> tuple[dict[str, Fraction], Fraction] | None:
        """Extract exact multivariable affine form: sum(coeff[v]*v) + const, or None if nonlinear."""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return {}, cls._guard_fraction(Fraction(str(node.value)))
        if isinstance(node, ast.Name):
            return {node.id: Fraction(1)}, Fraction(0)
        if isinstance(node, ast.UnaryOp):
            sub = cls._extract_linear_form(node.operand)
            if sub is None:
                return None
            coeffs, cst = sub
            if isinstance(node.op, ast.UAdd):
                return coeffs, cst
            return {k: -v for k, v in coeffs.items()}, -cst
        if isinstance(node, ast.BinOp):
            l = cls._extract_linear_form(node.left)
            r = cls._extract_linear_form(node.right)
            if l is None or r is None:
                return None
            lc, lk = l
            rc, rk = r
            if isinstance(node.op, (ast.Add, ast.Sub)):
                sgn = 1 if isinstance(node.op, ast.Add) else -1
                out = defaultdict(Fraction, lc)
                for k, v in rc.items():
                    out[k] = cls._guard_fraction(out[k] + sgn * v)
                return {k: v for k, v in out.items() if v != 0}, cls._guard_fraction(lk + sgn * rk)
            if isinstance(node.op, ast.Mult):
                if not lc:
                    return {k: cls._guard_fraction(lk * v) for k, v in rc.items() if lk * v != 0}, cls._guard_fraction(lk * rk)
                if not rc:
                    return {k: cls._guard_fraction(rk * v) for k, v in lc.items() if rk * v != 0}, cls._guard_fraction(lk * rk)
                return None
            if isinstance(node.op, ast.Div):
                if not rc and rk != 0:
                    return {k: cls._guard_fraction(v / rk) for k, v in lc.items() if v != 0}, cls._guard_fraction(lk / rk)
                return None
        return None

    @classmethod
    def _solve_equations(cls, body: str) -> CASResult | None:
        if "=" not in body:
            return None
        eq_strs = [e.strip() for e in re.split(r"(?:;|,|\band\b)", body) if "=" in e]
        if not eq_strs or len(eq_strs) > 8:
            return None

        # Single-equation univariate polynomial (linear or quadratic)
        if len(eq_strs) == 1:
            lhs_s, rhs_s = eq_strs[0].split("=", 1)
            lhs_ast = cls._parse_safe_ast(lhs_s)
            rhs_ast = cls._parse_safe_ast(rhs_s)
            diff_ast = ast.BinOp(left=lhs_ast, op=ast.Sub(), right=rhs_ast)
            names = sorted({nd.id for nd in ast.walk(diff_ast) if isinstance(nd, ast.Name) and nd.id not in cls._SAFE_FUNCS})
            if len(names) == 1:
                var = names[0]
                poly = cls._ast_to_poly(diff_ast, var)
                if poly is not None:
                    max_d = max(poly.keys(), default=0)
                    if max_d == 1:
                        a = poly.get(1, Fraction(0))
                        b = poly.get(0, Fraction(0))
                        root = cls._guard_fraction(-b / a)
                        lhs_v = cls._eval_poly_frac(cls._ast_to_poly(lhs_ast, var) or {}, root)
                        rhs_v = cls._eval_poly_frac(cls._ast_to_poly(rhs_ast, var) or {}, root)
                        r_str = str(root.numerator) if root.denominator == 1 else f"{root.numerator}/{root.denominator}"
                        return CASResult(
                            answer=f"{var} = {r_str}",
                            mode="symbolic_cas_exact",
                            verified=(lhs_v == rhs_v),
                            trace=f"Exact rational linear root {var} = {r_str} (substitution verified={lhs_v == rhs_v})",
                        )
                    if max_d == 2:
                        a = poly.get(2, Fraction(0))
                        b = poly.get(1, Fraction(0))
                        c = poly.get(0, Fraction(0))
                        disc = cls._guard_fraction(b * b - 4 * a * c)
                        if disc >= 0:
                            num_sq = math.isqrt(disc.numerator)
                            den_sq = math.isqrt(disc.denominator)
                            if num_sq * num_sq == disc.numerator and den_sq * den_sq == disc.denominator:
                                sq = Fraction(num_sq, den_sq)
                                r1 = (-b - sq) / (2 * a)
                                r2 = (-b + sq) / (2 * a)
                                roots = sorted({r1, r2})
                                ok = all(cls._eval_poly_frac(poly, r) == 0 for r in roots)
                                fmt_r = ", ".join(
                                    str(r.numerator) if r.denominator == 1 else f"{r.numerator}/{r.denominator}"
                                    for r in roots
                                )
                                return CASResult(
                                    answer=f"{var} = {fmt_r}",
                                    mode="symbolic_cas_exact",
                                    verified=ok,
                                    trace=f"Exact rational quadratic roots {var} in {{{fmt_r}}} (verified={ok})",
                                )
                            sq_f = math.sqrt(float(disc))
                            x1 = (float(-b) - sq_f) / float(2 * a)
                            x2 = (float(-b) + sq_f) / float(2 * a)
                            ok_f = abs(cls._eval_numeric(diff_ast, {var: x1})) < 1e-6
                            return CASResult(
                                answer=f"{var} = {x1:.6g}, {x2:.6g}",
                                mode="symbolic_cas_exact",
                                verified=ok_f,
                                trace="Quadratic formula real roots verified by substitution",
                            )
                        return CASResult(
                            answer=f"No real roots (discriminant = {disc} < 0).",
                            mode="symbolic_cas_exact",
                            verified=True,
                            trace=f"Exact negative discriminant {disc}",
                        )

        # Multi-variable exact rational linear system via Gauss-Jordan elimination
        parsed_eqs: list[tuple[dict[str, Fraction], Fraction]] = []
        all_vars_set: set[str] = set()
        raw_ast_pairs: list[tuple[ast.AST, ast.AST]] = []
        for eq in eq_strs:
            lhs_s, rhs_s = eq.split("=", 1)
            l_ast, r_ast = cls._parse_safe_ast(lhs_s), cls._parse_safe_ast(rhs_s)
            raw_ast_pairs.append((l_ast, r_ast))
            lin = cls._extract_linear_form(ast.BinOp(left=l_ast, op=ast.Sub(), right=r_ast))
            if lin is None:
                return None
            coeffs, cst = lin
            # sum(coeffs[v]*v) + cst = 0  ==>  sum(coeffs[v]*v) = -cst
            parsed_eqs.append((coeffs, -cst))
            all_vars_set.update(coeffs.keys())

        vars_sorted = sorted(all_vars_set)
        n = len(vars_sorted)
        if n == 0 or len(parsed_eqs) < n or n > 8:
            return None

        mat: list[list[Fraction]] = []
        for coeffs, rhs_val in parsed_eqs:
            row = [coeffs.get(v, Fraction(0)) for v in vars_sorted] + [rhs_val]
            mat.append(row)

        m_rows = len(mat)
        piv_row = 0
        for col in range(n):
            pivot = None
            for r in range(piv_row, m_rows):
                if mat[r][col] != 0:
                    pivot = r
                    break
            if pivot is None:
                return CASResult(
                    answer="Linear system is singular (no unique solution).",
                    mode="symbolic_cas_exact",
                    verified=True,
                    trace="Exact rational Gauss-Jordan detected rank deficiency",
                )
            mat[piv_row], mat[pivot] = mat[pivot], mat[piv_row]
            piv_val = mat[piv_row][col]
            mat[piv_row] = [cls._guard_fraction(x / piv_val) for x in mat[piv_row]]
            for r in range(m_rows):
                if r != piv_row and mat[r][col] != 0:
                    factor = mat[r][col]
                    mat[r] = [
                        cls._guard_fraction(mat[r][c_i] - factor * mat[piv_row][c_i])
                        for c_i in range(n + 1)
                    ]
            piv_row += 1

        for r in range(n, m_rows):
            if mat[r][n] != 0:
                return CASResult(
                    answer="Inconsistent linear system (no solution exists).",
                    mode="symbolic_cas_exact",
                    verified=True,
                    trace="Exact rational Gauss-Jordan detected inconsistent row",
                )

        sol = {vars_sorted[i]: mat[i][n] for i in range(n)}
        verified = all(
            sum(coeffs.get(v, Fraction(0)) * sol[v] for v in vars_sorted) == rhs_val
            for coeffs, rhs_val in parsed_eqs
        )
        _ = raw_ast_pairs
        sol_str = ", ".join(
            f"{v} = {sol[v].numerator if sol[v].denominator == 1 else f'{sol[v].numerator}/{sol[v].denominator}'}"
            for v in vars_sorted
        )
        return CASResult(
            answer=sol_str,
            mode="symbolic_cas_exact",
            verified=verified,
            trace=f"Exact rational Gauss-Jordan linear system solution ({sol_str}, verified={verified})",
        )

    @classmethod
    def _solve_unit_conversion(cls, val_s: str, u_from_raw: str, u_to_raw: str) -> CASResult | None:
        u_from = u_from_raw.strip().lower()
        u_to = u_to_raw.strip().lower()
        val = Fraction(val_s)

        if u_from in _TEMP_ALIASES or u_to in _TEMP_ALIASES:
            if u_from not in _TEMP_ALIASES or u_to not in _TEMP_ALIASES:
                return CASResult(
                    answer=f"Dimensional mismatch: cannot convert '{u_from_raw}' to '{u_to_raw}'.",
                    mode="cas_dimensional_mismatch",
                    verified=True,
                    trace="DimensionalMismatchGuard blocked incompatible temperature conversion",
                )
            tf, tt = _TEMP_ALIASES[u_from], _TEMP_ALIASES[u_to]
            # Convert to Celsius exact Fraction first
            if tf == "c":
                c_val = val
            elif tf == "f":
                c_val = (val - 32) * Fraction(5, 9)
            else:
                c_val = val - Fraction(27315, 100)

            if tt == "c":
                out_v = c_val
            elif tt == "f":
                out_v = c_val * Fraction(9, 5) + 32
            else:
                out_v = c_val + Fraction(27315, 100)
            f_num = float(out_v)
            disp = str(int(f_num)) if f_num.is_integer() else f"{f_num:.6g}"
            return CASResult(
                answer=f"{val_s} {u_from_raw} = {disp} {u_to_raw}",
                mode="symbolic_cas_exact",
                verified=True,
                trace=f"Exact affine temperature conversion ({tf} -> {tt})",
            )

        if u_from not in _UNIT_TABLE and u_to not in _UNIT_TABLE:
            return None
        if u_from not in _UNIT_TABLE or u_to not in _UNIT_TABLE:
            return None

        dim_f, fac_f = _UNIT_TABLE[u_from]
        dim_t, fac_t = _UNIT_TABLE[u_to]
        if dim_f != dim_t:
            return CASResult(
                answer=f"Dimensional mismatch: '{u_from_raw}' ({dim_f}) cannot be converted to '{u_to_raw}' ({dim_t}).",
                mode="cas_dimensional_mismatch",
                verified=True,
                trace=f"DimensionalMismatchGuard rejected {dim_f} -> {dim_t}",
            )
        exact_out = cls._guard_fraction(val * fac_f / fac_t)
        f_out = float(exact_out)
        disp = str(exact_out.numerator) if exact_out.denominator == 1 else f"{f_out:.6g}"
        # Reverse conversion round-trip check
        back = exact_out * fac_t / fac_f
        return CASResult(
            answer=f"{val_s} {u_from_raw} = {disp} {u_to_raw}",
            mode="symbolic_cas_exact",
            verified=(back == val),
            trace=f"Exact rational dimensional conversion ({dim_f}, round-trip verified={back == val})",
        )

    @classmethod
    def try_solve(cls, query: str) -> CASResult | None:
        """Fast gate: solve symbolic calculus, algebra/systems, or unit conversions in < 2 ms; None otherwise."""
        q = query.strip()
        if not q or len(q) > 400:
            return None
        # P6-CrossLingualCAS Bridge: reorder non-ASCII UniversalConceptHub suffix ('求导 x^3 + 5*x differentiate')
        if not q.isascii():
            for cas_kw in ("differentiate", "integrate", "solve"):
                sfx = f" {cas_kw}"
                if q.lower().endswith(sfx):
                    ascii_math = "".join(ch for ch in q[: -len(sfx)] if ord(ch) < 128).strip()
                    if ascii_math:
                        q = f"{cas_kw} {ascii_math}"
                    break
        ql = q.lower()
        if not any(
            cue in ql
            for cue in (
                "differentiate",
                "derivative",
                "d/d",
                "integrate",
                "integral",
                "antiderivative",
                "solve",
                "roots of",
                "convert",
                " to ",
                " in ",
            )
        ):
            return None

        try:
            m_diff = _DIFF_RE.match(q)
            if m_diff:
                var = (m_diff.group(1) or m_diff.group(3) or "x").lower()
                return cls._solve_differentiation(m_diff.group(2), var)

            m_idef = _INTEG_DEF_RE.match(q)
            if m_idef:
                var = (m_idef.group(4) or "x").lower()
                a = Fraction(m_idef.group(2))
                b = Fraction(m_idef.group(3))
                return cls._solve_integration(m_idef.group(1), var, bounds=(a, b))

            m_iind = _INTEG_INDEF_RE.match(q)
            if m_iind:
                var = (m_iind.group(2) or "x").lower()
                return cls._solve_integration(m_iind.group(1), var, bounds=None)

            m_sol = _SOLVE_RE.match(q)
            if m_sol and "=" in m_sol.group(1):
                return cls._solve_equations(m_sol.group(1))

            m_conv = _CONVERT_RE.match(q)
            if m_conv:
                return cls._solve_unit_conversion(m_conv.group(1), m_conv.group(2), m_conv.group(3))
        except CASComplexityExplosionError as exc:
            return CASResult(
                answer=f"Symbolic CAS complexity guard triggered: {exc}",
                mode="cas_complexity_guarded",
                verified=True,
                trace=f"CASComplexityExplosionGuard aborted unsafe symbolic expansion: {exc}",
            )
        except Exception:
            return None
        return None


# ============================================================================
# 2. PHASE-5 #2 + PHASE-6 B6-1: 1.58-Bit Ternary Bicameral Fluid Composer
# ============================================================================

_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_BIG_O_RE = re.compile(r"\bO\([^)]+\)")
_NEG_WORDS = frozenset({"not", "never", "no", "cannot", "impossible", "invalid", "unbounded"})

_STYLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "apology",
        re.compile(
            r"\b(?:draft|write|compose|format\s+as)\s+(?:a\s+|an\s+)?(?:polite\s+|empathetic\s+|client\s+|short\s+)*"
            r"apolog(?:y|etic)\s+(?:email|note|message|update)\b",
            re.I,
        ),
    ),
    (
        "email",
        re.compile(
            r"\b(?:draft|write|compose|format\s+as)\s+(?:a\s+|an\s+)?(?:polite\s+|client\s+|team\s+|short\s+)*"
            r"(?:email|memo)\b",
            re.I,
        ),
    ),
    (
        "executive_brief",
        re.compile(r"\b(?:executive\s+(?:brief(?:ing)?|summary)|format\s+as\s+(?:a\s+)?briefing|tl;?dr\s+brief)\b", re.I),
    ),
    (
        "beginner_guide",
        re.compile(r"\b(?:eli5|explain\s+like\s+i(?:'m|\s+am)\s+5|in\s+beginner\s+terms)\b", re.I),
    ),
)

_CREATIVE_PROMPT_RE = re.compile(
    r"\b(?:write\s+(?:a|an)\s+(?:short\s+)?(?:story|poem|haiku|fable|monologue|metaphor)|"
    r"draft\s+(?:a|an)\s+(?:empathetic\s+|polite\s+|custom\s+)?(?:apology|welcome\s+letter|farewell\s+note)|"
    r"imagine\s+(?:a|an)\s+world)\b",
    re.I,
)


class PackedTernaryProjection:
    """
    True 1.58-bit {-1, 0, +1} Linear Projection packed at 4 weights per byte (2 bits/weight).
    Forward pass executes pure integer additions/subtractions (0 floating-point multiplies, 0 GPU).
    Encoding per 2-bit slot: 00 -> 0, 01 -> +1, 10 -> -1.
    """

    def __init__(self, in_dim: int = 64, out_dim: int = 8, seed: bytes = b"mirror_ternary_v1") -> None:
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.bytes_per_row = (self.in_dim + 3) // 4
        self.packed = bytearray(self.out_dim * self.bytes_per_row)
        self._init_deterministic_ternary(seed)

    def _init_deterministic_ternary(self, seed: bytes) -> None:
        for r in range(self.out_dim):
            digest = hashlib.blake2b(seed + r.to_bytes(2, "little"), digest_size=self.bytes_per_row).digest()
            base = r * self.bytes_per_row
            for b_i, byte_val in enumerate(digest):
                out_b = 0
                for lane in range(4):
                    code = (byte_val >> (2 * lane)) & 0b11
                    # Map 00, 11 -> 0 (~50% zero sparsity, matching TernaryBitLinear!), 01 -> +1, 10 -> -1
                    enc = 0 if code in (0, 3) else code
                    out_b |= enc << (2 * lane)
                self.packed[base + b_i] = out_b

    @classmethod
    def from_ternary_matrix(cls, rows: list[list[int]]) -> PackedTernaryProjection:
        """Pack any {-1, 0, +1} matrix (including frozen TernaryBitLinear weights) into 2-bit bytes."""
        out_d = len(rows)
        in_d = len(rows[0]) if out_d else 0
        inst = cls(in_dim=max(1, in_d), out_dim=max(1, out_d))
        inst.packed = bytearray(out_d * inst.bytes_per_row)
        for r, row in enumerate(rows):
            base = r * inst.bytes_per_row
            for c, val in enumerate(row):
                enc = 1 if val > 0 else (2 if val < 0 else 0)
                inst.packed[base + (c // 4)] |= enc << (2 * (c % 4))
        return inst

    def project_int(self, x: list[int]) -> list[int]:
        """Pure integer forward pass: y[r] = sum_{w=+1} x[c] - sum_{w=-1} x[c] (0 multiplications!)."""
        out: list[int] = [0] * self.out_dim
        limit = min(len(x), self.in_dim)
        for r in range(self.out_dim):
            acc = 0
            base = r * self.bytes_per_row
            for c in range(limit):
                code = (self.packed[base + (c // 4)] >> (2 * (c % 4))) & 0b11
                if code == 1:
                    acc += x[c]
                elif code == 2:
                    acc -= x[c]
            out[r] = acc
        return out

    def storage_bytes(self) -> int:
        return len(self.packed)


class FactFidelityGuard:
    """
    Phase-6 Bottleneck B6-1 Defense:
    Guarantees Right-Brain rhetorical composition NEVER drops verified Left-Brain numbers,
    Big-O complexities, or negations, and NEVER invents hallucinated numeric claims!
    """

    @staticmethod
    def verify_preservation(source_fact: str, composed_text: str) -> bool:
        src_nums = set(_NUM_RE.findall(source_fact))
        cmp_nums = set(_NUM_RE.findall(composed_text))
        # Every source number must survive, and no ungrounded numbers may appear
        if not src_nums.issubset(cmp_nums):
            return False
        # Allow small structural bullet indices {"1", "2", "3"} only if bulleted, otherwise block new numbers
        extra_nums = cmp_nums - src_nums - {"1", "2", "3"}
        if extra_nums:
            return False

        src_bigo = {s.replace(" ", "") for s in _BIG_O_RE.findall(source_fact)}
        cmp_bigo = {s.replace(" ", "") for s in _BIG_O_RE.findall(composed_text)}
        if not src_bigo.issubset(cmp_bigo):
            return False

        src_toks = {w.lower() for w in re.findall(r"[A-Za-z]+", source_fact)}
        cmp_toks = {w.lower() for w in re.findall(r"[A-Za-z]+", composed_text)}
        src_negs = src_toks & _NEG_WORDS
        if src_negs and not (cmp_toks & _NEG_WORDS):
            return False
        return True


class TernaryFluidComposer:
    """
    Right-Brain 1.58-Bit Ternary Fluid Composer:
      - Mode A (Grounded Rhetorical Realization): Re-frames Left-Brain verified facts into requested
        email / apology / executive brief / beginner guide formats under FactFidelityGuard.
      - Mode B (Creative Synthesis): Generates structured imaginative prose ONLY for explicit creative
        prompts when Left-Brain abstains, tagged 'creative_synthesis' (never self-poisoning factual rules).
    """

    def __init__(self) -> None:
        self.ternary_head = PackedTernaryProjection(in_dim=64, out_dim=6)
        self.collocations: dict[str, list[str]] = defaultdict(list)

    def observe_corpus_text(self, text: str) -> None:
        words = [w.lower() for w in re.findall(r"[A-Za-z]{4,18}", text)]
        for i in range(len(words) - 1):
            a, b = words[i], words[i + 1]
            if a != b and len(self.collocations) < 4096:
                bucket = self.collocations[a]
                if b not in bucket and len(bucket) < 6:
                    bucket.append(b)

    @staticmethod
    def detect_requested_style(prompt: str) -> str | None:
        for style, pat in _STYLE_PATTERNS:
            if pat.search(prompt):
                return style
        return None

    @classmethod
    def strip_style_framing(cls, prompt: str) -> str:
        """Strip rhetorical style wrapper ('Draft a polite email explaining ...') so Left-Brain locks cleanly."""
        if cls.detect_requested_style(prompt) is None:
            return prompt
        cleaned = re.sub(
            r"^\s*(?:please\s+)?(?:draft|write|compose|format\s+as|give\s+me)\s+"
            r"(?:a\s+|an\s+)?(?:polite\s+|empathetic\s+|client\s+|team\s+|short\s+|concise\s+)*"
            r"(?:apolog(?:y|etic)\s+)?(?:email|memo|note|message|update|executive\s+(?:brief(?:ing)?|summary)|briefing)"
            r"(?:\s+(?:explaining|about|on|regarding|covering|for))?\s*",
            "",
            prompt,
            flags=re.I,
        )
        cleaned = re.sub(r"\b(?:in\s+beginner\s+terms|explain\s+like\s+i(?:'m|\s+am)\s+5|eli5)\b[:,\s]*", "", cleaned, flags=re.I)
        return cleaned.strip() or prompt

    @staticmethod
    def is_explicit_creative_request(prompt: str) -> bool:
        return bool(_CREATIVE_PROMPT_RE.search(prompt))

    def _signature_to_int_features(self, sig2048: int) -> list[int]:
        # Pool 2,048 bits into 64 integer popcount buckets (0..32 each) -> centered (-16..+16)
        feats: list[int] = []
        for i in range(64):
            chunk = (sig2048 >> (i * 32)) & 0xFFFFFFFF
            feats.append(chunk.bit_count() - 8)
        return feats

    def realize_grounded_style(self, prompt: str, verified_fact: str, sig2048: int = 0) -> tuple[str, bool]:
        """Style verified Left-Brain fact into requested register; returns (text, was_styled)."""
        style = self.detect_requested_style(prompt)
        if style is None or "```" in verified_fact or "def " in verified_fact:
            return verified_fact, False

        scores = self.ternary_head.project_int(self._signature_to_int_features(sig2048))
        tone_idx = abs(scores[0]) % 2
        clean_fact = verified_fact.strip()

        if style == "apology":
            opening = (
                "Thank you for your patience — we sincerely apologize for the inconvenience."
                if tone_idx == 0
                else "Please accept our sincere apology regarding this update."
            )
            composed = (
                f"Subject: Service Update & Technical Clarification\n\n"
                f"{opening}\n\n"
                f"Verified Details:\n{clean_fact}\n\n"
                f"We appreciate your trust and remain committed to transparent resolution."
            )
        elif style == "email":
            greeting = "Hello," if tone_idx == 0 else "Hi team,"
            composed = (
                f"Subject: Technical Summary\n\n"
                f"{greeting}\n\n"
                f"Here is the verified breakdown for your request:\n{clean_fact}\n\n"
                f"Best regards,\nMirror AI"
            )
        elif style == "executive_brief":
            composed = (
                f"Executive Briefing (Verified Grounding):\n"
                f"- Core Finding: {clean_fact}\n"
                f"- Epistemic Status: Grounded directly in Mirror Crystal Vault (zero unverified extrapolation)."
            )
        else:  # beginner_guide
            composed = (
                f"In clear, everyday terms:\n{clean_fact}\n"
                f"(Key takeaway: every number and rule above comes directly from verified memory.)"
            )

        if not FactFidelityGuard.verify_preservation(clean_fact, composed):
            return verified_fact, False
        return composed, True

    def synthesize_creative(self, prompt: str, sig2048: int = 0) -> str | None:
        """Synthesize open-ended story/poem/empathetic draft ONLY when explicitly requested."""
        if not self.is_explicit_creative_request(prompt):
            return None

        stop = {
            "write", "draft", "short", "story", "poem", "haiku", "fable", "about", "with",
            "that", "this", "from", "into", "their", "empathetic", "polite", "custom",
            "apology", "letter", "note", "imagine", "world", "please", "create",
        }
        raw_words = [w for w in re.findall(r"[A-Za-z]{3,18}", prompt) if w.lower() not in stop]
        themes = list(dict.fromkeys(w.lower() for w in raw_words))[:5] or ["horizon", "silence", "light"]
        scores = self.ternary_head.project_int(self._signature_to_int_features(sig2048))
        cadence = abs(scores[1]) % 2

        ql = prompt.lower()
        if "haiku" in ql or "poem" in ql:
            t0 = themes[0]
            t1 = themes[1] if len(themes) > 1 else "starlight"
            t2 = themes[2] if len(themes) > 2 else "memory"
            return (
                f"Across quiet {t0},\n"
                f"Bright {t1} wakes the sleeping dark,\n"
                f"Still {t2} endures."
            )
        if "apology" in ql or "letter" in ql or "note" in ql:
            topic_phrase = " ".join(themes[:3])
            return (
                f"Dear Valued Partner,\n\n"
                f"We sincerely apologize for the disruption involving {topic_phrase}. "
                f"We understand how much reliability matters to your workflow, and we take full responsibility "
                f"for addressing the root cause promptly.\n\n"
                f"Our team has put corrective safeguards in place and is monitoring every step closely. "
                f"Thank you for your patience and partnership."
            )

        # Narrative short story grounded in prompt's extracted concepts + corpus collocations
        anchor = themes[0]
        setting = themes[1] if len(themes) > 1 else "distant frontier"
        detail = themes[2] if len(themes) > 2 else "quiet beacon"
        colloc = self.collocations.get(anchor, ["enduring"])[0]
        arc_line = (
            f"Night after night, the {anchor} stood watch across {setting}, guided by one {colloc} purpose."
            if cadence == 0
            else f"At the edge of {setting}, the {anchor} hummed softly beside the {detail}, carrying a {colloc} signal."
        )
        return (
            f"{arc_line} Even when storms swept across {setting}, the {detail} kept shining as a reminder "
            f"that careful craft outlasts uncertainty.\n\n"
            f"When travelers finally arrived at dawn, they realized the {anchor} had not merely marked "
            f"safe passage through {setting} — it had turned isolation into connection."
        )


# ============================================================================
# 3. PHASE-5 #3 + PHASE-6 B6-4: ActiveCuriosityPlanner + CuriosityLoopGuard
# ============================================================================


@dataclass
class _PendingClarification:
    original_query: str
    options: tuple[str, ...]


class ActiveCuriosityPlanner:
    """
    Diagnostic Information-Gain Clarifier:
      - Inspects competing LSH candidate rules when a short/vague query straddles >= 2 genuinely
        distinct technical branches (low mutual token Jaccard < 0.35).
      - Asks at most 1 targeted clarifying question per topic (CuriosityLoopGuard prevents nagging loops!).
    """

    _GENERIC_DISCRIM = frozenset(
        {
            "explain", "compute", "algorithm", "method", "system", "function", "code", "example",
            "part", "step", "using", "value", "number", "result", "problem", "solution", "time",
            "complexity", "works", "work", "fast", "data", "model", "process", "approach",
            "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "act", "scene", "chapter", "section",
        }
    )

    def __init__(self) -> None:
        self._pending: dict[str, _PendingClarification] = {}

    def merge_followup_if_pending(self, session_id: str, new_query: str) -> tuple[str, bool]:
        """If previous turn asked a clarification, fuse original query + user's choice and block re-asking!"""
        pend = self._pending.pop(session_id, None)
        if pend is None:
            return new_query, False
        nw_toks = [w.lower() for w in re.findall(r"[A-Za-z0-9_]+", new_query)]
        # Only fuse if user gave a short/direct clarifying reply (<= 8 words) or named one of the options
        if len(nw_toks) <= 8 or any(opt in nw_toks for opt in pend.options):
            return f"{pend.original_query} {new_query}", True
        return new_query, False

    def check_diagnostic_ambiguity(
        self,
        session_id: str,
        query: str,
        candidates: list[tuple[float, Any]],
        query_salient_tokens: set[str],
        already_clarified_this_turn: bool = False,
    ) -> tuple[str, list[str]] | None:
        """Return (clarification_question, trace_lines) if genuinely ambiguous; None otherwise."""
        if already_clarified_this_turn or len(candidates) < 2:
            return None

        sc0, r0 = candidates[0]
        sc1, r1 = candidates[1]
        # Only clarify when query is short/general (<= 4 salient words), non-slot, non-axiom, and top 2 rules are solidly grounded and nearly tied!
        if (
            sc0 >= 0.85
            or sc0 < 0.28
            or sc1 < 0.22
            or (sc0 - sc1) > 0.08
            or getattr(r0, "trust_tier", 2) == 1
            or getattr(r0, "domain", "") == "identity"
        ):
            return None
        if len(query_salient_tokens) > 4 or r0.num_slots > 0 or r1.num_slots > 0:
            return None

        t0 = {t for t in r0.token_set if len(t) >= 3 and not t.startswith(("sense_", "__", "vq"))}
        t1 = {t for t in r1.token_set if len(t) >= 3 and not t.startswith(("sense_", "__", "vq"))}
        query_substantive = {t for t in query_salient_tokens if t not in self._GENERIC_DISCRIM and len(t) >= 3 and not t.isdigit()}
        if not query_substantive or not (t0 & query_substantive) or not (t1 & query_substantive):
            return None
        cov0 = len(t0 & query_substantive) / max(1, len(query_substantive))
        cov1 = len(t1 & query_substantive) / max(1, len(query_substantive))
        if cov0 < 0.60 or cov1 < 0.60:
            return None
        rem0 = t0 - query_salient_tokens - self._GENERIC_DISCRIM
        rem1 = t1 - query_salient_tokens - self._GENERIC_DISCRIM
        if not rem0 or not rem1:
            return None
        # GenuineAmbiguityFilter: paraphrases of the same concept share remaining keywords; true forks are disjoint!
        rem_union = len(rem0 | rem1)
        rem_jaccard = len(rem0 & rem1) / max(1, rem_union)
        if rem_jaccard > 0.0 or r0.abstract_response[:80] == r1.abstract_response[:80]:
            return None

        diff0 = sorted(rem0)
        diff1 = sorted(rem1)

        options: list[str] = [diff0[0], diff1[0]]
        if len(candidates) >= 3 and (sc0 - candidates[2][0]) <= 0.10:
            t2 = {t for t in candidates[2][1].token_set if len(t) >= 3 and not t.startswith(("sense_", "__", "vq"))}
            diff2 = sorted(t2 - query_salient_tokens - rem0 - rem1 - self._GENERIC_DISCRIM)
            if diff2 and diff2[0] not in options:
                options.append(diff2[0])

        self._pending[session_id] = _PendingClarification(original_query=query, options=tuple(options))
        opts_fmt = ", ".join(f"({i + 1}) {opt}" for i, opt in enumerate(options))
        q_msg = (
            f"Your request matches {len(options)} distinct verified knowledge branches with nearly equal support. "
            f"Which specific context do you mean: {opts_fmt}?"
        )
        trace = [
            f"ActiveCuriosityPlanner detected multi-branch ambiguity (scores={sc0:.3f} vs {sc1:.3f}, "
            f"rem_jaccard={rem_jaccard:.2f}) -> options={options}"
        ]
        return q_msg, trace


# ============================================================================
# 4. PHASE-5 #4 + PHASE-6 B6-6: DocumentAutoCrystallizer + Safety/Boilerplate Guard
# ============================================================================

_BOILERPLATE_HEADINGS = frozenset(
    {
        "license", "copyright", "table of contents", "contents", "contributing",
        "changelog", "acknowledgments", "acknowledgements", "authors",
    }
)
_DEF_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*([A-Za-z0-9_\- ]{3,48})\*\*|([A-Z][A-Za-z0-9_\- ]{2,44}))"
    r"\s*(?::|—|--|\bis\s+defined\s+as\b)\s+(.{25,500})$"
)
_CAUSAL_CUE_RE = re.compile(r"\b(?:causes|triggers|implies|leads\s+to)\b|^if\s+.+,\s+.+", re.I)


class DocumentAutoCrystallizer:
    """
    Zero-.jsonl Raw Knowledge Extractor:
      - Ingests Markdown (.md), plain-text books/notes (.txt), and Python source modules (.py).
      - Filters license/TOC boilerplate (DocBoilerplateAndSafetyGuard).
      - Runs ZeroTrustSandboxJudge on extracted Python functions + inline assertions to promote
        verified functions directly to trust_tier=1!
    """

    @classmethod
    def ingest_python_source(
        cls, target_brain: Any, source_code: str, module_name: str = "module", domain: str = "code_crystallized"
    ) -> dict[str, int]:
        stats = {"functions_found": 0, "verified_tier1": 0, "documented_tier2": 0, "rejected_unsafe": 0}
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return stats

        # Collect top-level or __main__ assert statements referencing functions
        func_asserts: dict[str, list[str]] = defaultdict(list)
        for node in tree.body:
            if isinstance(node, ast.Assert):
                src_a = ast.unparse(node)
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                        func_asserts[sub.func.id].append(src_a)
            elif isinstance(node, ast.If):
                for stmt in node.body:
                    if isinstance(stmt, ast.Assert):
                        src_a = ast.unparse(stmt)
                        for sub in ast.walk(stmt):
                            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                                func_asserts[sub.func.id].append(src_a)

        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                continue
            stats["functions_found"] += 1
            fn_name = node.name
            doc = ast.get_docstring(node) or ""
            clean_doc = doc.splitlines()[0].strip() if doc.strip() else f"Implement {fn_name} from {module_name}"
            fn_src = ast.unparse(node)
            assert_lines = func_asserts.get(fn_name, [])
            candidate_bundle = fn_src + ("\n" + "\n".join(assert_lines) if assert_lines else "")

            ok, judge_msg = ZeroTrustSandboxJudge.verify_code_and_test(candidate_bundle)
            if not ok:
                stats["rejected_unsafe"] += 1
                continue

            prompt = f"Write Python function {fn_name}: {clean_doc}"
            ans_block = f"```python\n{fn_src}\n```"
            if assert_lines and ok and judge_msg != "non_code":
                joined_asserts = "\n".join(assert_lines)
                target_brain.ingest_pair(
                    prompt, ans_block, tool=joined_asserts, test_code=joined_asserts, domain=domain, trust_tier=1
                )
                stats["verified_tier1"] += 1
            else:
                target_brain.ingest_pair(prompt, ans_block, domain=domain, trust_tier=2)
                stats["documented_tier2"] += 1
        return stats

    @classmethod
    def ingest_document_text(
        cls, target_brain: Any, text: str, title: str = "Document", domain: str = "doc_crystallized"
    ) -> dict[str, int]:
        stats = {
            "sections_crystallized": 0,
            "definitions_crystallized": 0,
            "causal_laws_observed": 0,
            "code_blocks_verified": 0,
            "boilerplate_skipped": 0,
        }
        if not text or not text.strip():
            return stats

        # 1. Extract fenced ```python code blocks and verify via AST + ZeroTrustSandboxJudge
        for m_py in re.finditer(r"```python\s*\n(.*?)```", text, flags=re.S | re.I):
            py_res = cls.ingest_python_source(target_brain, m_py.group(1), module_name=title, domain=domain)
            stats["code_blocks_verified"] += py_res["verified_tier1"] + py_res["documented_tier2"]

        # 2. Split Markdown/text into headed sections
        current_heading = title.strip()
        buf: list[str] = []
        sections: list[tuple[str, str]] = []
        for raw_ln in text.splitlines():
            m_h = re.match(r"^\s*(#{1,4})\s+(.+?)\s*$", raw_ln)
            if m_h:
                if buf:
                    sections.append((current_heading, "\n".join(buf).strip()))
                    buf = []
                current_heading = m_h.group(2).strip()
            else:
                buf.append(raw_ln)
                # Also check inline definitions & causal statements
                m_def = _DEF_LINE_RE.match(raw_ln)
                if m_def:
                    term = (m_def.group(1) or m_def.group(2) or "").strip()
                    defn = m_def.group(3).strip()
                    if len(term) >= 3 and term.lower() not in _BOILERPLATE_HEADINGS:
                        if target_brain.ingest_pair(f"What is {term}?", f"{term}: {defn}", domain=domain):
                            stats["definitions_crystallized"] += 1
                if _CAUSAL_CUE_RE.search(raw_ln) and 15 <= len(raw_ln.strip()) <= 240:
                    target_brain.observe_unlabeled_text(raw_ln.strip())
                    stats["causal_laws_observed"] += 1

        if buf:
            sections.append((current_heading, "\n".join(buf).strip()))

        for heading, body in sections:
            h_clean = re.sub(r"[`*_#]+", "", heading).strip()
            if not h_clean or h_clean.lower() in _BOILERPLATE_HEADINGS:
                stats["boilerplate_skipped"] += 1
                continue
            # Strip already-extracted fenced code blocks so unsafe/unrelated code snippets never taint prose rules
            prose_clean = re.sub(r"```.*?```", "", body, flags=re.S).strip()
            if len(prose_clean) < 35 or len(prose_clean) > 3500:
                continue
            if target_brain.ingest_pair(f"Explain {h_clean}", prose_clean, domain=domain):
                stats["sections_crystallized"] += 1
        return stats

    @classmethod
    def ingest_file(cls, target_brain: Any, path: Path | str, domain: str | None = None) -> dict[str, int]:
        p = Path(path)
        content = p.read_text(encoding="utf-8", errors="replace")
        dom = domain or f"file:{p.stem}"
        if p.suffix.lower() == ".py":
            return cls.ingest_python_source(target_brain, content, module_name=p.stem, domain=dom)
        return cls.ingest_document_text(target_brain, content, title=p.stem, domain=dom)

    @classmethod
    def ingest_directory(
        cls,
        target_brain: Any,
        directory_path: Path | str,
        domain: str | None = None,
        recursive: bool = True,
        extensions: tuple[str, ...] = (".md", ".txt", ".rst", ".py", ".json"),
    ) -> dict[str, int]:
        d = Path(directory_path)
        stats = {
            "files_processed": 0,
            "sections_crystallized": 0,
            "definitions_crystallized": 0,
            "code_blocks_verified": 0,
            "boilerplate_skipped": 0,
            "errors": 0,
        }
        if not d.exists() or not d.is_dir():
            return stats

        pattern = "**/*" if recursive else "*"
        for f in sorted(d.glob(pattern)):
            if f.is_file() and f.suffix.lower() in extensions:
                try:
                    f_stats = cls.ingest_file(target_brain, f, domain=domain or f"doc:{f.stem}")
                    stats["files_processed"] += 1
                    for k in ("sections_crystallized", "definitions_crystallized", "code_blocks_verified", "boilerplate_skipped"):
                        if k in f_stats:
                            stats[k] += f_stats[k]
                except Exception:
                    stats["errors"] += 1
        return stats
