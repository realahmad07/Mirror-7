"""Independent exact arithmetic evaluator.

This is a *second*, separately implemented evaluator (Python ``ast`` over
``fractions.Fraction``). When its result agrees with the Cerebrum/CAS result the
orchestrator may label the answer ``verified_computation``: two independent
implementations produced the same exact value.
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from .text import normalize
from .types import NativeAnswer

_MAX_EXPONENT = 4096
_MAX_DIGITS = 4000
_MAX_LEN = 240

_PREFIX = re.compile(
    r"^(?:please\s+)?(?:what(?:'s| is| are)|whats|calculate|compute|evaluate|solve|find|work out|"
    r"how much is|how much are|tell me|can you (?:calculate|compute|tell me)|what does)\s+",
    re.I,
)
_SUFFIX = re.compile(r"\s*(?:equal|equals|come to|make)?\s*[?.!=]*\s*$", re.I)

_WORD_OPS: tuple[tuple[str, str], ...] = (
    (r"\bmultiplied by\b", "*"), (r"\btimes\b", "*"), (r"\bdivided by\b", "/"), (r"\bover\b", "/"),
    (r"\bplus\b", "+"), (r"\badded to\b", "+"), (r"\bminus\b", "-"), (r"\bsubtract(?:ed)? from\b", "-from-"),
    (r"\bto the power of\b", "**"), (r"\braised to(?: the power of)?\b", "**"), (r"\bmod(?:ulo)?\b", "%"),
    (r"\bsquared\b", "**2"), (r"\bcubed\b", "**3"),
    ("×", "*"), ("÷", "/"), ("−", "-"), ("^", "**"),
)
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}


class ArithmeticError_(ValueError):
    pass


@dataclass(frozen=True)
class ArithmeticResult:
    expression: str
    value: Fraction
    exact: bool
    float_value: float | None = None

    def display(self) -> str:
        if not self.exact and self.float_value is not None:
            return _fmt_float(self.float_value)
        return format_fraction(self.value)


def _fmt_float(x: float) -> str:
    if math.isfinite(x) and abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    return f"{x:.10g}"


def format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    decimal = value.numerator / value.denominator
    # Terminating decimal? (denominator only has factors 2 and 5)
    d = value.denominator
    while d % 2 == 0:
        d //= 2
    while d % 5 == 0:
        d //= 5
    if d == 1:
        text = (Decimal(value.numerator) / Decimal(value.denominator)).normalize()
        return format(text, "f")
    return f"{value.numerator}/{value.denominator} (≈ {decimal:.10g})"


def _to_fraction(token: str) -> Fraction:
    try:
        return Fraction(Decimal(token.replace(",", "")))
    except (InvalidOperation, ValueError) as exc:
        raise ArithmeticError_(f"bad number {token!r}") from exc


def extract_expression(prompt: str) -> str | None:
    """Turn a natural-language arithmetic request into a bare expression, or None."""
    text = normalize(prompt)
    if len(text) > _MAX_LEN:
        return None
    text = _PREFIX.sub("", text, count=1)
    text = _SUFFIX.sub("", text)
    text = re.sub(r"^(?:the (?:value|result|answer) of|the sum of|the product of)\s+", "", text, flags=re.I)
    low = text.casefold()
    # Percent-of: "15% of 80" / "15 percent of 80"
    low = re.sub(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+", r"(\1/100)*", low)
    # sqrt forms
    low = re.sub(r"(?:the\s+)?square root of\s+(\d+(?:\.\d+)?)", r"sqrt(\1)", low)
    low = re.sub(r"√\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", low)
    for pattern, repl in _WORD_OPS:
        low = re.sub(pattern, f" {repl} ", low) if pattern.startswith("\\b") else low.replace(pattern, f" {repl} ")
    # "subtract A from B" -> B - A
    m = re.fullmatch(r"\s*subtract\s+(.+?)\s+-from-\s+(.+)", low)
    if m:
        low = f"({m.group(2)}) - ({m.group(1)})"
    elif "-from-" in low:
        a, b = low.split("-from-", 1)
        low = f"({b}) - ({a})"
    # small number words
    low = re.sub(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", lambda mm: str(_NUMBER_WORDS[mm.group(1)]), low)
    # implicit multiplication "x" between numbers: "12 x 4"
    low = re.sub(r"(?<=\d)\s*x\s*(?=\d)", " * ", low)
    low = low.replace(",", "") if re.search(r"\d,\d{3}", low) else low
    low = re.sub(r"\s+", " ", low).strip()
    if not low or not re.search(r"\d", low):
        return None
    if not re.fullmatch(r"[\d\s.+\-*/%()]*(?:sqrt\([\d.]+\)[\d\s.+\-*/%()]*)*", low):
        return None
    # Must contain at least one operator or function; a bare number is not a calculation.
    if not re.search(r"[+\-*/%]|sqrt", low.lstrip("-")):
        return None
    return low


class IndependentArithmeticEvaluator:
    """Exact evaluator over a whitelisted AST."""

    def evaluate(self, expression: str) -> ArithmeticResult:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ArithmeticError_("not a valid expression") from exc
        exact = [True]
        value = self._eval(tree.body, exact)
        float_value = None
        if not exact[0]:
            float_value = float(value)
        return ArithmeticResult(expression=expression, value=value, exact=exact[0], float_value=float_value)

    def _eval(self, node: ast.AST, exact: list[bool]) -> Fraction:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return _to_fraction(repr(node.value)) if isinstance(node.value, float) else Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = self._eval(node.operand, exact)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left, exact)
            right = self._eval(node.right, exact)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return self._guard(left * right)
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ArithmeticError_("division by zero")
                return left / right
            if isinstance(node.op, ast.Mod):
                if right == 0:
                    raise ArithmeticError_("modulo by zero")
                return left % right
            if isinstance(node.op, ast.Pow):
                if right.denominator != 1 or abs(right.numerator) > _MAX_EXPONENT:
                    raise ArithmeticError_("exponent too large or not an integer")
                if left == 0 and right < 0:
                    raise ArithmeticError_("division by zero")
                return self._guard(left ** int(right))
        if (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt"
            and len(node.args) == 1 and not node.keywords
        ):
            v = self._eval(node.args[0], exact)
            if v < 0:
                raise ArithmeticError_("square root of a negative number")
            num_root, den_root = math.isqrt(v.numerator), math.isqrt(v.denominator)
            if num_root * num_root == v.numerator and den_root * den_root == v.denominator:
                return Fraction(num_root, den_root)
            exact[0] = False
            return Fraction(math.sqrt(v.numerator / v.denominator))
        raise ArithmeticError_(f"unsupported expression element: {type(node).__name__}")

    @staticmethod
    def _guard(value: Fraction) -> Fraction:
        if len(str(abs(value.numerator))) > _MAX_DIGITS or len(str(value.denominator)) > _MAX_DIGITS:
            raise ArithmeticError_("result too large")
        return value


def _pretty(expression: str) -> str:
    return expression.replace("**", "^").replace("*", "×").replace("/", "÷")


class ArithmeticOrgan:
    def __init__(self) -> None:
        self.evaluator = IndependentArithmeticEvaluator()

    def respond(self, prompt: str) -> NativeAnswer | None:
        expression = extract_expression(prompt)
        if expression is None:
            return None
        try:
            result = self.evaluator.evaluate(expression)
        except ArithmeticError_ as exc:
            if "division by zero" in str(exc) or "modulo by zero" in str(exc):
                return NativeAnswer(
                    kind="arithmetic",
                    answer="That is undefined — you can't divide by zero.",
                    qualification="computed_result",
                    confidence=0.95,
                    metadata={"expression": expression, "error": str(exc)},
                )
            return None
        shown = result.display()
        answer = f"{_pretty(expression)} = {shown}" if result.exact else f"{_pretty(expression)} ≈ {shown}"
        return NativeAnswer(
            kind="arithmetic",
            answer=answer,
            qualification="computed_result",
            confidence=0.99 if result.exact else 0.9,
            steps=(f"Parsed expression: {expression}", f"Exact rational evaluation: {result.value}"),
            metadata={
                "expression": expression,
                "value": str(result.value),
                "display": shown,
                "exact": result.exact,
            },
        )


def values_agree(native_value: str, other_answer: str) -> bool:
    """True when ``other_answer`` states the same exact value as ``native_value`` (a Fraction string)."""
    try:
        target = Fraction(native_value)
    except (ValueError, ZeroDivisionError):
        return False
    text = (other_answer or "").replace(",", "")
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    candidates = boxed or re.findall(r"=\s*(-?\d+(?:\.\d+)?(?:/\d+)?)", text) or re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)[-1:]
    for cand in candidates:
        cand = cand.strip().replace(" ", "")
        m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(\d+)\}", cand)
        if m:
            cand = f"{m.group(1)}/{m.group(2)}"
        try:
            if Fraction(cand) == target:
                return True
        except (ValueError, ZeroDivisionError):
            continue
    return False
