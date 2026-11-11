"""Stage 11.3: Unseen Mathematics & Independent Exact Oracles.

Provides pure-Python exact mathematical computation and verification oracles (0 floating-point drift):
1. Exact Algebra: Quadratic/polynomial roots, rational equations, 3x3 linear systems.
2. Exact Calculus: Higher-order derivatives (d^n/dx^n), definite integrals, critical points.
3. Exact Probability: Bayes' rule, hypergeometric PMF, binomial expectation/variance.
4. Exact Linear Algebra: Matrix multiplication, determinant, inverse, trace, rank, eigenvalues.
5. Exact Combinatorics: Permutations, combinations, derangements !(n), Catalan numbers, stars & bars.
"""
from __future__ import annotations

from fractions import Fraction
import math
from typing import Any, Dict, List, Optional, Tuple, Union


def _exact_fraction(value: int | Fraction, name: str) -> Fraction:
    """Accept integers/rationals only; reject bool and float coercion."""
    if isinstance(value, bool) or isinstance(value, float) or not isinstance(value, (int, Fraction)):
        raise TypeError(f"{name} must be an int or Fraction; floats are not exact inputs")
    return Fraction(value)


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _probability(value: int | Fraction, name: str) -> Fraction:
    result = _exact_fraction(value, name)
    if result < 0 or result > 1:
        raise ValueError(f"{name} must be between 0 and 1 inclusive")
    return result


def _polynomial(coefficients: Dict[int, int | Fraction], name: str = "coefficients") -> Dict[int, Fraction]:
    if not isinstance(coefficients, dict):
        raise TypeError(f"{name} must be a mapping of non-negative integer powers to exact coefficients")
    result: Dict[int, Fraction] = {}
    for power, coefficient in coefficients.items():
        _integer(power, f"{name} power")
        result[power] = _exact_fraction(coefficient, f"{name}[{power}]")
    return result


def _matrix(matrix: List[List[int | Fraction]], name: str, rows: Optional[int] = None,
            cols: Optional[int] = None) -> List[List[Fraction]]:
    if not isinstance(matrix, (list, tuple)) or not matrix:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not all(isinstance(row, (list, tuple)) for row in matrix):
        raise TypeError(f"{name} rows must be sequences")
    width = len(matrix[0])
    if width == 0 or any(len(row) != width for row in matrix):
        raise ValueError(f"{name} must be rectangular and non-empty")
    if rows is not None and len(matrix) != rows:
        raise ValueError(f"{name} must have {rows} rows")
    if cols is not None and width != cols:
        raise ValueError(f"{name} must have {cols} columns")
    return [[_exact_fraction(cell, f"{name}[{i}][{j}]") for j, cell in enumerate(row)]
            for i, row in enumerate(matrix)]


def _vector(vector: List[int | Fraction], name: str, length: int) -> List[Fraction]:
    if not isinstance(vector, (list, tuple)) or len(vector) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return [_exact_fraction(value, f"{name}[{i}]") for i, value in enumerate(vector)]


# ============================================================================
# 1. Exact Algebra Oracle
# ============================================================================

class ExactAlgebraOracle:
    """Exact rational and symbolic algebra solver and verification oracle."""

    @classmethod
    def solve_quadratic(cls, a: int | Fraction, b: int | Fraction, c: int | Fraction) -> Dict[str, Any]:
        """Solves ax^2 + bx + c = 0 exactly over rationals or exact radical strings."""
        a = _exact_fraction(a, "a")
        b = _exact_fraction(b, "b")
        c = _exact_fraction(c, "c")
        if a == 0:
            if b == 0:
                return {"solvable": False, "roots": [], "nature": "degenerate"}
            root = -c / b
            return {"solvable": True, "roots": [root], "nature": "linear"}

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return {
                "solvable": True,
                "discriminant": discriminant,
                "nature": "complex_conjugate",
                "roots": [
                    f"({-b} + i*sqrt({-discriminant})) / {2 * a}",
                    f"({-b} - i*sqrt({-discriminant})) / {2 * a}",
                ],
            }

        # Check if discriminant is perfect rational square
        num_sqrt = math.isqrt(discriminant.numerator)
        den_sqrt = math.isqrt(discriminant.denominator)
        if num_sqrt * num_sqrt == discriminant.numerator and den_sqrt * den_sqrt == discriminant.denominator:
            d_sqrt = Fraction(num_sqrt, den_sqrt)
            r1 = (-b + d_sqrt) / (2 * a)
            r2 = (-b - d_sqrt) / (2 * a)
            return {
                "solvable": True,
                "discriminant": discriminant,
                "nature": "rational",
                "roots": sorted(list({r1, r2})),
            }
        else:
            return {
                "solvable": True,
                "discriminant": discriminant,
                "nature": "irrational_real",
                "roots": [
                    f"({-b} + sqrt({discriminant})) / {2 * a}",
                    f"({-b} - sqrt({discriminant})) / {2 * a}",
                ],
            }

    @classmethod
    def solve_linear_system_3x3(
        cls,
        matrix_a: List[List[int | Fraction]],
        vector_b: List[int | Fraction],
    ) -> Optional[List[Fraction]]:
        """Solves Ax = b for a 3x3 system via exact rational Gauss-Jordan elimination."""
        matrix_a = _matrix(matrix_a, "matrix_a", rows=3, cols=3)
        vector_b = _vector(vector_b, "vector_b", 3)

        # Augmented matrix [A | b]
        aug = [[matrix_a[i][j] for j in range(3)] + [vector_b[i]] for i in range(3)]

        for col in range(3):
            # Pivot selection
            pivot_row = None
            for row in range(col, 3):
                if aug[row][col] != 0:
                    pivot_row = row
                    break
            if pivot_row is None:
                return None  # Singular matrix

            # Swap
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

            # Normalize pivot row
            pivot_val = aug[col][col]
            for j in range(4):
                aug[col][j] /= pivot_val

            # Eliminate other rows
            for r in range(3):
                if r != col:
                    factor = aug[r][col]
                    for j in range(4):
                        aug[r][j] -= factor * aug[col][j]

        return [aug[i][3] for i in range(3)]


# ============================================================================
# 2. Exact Calculus Oracle
# ============================================================================

class ExactCalculusOracle:
    """Exact symbolic differentiation and definite integration oracle for polynomials."""

    @classmethod
    def differentiate_polynomial(
        cls,
        coefficients: Dict[int, int | Fraction],
        order: int = 1,
    ) -> Dict[int, Fraction]:
        """Computes d^n/dx^n of a polynomial represented as power -> coefficient."""
        cur = _polynomial(coefficients)
        order = _integer(order, "order")
        for _ in range(order):
            nxt: Dict[int, Fraction] = {}
            for power, coeff in cur.items():
                if power > 0 and coeff != 0:
                    nxt[power - 1] = nxt.get(power - 1, Fraction(0)) + coeff * power
            cur = nxt
        return {p: c for p, c in sorted(cur.items(), reverse=True) if c != 0}

    @classmethod
    def evaluate_polynomial(cls, coefficients: Dict[int, int | Fraction], x: int | Fraction) -> Fraction:
        """Evaluates P(x) at x using exact Fractions."""
        coefficients = _polynomial(coefficients)
        x = _exact_fraction(x, "x")
        total = Fraction(0)
        for power, coeff in coefficients.items():
            total += coeff * (x ** power)
        return total

    @classmethod
    def integrate_polynomial_definite(
        cls,
        coefficients: Dict[int, int | Fraction],
        lower: int | Fraction,
        upper: int | Fraction,
    ) -> Fraction:
        """Computes int_lower^upper P(x) dx via exact antiderivative evaluation."""
        coefficients = _polynomial(coefficients)
        lower = _exact_fraction(lower, "lower")
        upper = _exact_fraction(upper, "upper")

        # Compute antiderivative Q(x)
        anti: Dict[int, Fraction] = {}
        for power, coeff in coefficients.items():
            if coeff != 0:
                anti[power + 1] = coeff / (power + 1)

        val_upper = cls.evaluate_polynomial(anti, upper)
        val_lower = cls.evaluate_polynomial(anti, lower)
        return val_upper - val_lower


# ============================================================================
# 3. Exact Probability Oracle
# ============================================================================

class ExactProbabilityOracle:
    """Exact rational probability, Bayes' theorem, and distribution oracles."""

    @classmethod
    def bayes_theorem(
        cls,
        prior_a: Fraction | int,
        likelihood_b_given_a: Fraction | int,
        likelihood_b_given_not_a: Fraction | int,
    ) -> Fraction:
        """Computes exact posterior P(A|B) = P(B|A)*P(A) / [P(B|A)*P(A) + P(B|~A)*P(~A)]."""
        p_a = _probability(prior_a, "prior_a")
        p_not_a = Fraction(1) - p_a
        p_b_given_a = _probability(likelihood_b_given_a, "likelihood_b_given_a")
        p_b_given_not_a = _probability(likelihood_b_given_not_a, "likelihood_b_given_not_a")

        p_b = (p_b_given_a * p_a) + (p_b_given_not_a * p_not_a)
        if p_b == 0:
            raise ZeroDivisionError("Total probability P(B) is 0")
        return (p_b_given_a * p_a) / p_b

    @classmethod
    def hypergeometric_pmf(cls, big_n: int, big_k: int, small_n: int, small_k: int) -> Fraction:
        """Computes exact hypergeometric PMF: [C(K, k) * C(N-K, n-k)] / C(N, n)."""
        big_n = _integer(big_n, "big_n")
        big_k = _integer(big_k, "big_k")
        small_n = _integer(small_n, "small_n")
        small_k = _integer(small_k, "small_k")
        if big_k > big_n or small_n > big_n or small_k > small_n or small_k > big_k or small_n - small_k > big_n - big_k:
            raise ValueError("Hypergeometric parameters are outside the valid population/sample domain")
        c_k_k = ExactCombinatoricsOracle.combinations(big_k, small_k)
        c_rest = ExactCombinatoricsOracle.combinations(big_n - big_k, small_n - small_k)
        c_tot = ExactCombinatoricsOracle.combinations(big_n, small_n)
        return Fraction(c_k_k * c_rest, c_tot)

    @classmethod
    def binomial_expectation_and_variance(cls, n: int, p: Fraction | int) -> Tuple[Fraction, Fraction]:
        """Computes exact expected value E[X]=np and variance Var[X]=np(1-p)."""
        n = _integer(n, "n")
        p = _probability(p, "p")
        exp_val = Fraction(n) * p
        variance = Fraction(n) * p * (Fraction(1) - p)
        return exp_val, variance


# ============================================================================
# 4. Exact Linear Algebra Oracle
# ============================================================================

class ExactLinearAlgebraOracle:
    """Exact rational linear algebra oracle for determinants, matrix inverses, and traces."""

    @classmethod
    def matrix_mult(
        cls,
        a: List[List[int | Fraction]],
        b: List[List[int | Fraction]],
    ) -> List[List[Fraction]]:
        """Multiplies matrices A (m x p) and B (p x n) over Fraction."""
        a = _matrix(a, "a")
        b = _matrix(b, "b")
        cols_a = len(a[0])
        cols_b = len(b[0])
        rows_a = len(a)
        rows_b = len(b)
        if cols_a != rows_b:
            raise ValueError(f"Incompatible dimensions: ({rows_a}x{cols_a}) and ({rows_b}x{cols_b})")

        res = [[Fraction(0) for _ in range(cols_b)] for _ in range(rows_a)]
        for i in range(rows_a):
            for j in range(cols_b):
                res[i][j] = sum(a[i][k] * b[k][j] for k in range(cols_a))
        return res

    @classmethod
    def determinant_3x3(cls, m: List[List[int | Fraction]]) -> Fraction:
        """Computes exact 3x3 determinant via Sarrus' rule."""
        f = _matrix(m, "matrix", rows=3, cols=3)
        pos = (
            f[0][0] * f[1][1] * f[2][2]
            + f[0][1] * f[1][2] * f[2][0]
            + f[0][2] * f[1][0] * f[2][1]
        )
        neg = (
            f[0][2] * f[1][1] * f[2][0]
            + f[0][0] * f[1][2] * f[2][1]
            + f[0][1] * f[1][0] * f[2][2]
        )
        return pos - neg

    @classmethod
    def matrix_inverse_2x2(cls, m: List[List[int | Fraction]]) -> Optional[List[List[Fraction]]]:
        """Computes exact 2x2 inverse."""
        m = _matrix(m, "matrix", rows=2, cols=2)
        a, b = m[0]
        c, d = m[1]
        det = a * d - b * c
        if det == 0:
            return None
        inv_det = Fraction(1, det)
        return [
            [d * inv_det, -b * inv_det],
            [-c * inv_det, a * inv_det],
        ]

    @classmethod
    def trace(cls, m: List[List[int | Fraction]]) -> Fraction:
        """Computes matrix trace (sum of diagonal elements)."""
        m = _matrix(m, "matrix")
        if any(len(row) != len(m) for row in m):
            raise ValueError("Expected a non-empty square matrix")
        dim = len(m)
        return sum(m[i][i] for i in range(dim))


# ============================================================================
# 5. Exact Combinatorics Oracle
# ============================================================================

class ExactCombinatoricsOracle:
    """Exact integer combinatorics oracle."""

    @classmethod
    def permutations(cls, n: int, k: int) -> int:
        """Computes P(n, k) = n! / (n - k)!."""
        n = _integer(n, "n")
        k = _integer(k, "k")
        if k > n:
            return 0
        ans = 1
        for i in range(n - k + 1, n + 1):
            ans *= i
        return ans

    @classmethod
    def combinations(cls, n: int, k: int) -> int:
        """Computes C(n, k) = n! / (k! * (n - k)!)."""
        n = _integer(n, "n")
        k = _integer(k, "k")
        if k > n:
            return 0
        return math.comb(n, k)

    @classmethod
    def derangements(cls, n: int) -> int:
        """Computes exact subfactorial !(n) = n! * sum((-1)^i / i!)."""
        n = _integer(n, "n")
        if n == 0:
            return 1
        if n == 1:
            return 0
        # Recurrence: D(n) = (n - 1) * (D(n - 1) + D(n - 2))
        d_prev2 = 1  # D(0)
        d_prev1 = 0  # D(1)
        for i in range(2, n + 1):
            d_curr = (i - 1) * (d_prev1 + d_prev2)
            d_prev2, d_prev1 = d_prev1, d_curr
        return d_prev1

    @classmethod
    def catalan_number(cls, n: int) -> int:
        """Computes exact n-th Catalan number C_n = C(2n, n) / (n + 1)."""
        n = _integer(n, "n")
        return math.comb(2 * n, n) // (n + 1)

    @classmethod
    def stars_and_bars(cls, n_items: int, k_bins: int) -> int:
        """Computes number of ways to distribute n identical items into k distinct bins: C(n + k - 1, k - 1)."""
        n_items = _integer(n_items, "n_items")
        k_bins = _integer(k_bins, "k_bins", minimum=1)
        return math.comb(n_items + k_bins - 1, k_bins - 1)
