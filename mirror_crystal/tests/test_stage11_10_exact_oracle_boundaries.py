"""Stage 11.10 boundary tests for exact mathematical oracle inputs."""
from fractions import Fraction

import pytest

from mirror_crystal.exact_math_oracles import (
    ExactCombinatoricsOracle,
    ExactLinearAlgebraOracle,
    ExactProbabilityOracle,
)


@pytest.mark.parametrize(
    "args",
    [
        (Fraction(-1, 10), Fraction(1, 2), Fraction(1, 2)),
        (Fraction(11, 10), Fraction(1, 2), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(-1, 10), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(1, 2), Fraction(11, 10)),
    ],
)
def test_bayes_rejects_values_outside_probability_domain(args):
    with pytest.raises(ValueError):
        ExactProbabilityOracle.bayes_theorem(*args)


def test_exact_probability_rejects_float_coercion():
    with pytest.raises(TypeError):
        ExactProbabilityOracle.bayes_theorem(0.1, Fraction(1, 2), Fraction(1, 3))


def test_binomial_distribution_validates_trials_and_probability():
    with pytest.raises(ValueError):
        ExactProbabilityOracle.binomial_expectation_and_variance(-1, Fraction(1, 2))
    with pytest.raises(ValueError):
        ExactProbabilityOracle.binomial_expectation_and_variance(3, Fraction(3, 2))


def test_hypergeometric_rejects_impossible_sample_domain():
    with pytest.raises(ValueError):
        ExactProbabilityOracle.hypergeometric_pmf(10, 3, 5, 4)


def test_combinatorics_rejects_negative_population_and_boolean_inputs():
    with pytest.raises(ValueError):
        ExactCombinatoricsOracle.combinations(-2, 1)
    with pytest.raises(TypeError):
        ExactCombinatoricsOracle.derangements(True)


def test_matrix_multiplication_rejects_ragged_and_empty_matrices():
    with pytest.raises(ValueError):
        ExactLinearAlgebraOracle.matrix_mult([[1, 2], [3]], [[1], [2]])
    with pytest.raises(ValueError):
        ExactLinearAlgebraOracle.matrix_mult([], [[1]])


def test_trace_requires_square_matrix():
    with pytest.raises(ValueError):
        ExactLinearAlgebraOracle.trace([[1, 2], [3, 4], [5, 6]])


@pytest.mark.parametrize("bad", [0.5, True, -1])
def test_polynomial_coefficients_reject_float_bool_and_negative_powers(bad):
    from mirror_crystal.exact_math_oracles import ExactCalculusOracle

    if bad == -1:
        with pytest.raises(ValueError):
            ExactCalculusOracle.evaluate_polynomial({-1: 2}, 3)
    else:
        with pytest.raises(TypeError):
            ExactCalculusOracle.evaluate_polynomial({1: bad}, 3)


def test_calculus_rejects_invalid_order_and_float_inputs():
    from mirror_crystal.exact_math_oracles import ExactCalculusOracle

    with pytest.raises(ValueError):
        ExactCalculusOracle.differentiate_polynomial({1: 2}, -1)
    with pytest.raises(TypeError):
        ExactCalculusOracle.integrate_polynomial_definite({1: 2}, 0.1, 2)
    with pytest.raises(TypeError):
        ExactCalculusOracle.evaluate_polynomial({1: 2}, 0.25)


def test_linear_system_rejects_float_and_boolean_matrix_entries():
    from mirror_crystal.exact_math_oracles import ExactAlgebraOracle

    matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    with pytest.raises(TypeError):
        ExactAlgebraOracle.solve_linear_system_3x3(
            [[0.1, *matrix[0][1:]], *matrix[1:]], [1, 2, 3]
        )
    with pytest.raises(TypeError):
        ExactAlgebraOracle.solve_linear_system_3x3(
            [[True, 0, 0], *matrix[1:]], [1, 2, 3]
        )


@pytest.mark.parametrize("oracle_call", [
    lambda: ExactLinearAlgebraOracle.matrix_mult([[1.0]], [[1]]),
    lambda: ExactLinearAlgebraOracle.determinant_3x3([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]),
    lambda: ExactLinearAlgebraOracle.matrix_inverse_2x2([[1, 0], [0, 0.5]]),
    lambda: ExactLinearAlgebraOracle.trace([[True]]),
])
def test_linear_algebra_rejects_float_and_boolean_entries(oracle_call):
    with pytest.raises(TypeError):
        oracle_call()


def test_exact_quadratic_rejects_boolean_coefficients():
    from mirror_crystal.exact_math_oracles import ExactAlgebraOracle

    with pytest.raises(TypeError):
        ExactAlgebraOracle.solve_quadratic(True, 2, 1)
