"""Stage 16 Test Suite: Advanced Reasoning Subsystem.

Tests:
1. Multi-step reasoning step types and state verification.
2. Exact rational mathematical verification using Fraction oracles.
3. Causal graph reasoning, path tracing, and cumulative effect propagation.
4. Pearl's do-calculus intervention simulation across causal variables.
5. Falsifiable hypothesis generation from diagnostic observations.
6. Dual-path independent answer checking with consensus confidence and divergence detection.
7. End-to-end multi-step reasoning orchestrator execution.
"""
from __future__ import annotations

from fractions import Fraction
import pytest

from mirror_crystal.advanced_reasoning import (
    AdvancedReasoningEngine,
    CausalEdge,
    CausalReasoningGraph,
    ExactMathVerifier,
    HypothesisGenerator,
    IndependentAnswerChecker,
    ReasoningError,
    ReasoningStep,
    ReasoningStepType,
)


def test_reasoning_step_creation_and_serialization():
    """Verify reasoning step structure and serialization."""
    step = ReasoningStep(
        step_number=1,
        step_type=ReasoningStepType.DEDUCTION,
        premise="All continuous functions on [a, b] are integrable.",
        deduction="f(x) = x^2 is continuous on [0, 1], therefore it is integrable.",
        intermediate_value="integrable",
    )
    assert step.step_number == 1
    assert step.step_type == ReasoningStepType.DEDUCTION
    d = step.to_dict()
    assert d["step_type"] == "deduction"
    assert d["intermediate_value"] == "integrable"


def test_exact_rational_mathematical_verification():
    """Verify exact Fraction evaluations prevent floating point hallucinations."""
    # 1. Exact equality: 1/3 + 1/6 == 1/2 (since 2/6 + 1/6 = 3/6 = 1/2)
    valid, note = ExactMathVerifier.verify_rational_equality("1/2", "1/2")
    assert valid is True

    # 2. Sum sequence: 1/4 + 1/4 + 1/2 == 1
    valid_sum, sum_note = ExactMathVerifier.verify_sum_sequence(["1/4", "1/4", "1/2"], "1")
    assert valid_sum is True
    assert "Sum=1" in sum_note

    # 3. Mismatch detection: 3/5 != 2/3
    mismatch, m_note = ExactMathVerifier.verify_rational_equality("3/5", "2/3")
    assert mismatch is False
    assert "!=" in m_note


def test_causal_graph_path_and_effect_propagation():
    """Verify causal effect calculation across directed chains."""
    graph = CausalReasoningGraph()
    graph.add_edge("price_increase", "demand_drop", -0.8)
    graph.add_edge("demand_drop", "inventory_accumulation", 0.7)

    strength, path = graph.query_causal_chain("price_increase", "inventory_accumulation")
    assert path == ["price_increase", "demand_drop", "inventory_accumulation"]
    # Net effect: -0.8 * 0.7 = -0.56
    assert strength == -0.56

    # Non-existent path returns 0.0
    s_none, p_none = graph.query_causal_chain("price_increase", "unrelated_var")
    assert s_none == 0.0
    assert len(p_none) == 0


def test_pearl_do_intervention_simulation():
    """Verify interventional do(X = x) updates downstream causal nodes."""
    graph = CausalReasoningGraph()
    graph.add_edge("code_review_coverage", "defect_rate", -0.6)
    graph.add_edge("defect_rate", "customer_complaints", 0.9)

    # Base state: default 0
    # Intervene: do(code_review_coverage = 1.0)
    sim_states = graph.simulate_do_intervention("code_review_coverage", 1.0)

    assert sim_states["code_review_coverage"] == 1.0
    assert sim_states["defect_rate"] == -0.6
    # -0.6 * 0.9 = -0.54
    assert sim_states["customer_complaints"] == -0.54


def test_hypothesis_generation_for_diagnostics():
    """Verify hypothesis generator formulates testable propositions and falsifiable checks."""
    hyps = HypothesisGenerator.generate_hypotheses("API latency is unusually high during peak hours")
    assert len(hyps) >= 2
    assert any("connection pool" in h["statement"].lower() for h in hyps)
    assert any("falsifiable_prediction" in h for h in hyps)


def test_dual_path_independent_answer_checking():
    """Verify consensus certification when two orthogonal methods agree, and divergence flagging when they disagree."""
    # 1. Numerical consensus
    agreed, conf, note = IndependentAnswerChecker.verify_dual_path(
        problem_description="Calculate area of circle with r=3",
        path_a_solution=28.2743338,
        path_b_solution=28.2743339,
    )
    assert agreed is True
    assert conf >= 0.95
    assert "converged" in note

    # 2. Numerical divergence
    diverged, d_conf, d_note = IndependentAnswerChecker.verify_dual_path(
        problem_description="Calculate determinant of matrix",
        path_a_solution=42,
        path_b_solution=17,
    )
    assert diverged is False
    assert d_conf < 0.20
    assert "diverged" in d_note

    # 3. Discrete exact consensus
    agree_discrete, disc_conf, disc_note = IndependentAnswerChecker.verify_dual_path(
        problem_description="Check graph acyclicity",
        path_a_solution="acyclic",
        path_b_solution="acyclic",
    )
    assert agree_discrete is True
    assert disc_conf >= 0.95


def test_end_to_end_advanced_reasoning_orchestrator():
    """Verify end-to-end multi-step reasoning validation and secondary oracle agreement."""
    engine = AdvancedReasoningEngine()

    steps = [
        ReasoningStep(
            step_number=1,
            step_type=ReasoningStepType.DEDUCTION,
            premise="A triangle has angles alpha=60 and beta=70.",
            deduction="Sum of interior angles in Euclidean plane is 180.",
            intermediate_value=180,
        ),
        ReasoningStep(
            step_number=2,
            step_type=ReasoningStepType.MATHEMATICAL_VERIFICATION,
            premise="gamma = 180 - (alpha + beta)",
            deduction="180 - (60 + 70) == 50",  # 180 - 130 = 50
            intermediate_value=50,
        ),
    ]

    # Independent second-path calculation
    def independent_oracle():
        return 50

    result = engine.solve_multi_step_problem(
        problem="Find the third angle of triangle with angles 60 and 70",
        steps=steps,
        independent_verification_fn=independent_oracle,
    )

    assert result["all_steps_valid"] is True
    assert result["consensus_verified"] is True
    assert result["calibrated_confidence"] >= 0.95
    assert result["final_answer"] == 50
    assert len(result["steps"]) == 2
