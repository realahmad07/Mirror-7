"""
Unit tests for the Independent Evaluation Suite and Teacher Disconnection Verifier.
"""
import pytest
from training.evaluation_suite import (
    EvaluationDimension,
    ALL_EVAL_DIMENSIONS,
    load_evaluation_benchmark,
    IndependentEvaluator,
    TeacherDisconnectionVerifier,
)


def test_evaluation_dimensions_coverage():
    assert len(ALL_EVAL_DIMENSIONS) == 10
    bench_all = load_evaluation_benchmark()
    bench_dev = load_evaluation_benchmark(partition="dev")
    bench_test = load_evaluation_benchmark(partition="held_out")

    assert len(bench_all) == 20
    assert len(bench_dev) == 10
    assert len(bench_test) == 10

    # Ensure every single dimension has coverage in both dev and held_out
    for dim in ALL_EVAL_DIMENSIONS:
        dev_cases = [e for e in bench_dev if e.dimension == dim]
        test_cases = [e for e in bench_test if e.dimension == dim]
        assert len(dev_cases) == 1, f"Missing dev benchmark case for {dim}"
        assert len(test_cases) == 1, f"Missing held_out benchmark case for {dim}"


def test_independent_evaluator_scoring():
    evaluator = IndependentEvaluator(partition="dev")

    # Mock high-quality reference generator
    def oracle_generator(prompt: str) -> str:
        for ex in evaluator.benchmark:
            if ex.prompt == prompt:
                return ex.reference_targets[0]
        return "I am unable to answer."

    report = evaluator.evaluate(oracle_generator, model_name="oracle_model")
    assert report.overall_score >= 0.90
    assert len(report.dimension_scores) == 10
    assert len(report.successes) >= 9

    # Mock degenerate generator
    def degenerate_generator(prompt: str) -> str:
        return "the the the the the the the"

    rep_degen = evaluator.evaluate(degenerate_generator, model_name="degen_model")
    assert rep_degen.overall_score == 0.0
    assert len(rep_degen.failures) == 10


def test_teacher_disconnection_verifier():
    verifier = TeacherDisconnectionVerifier(checkpoint_path="artifacts/baseline_v3_5ep.pt")
    result = verifier.run_full_disconnection_audit()

    assert result.is_fully_independent is True
    assert result.checkpoint_loaded_standalone is True
    assert result.zero_teacher_calls is True
    assert result.zero_credential_requirements is True
    assert result.backend_subsystems_operational is True
    assert result.unseen_evaluation_handled is True
