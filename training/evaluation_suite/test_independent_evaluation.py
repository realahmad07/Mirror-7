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


def test_teacher_disconnection_verifier(tmp_path):
    from training.native_conversation_v5 import (
        build_native_conversation_v5,
        make_v5_enhanced_config,
        save_checkpoint,
    )

    config = make_v5_enhanced_config(
        embedding_dim=16,
        hidden_dim=32,
        state_dim=16,
        attention_dim=16,
        max_bytes=64,
        num_heads=2,
        dropout=0.0,
        scheduled_sampling_p=0.0,
    )
    model = build_native_conversation_v5(config)
    checkpoint = tmp_path / "offline-v5.pt"
    save_checkpoint(checkpoint, model, config, metadata={"test_fixture": True})

    verifier = TeacherDisconnectionVerifier(checkpoint_path=checkpoint)
    result = verifier.run_full_disconnection_audit()

    assert result.is_fully_independent is True
    assert result.checkpoint_loaded_standalone is True
    assert result.zero_teacher_calls is True
    assert result.zero_credential_requirements is True
    assert result.backend_subsystems_operational is True
    assert result.unseen_evaluation_handled is True
    assert result.held_out_cases == 10
    assert result.checkpoint_format == "mirror7-native-conversation-v5-input-conditioned"


def test_teacher_disconnection_fails_when_checkpoint_is_missing(tmp_path):
    verifier = TeacherDisconnectionVerifier(checkpoint_path=tmp_path / "missing.pt")
    result = verifier.run_full_disconnection_audit()
    assert result.is_fully_independent is False
    assert result.checkpoint_loaded_standalone is False
    assert result.unseen_evaluation_handled is False
