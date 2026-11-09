import pytest

from mirror_crystal.capability_synthesizer import (
    CapabilitySynthesizer, Example, SynthesisError, SynthesisGoal,
)
from mirror_crystal.operational_language import UOL, VariableSpec


def test_counterexample_guided_loop_replaces_an_underfit_candidate():
    goal = SynthesisGoal(
        "square-x", "mathematics",
        (VariableSpec("x", "integer"), VariableSpec("y", "integer")),
        "integer",
        training=(Example({"x": 0, "y": 0}, 0), Example({"x": 1, "y": 0}, 1)),
        validation=(Example({"x": 0, "y": 1}, 0),),
        constants=(0, 1), max_depth=2, candidate_budget=20000,
    )
    synthesizer = CapabilitySynthesizer(UOL())
    report = synthesizer.synthesize(
        goal,
        counterexample_inputs=({"x": 2, "y": 0}, {"x": 4, "y": 0}),
        counterexample_oracle=lambda values: values["x"] * values["x"],
        max_rounds=4,
    )
    assert report.succeeded
    assert report.rounds >= 2
    assert report.counterexamples
    assert report.counterexamples[0].expected == 4
    assert report.counterexamples[0].actual == 2
    assert UOL().execute(report.program, {"x": 8, "y": 0}) == 64
    assert report.probes_passed >= 1


def test_cegis_probe_pool_must_be_separate_from_training_and_validation():
    goal = SynthesisGoal(
        "identity", "mathematics", (VariableSpec("x", "integer"),), "integer",
        (Example({"x": 1}, 1), Example({"x": 2}, 2)),
        (Example({"x": 3}, 3),),
    )
    with pytest.raises(SynthesisError):
        CapabilitySynthesizer().synthesize(
            goal, counterexample_inputs=({"x": 1},),
            counterexample_oracle=lambda values: values["x"],
        )


def test_cegis_requires_explicit_oracle_and_valid_oracle_output():
    goal = SynthesisGoal(
        "identity", "mathematics", (VariableSpec("x", "integer"),), "integer",
        (Example({"x": 1}, 1), Example({"x": 2}, 2)),
        (Example({"x": 3}, 3),),
    )
    with pytest.raises(SynthesisError):
        CapabilitySynthesizer().synthesize(goal, counterexample_inputs=({"x": 4},))
    with pytest.raises(SynthesisError):
        CapabilitySynthesizer().synthesize(
            goal, counterexample_inputs=({"x": 4},),
            counterexample_oracle=lambda values: "not-an-integer",
        )


def test_counterexample_oracle_failure_is_not_misreported_as_candidate_failure():
    goal = SynthesisGoal(
        "identity", "mathematics", (VariableSpec("x", "integer"),), "integer",
        (Example({"x": 1}, 1), Example({"x": 2}, 2)),
        (Example({"x": 3}, 3),),
    )
    def broken_oracle(_values):
        raise RuntimeError("oracle unavailable")
    with pytest.raises(SynthesisError, match="oracle failed"):
        CapabilitySynthesizer().synthesize(
            goal, counterexample_inputs=({"x": 4},),
            counterexample_oracle=broken_oracle,
        )


def test_synthesized_candidate_stays_unverified_after_cegis():
    from mirror_crystal.blueprint_contracts import EvidenceLevel
    goal = SynthesisGoal(
        "identity", "mathematics", (VariableSpec("x", "integer"),), "integer",
        (Example({"x": 1}, 1), Example({"x": 2}, 2)),
        (Example({"x": 3}, 3),),
    )
    synthesizer = CapabilitySynthesizer()
    report = synthesizer.synthesize(
        goal, counterexample_inputs=({"x": 4},),
        counterexample_oracle=lambda values: values["x"],
    )
    candidate = synthesizer.to_candidate_capability(report)
    assert candidate.verification.level is EvidenceLevel.UNVERIFIED
    assert not candidate.executable

def test_cegis_counts_a_probe_that_passes_before_a_later_counterexample():
    goal = SynthesisGoal(
        "square-x", "mathematics",
        (VariableSpec("x", "integer"), VariableSpec("y", "integer")),
        "integer",
        training=(Example({"x": 0, "y": 0}, 0), Example({"x": 1, "y": 0}, 1)),
        validation=(Example({"x": 0, "y": 3}, 0),),
        constants=(0, 1), max_depth=2, candidate_budget=20000,
    )
    report = CapabilitySynthesizer(UOL()).synthesize(
        goal,
        counterexample_inputs=({"x": 0, "y": 4}, {"x": 2, "y": 0}),
        counterexample_oracle=lambda values: values["x"] * values["x"],
        max_rounds=4,
    )

    assert report.succeeded
    assert report.rounds >= 2
    assert len(report.counterexamples) == 1
    assert report.probes_passed == 2
    assert UOL().execute(report.program, {"x": 8, "y": 0}) == 64
