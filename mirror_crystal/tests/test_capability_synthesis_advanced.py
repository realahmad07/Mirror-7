import pytest

from mirror_crystal.capability_synthesizer import (
    CapabilitySynthesizer, Example, SynthesisGoal, SynthesisStatus,
)
from mirror_crystal.operational_language import UOL, VariableSpec


def test_synthesizes_conditional_selection_from_a_boolean_goal():
    goal = SynthesisGoal(
        "select-by-condition", "programming",
        (
            VariableSpec("choose_left", "boolean"),
            VariableSpec("left", "integer"),
            VariableSpec("right", "integer"),
        ),
        "integer",
        training=(
            Example({"choose_left": True, "left": 10, "right": 20}, 10),
            Example({"choose_left": False, "left": 8, "right": 2}, 2),
        ),
        validation=(Example({"choose_left": True, "left": 7, "right": 9}, 7),),
        constants=(0, 1),
        max_depth=2,
        candidate_budget=20000,
    )
    report = CapabilitySynthesizer().synthesize(goal)
    assert report.succeeded
    assert report.validation_passed == 1
    assert UOL().execute(report.program, {"choose_left": False, "left": 99, "right": 3}) == 3


def test_synthesizes_map_transformation_over_array_inputs():
    goal = SynthesisGoal(
        "increment-each", "programming", (VariableSpec("values", "array"),), "array",
        training=(
            Example({"values": [1, 2]}, (2, 3)),
            Example({"values": [4, 5, 6]}, (5, 6, 7)),
        ),
        validation=(Example({"values": [10]}, (11,)),),
        constants=(1,), max_depth=2, candidate_budget=20000,
    )
    report = CapabilitySynthesizer().synthesize(goal)
    assert report.succeeded
    assert UOL().execute(report.program, {"values": [0, 9]}) == (1, 10)


def test_synthesizes_fold_aggregation_from_array_examples():
    goal = SynthesisGoal(
        "sum-values", "mathematics", (VariableSpec("values", "array"),), "integer",
        training=(
            Example({"values": [1, 2]}, 3),
            Example({"values": [3, 4]}, 7),
        ),
        validation=(Example({"values": [5, 6]}, 11),),
        constants=(), max_depth=2, candidate_budget=20000,
        enable_map=False,
    )
    report = CapabilitySynthesizer().synthesize(goal)
    assert report.succeeded
    assert UOL().execute(report.program, {"values": [7, 8, 9]}) == 24


def test_advanced_search_budget_remains_explicit():
    goal = SynthesisGoal(
        "bounded-conditional", "programming",
        (VariableSpec("condition", "boolean"), VariableSpec("x", "integer")),
        "integer",
        training=(Example({"condition": True, "x": 1}, 1), Example({"condition": False, "x": 2}, 0)),
        validation=(Example({"condition": True, "x": 3}, 3),),
        constants=(0, 1), max_depth=4, candidate_budget=1,
    )
    assert CapabilitySynthesizer().synthesize(goal).status is SynthesisStatus.BUDGET_EXHAUSTED
