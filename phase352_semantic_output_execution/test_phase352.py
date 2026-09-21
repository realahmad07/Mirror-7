from phase34_reasoning_planning.mirror7_phase34 import GoalPlanner, TransitionModel
from phase343_semantic_state import SemanticStateInducer
from phase348_semantic_planning_bridge import build_planning_request
from phase352_semantic_output_execution import (
    desired_output_policy,
    execute_output_aware_planning,
    output_intent_is_consistent,
)


def make_model():
    model = TransitionModel()
    for x in range(2):
        model.observe({"x": x, "topic": "variance"}, "inc", {"x": x + 1, "topic": "variance"})
    return model


def test_desired_output_is_derived_from_semantic_state():
    state = SemanticStateInducer().discover("Explain variance")
    request = build_planning_request(state, {"x": 0, "topic": "variance"}, {"x": 1, "topic": "variance"})
    policy = desired_output_policy(request)
    assert policy.desired_output == "explanation"
    assert policy.output_mode == "explain"


def test_output_intent_does_not_change_world_state():
    state = SemanticStateInducer().discover("Explain variance")
    request = build_planning_request(state, {"x": 0, "topic": "variance"}, {"x": 1, "topic": "variance"})
    before = dict(request.world_state)
    result = execute_output_aware_planning(request, GoalPlanner(make_model()))
    assert result.plan is not None
    assert request.world_state == before


def test_output_intent_does_not_invent_actions():
    state = SemanticStateInducer().discover("Explain variance")
    request = build_planning_request(state, {"x": 0, "topic": "variance"}, {"x": 1, "topic": "variance"})
    result = execute_output_aware_planning(request, GoalPlanner(make_model()))
    assert [step.action for step in result.plan.steps] == ["inc"]
    assert output_intent_is_consistent(result)


def test_unknown_output_is_preserved_without_fabrication():
    state = SemanticStateInducer().discover("variance")
    request = build_planning_request(state, {"x": 0}, {"x": 1})
    result = execute_output_aware_planning(request, GoalPlanner(make_model()))
    assert result.output.desired_output is None
    assert result.output.output_mode is None
    assert output_intent_is_consistent(result)


def test_context_output_survives_continuation():
    inducer = SemanticStateInducer()
    first = inducer.discover("Explain variance")
    second = inducer.discover("Continue under 2 steps", previous=first)
    request = build_planning_request(second, {"x": 0, "topic": "variance"}, {"x": 1, "topic": "variance"})
    policy = desired_output_policy(request)
    assert policy.desired_output == "explanation"
    assert policy.output_mode == "explain"
