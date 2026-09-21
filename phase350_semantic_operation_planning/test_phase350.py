from phase34_reasoning_planning.mirror7_phase34 import GoalPlanner, TransitionModel
from phase343_semantic_state import SemanticStateInducer
from phase348_semantic_planning_bridge import build_planning_request
from phase350_semantic_operation_planning import (
    execute_operation_aware_planning,
    planning_policy,
)


def make_model():
    model = TransitionModel()
    for x in range(4):
        model.observe({"x": x}, "inc", {"x": x + 1})
    for x in range(4):
        model.observe({"x": x}, "noop", {"x": x})
    return model


def test_operation_is_exposed_as_policy():
    request = build_planning_request(
        SemanticStateInducer().discover("Calculate result"),
        {"x": 0},
        {"x": 2},
    )
    policy = planning_policy(request)
    assert policy.operation == "calculate"
    assert policy.preferred_actions == ("calculate", "compute")


def test_operation_does_not_invent_actions():
    request = build_planning_request(
        SemanticStateInducer().discover("Calculate result"),
        {"x": 0},
        {"x": 2},
    )
    plan = execute_operation_aware_planning(request, GoalPlanner(make_model()))
    assert plan is not None
    assert set(plan.actions) <= {"inc"}


def test_operation_aware_planning_preserves_world_state():
    request = build_planning_request(
        SemanticStateInducer().discover("Calculate result"),
        {"x": 0},
        {"x": 1},
    )
    before = dict(request.world_state)
    plan = execute_operation_aware_planning(request, GoalPlanner(make_model()))
    assert plan is not None
    assert request.world_state == before == {"x": 0}


def test_unknown_operation_has_no_action_hint():
    request = build_planning_request(
        SemanticStateInducer().discover("Tell me result"),
        {"x": 0},
        {"x": 1},
    )
    assert planning_policy(request).operation is None
    assert planning_policy(request).preferred_actions == ()


def test_explicit_step_constraint_still_applies():
    request = build_planning_request(
        SemanticStateInducer().discover("Calculate result under 1 step"),
        {"x": 0},
        {"x": 2},
    )
    assert execute_operation_aware_planning(request, GoalPlanner(make_model())) is None


def test_planner_actions_remain_model_defined():
    model = make_model()
    request = build_planning_request(
        SemanticStateInducer().discover("Calculate result"),
        {"x": 0},
        {"x": 1},
    )
    planner = GoalPlanner(model)
    before = model.actions()
    plan = execute_operation_aware_planning(request, planner)
    assert plan is not None
    assert model.actions() == before
    assert set(plan.actions) <= set(before)
