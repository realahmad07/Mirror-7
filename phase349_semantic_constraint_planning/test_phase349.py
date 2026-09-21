from phase34_reasoning_planning.mirror7_phase34 import GoalPlanner, TransitionModel
from phase343_semantic_state import SemanticStateInducer
from phase348_semantic_planning_bridge import build_planning_request
from phase349_semantic_constraint_planning import execute_constrained_planning, planning_constraints


def make_model():
    model = TransitionModel()
    for x in range(5):
        model.observe({"x": x}, "inc", {"x": x + 1})
    return model


def test_explicit_step_constraint_limits_planning():
    semantic = SemanticStateInducer().discover("Create result under 2 steps")
    request = build_planning_request(semantic, {"x": 0}, {"x": 3})
    assert planning_constraints(request).max_steps == 2
    assert execute_constrained_planning(request, GoalPlanner(make_model())) is None


def test_constraint_does_not_change_world_state():
    semantic = SemanticStateInducer().discover("Create result under 2 steps")
    request = build_planning_request(semantic, {"x": 0}, {"x": 1})
    before = dict(request.world_state)
    plan = execute_constrained_planning(request, GoalPlanner(make_model()))
    assert plan is not None
    assert request.world_state == before == {"x": 0}


def test_semantic_operation_does_not_invent_actions():
    semantic = SemanticStateInducer().discover("Explain variance under 3 steps")
    request = build_planning_request(semantic, {"x": 0}, {"x": 2})
    plan = execute_constrained_planning(request, GoalPlanner(make_model()))
    assert plan is not None
    assert set(plan.actions) <= {"inc"}


def test_context_carryover_preserved():
    inducer = SemanticStateInducer()
    first = inducer.discover("Explain variance")
    second = inducer.discover("Continue under 3 steps", previous=first)
    request = build_planning_request(second, {"x": 0}, {"x": 1})
    assert "carryover" in request.context["context"]
    assert "inherits_entity" in request.context["context"]


def test_unconstrained_request_keeps_planner_behavior():
    semantic = SemanticStateInducer().discover("Explain variance")
    request = build_planning_request(semantic, {"x": 0}, {"x": 3})
    plan = execute_constrained_planning(request, GoalPlanner(make_model()))
    assert plan is not None
    assert plan.actions == ("inc", "inc", "inc")
