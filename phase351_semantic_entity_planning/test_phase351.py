from phase34_reasoning_planning.mirror7_phase34 import GoalPlanner, TransitionModel
from phase343_semantic_state import SemanticStateInducer
from phase348_semantic_planning_bridge import build_planning_request
from phase351_semantic_entity_planning import (
    entity_target_is_consistent,
    execute_entity_aware_planning,
    resolve_entity_binding,
)


def make_model():
    model = TransitionModel()
    for x in range(2):
        model.observe({"x": x, "topic": "variance"}, "inc", {"x": x + 1, "topic": "variance"})
    return model


def test_entity_resolves_only_from_explicit_world_state():
    request = build_planning_request(
        SemanticStateInducer().discover("Explain variance"),
        {"x": 0, "topic": "variance"},
        {"x": 1, "topic": "variance"},
    )
    binding = resolve_entity_binding(request)
    assert binding is not None
    assert binding.entity == "variance"
    assert binding.world_key == "topic"


def test_missing_entity_is_not_fabricated():
    request = build_planning_request(
        SemanticStateInducer().discover("Explain variance"),
        {"x": 0},
        {"x": 1},
    )
    assert resolve_entity_binding(request) is None
    assert entity_target_is_consistent(request)


def test_matching_entity_and_goal_are_consistent():
    request = build_planning_request(
        SemanticStateInducer().discover("Explain variance"),
        {"x": 0, "topic": "variance"},
        {"x": 1, "topic": "variance"},
    )
    assert entity_target_is_consistent(request)


def test_contradictory_explicit_entity_goal_rejects_plan():
    request = build_planning_request(
        SemanticStateInducer().discover("Explain variance"),
        {"x": 0, "topic": "variance"},
        {"x": 1, "topic": "mean"},
    )
    assert not entity_target_is_consistent(request)
    assert execute_entity_aware_planning(request, GoalPlanner(make_model())) is None


def test_entity_resolution_does_not_modify_world_state():
    request = build_planning_request(
        SemanticStateInducer().discover("Explain variance"),
        {"x": 0, "topic": "variance"},
        {"x": 1, "topic": "variance"},
    )
    before = dict(request.world_state)
    plan = execute_entity_aware_planning(request, GoalPlanner(make_model()))
    assert plan is not None
    assert request.world_state == before


def test_context_entity_survives_continuation():
    inducer = SemanticStateInducer()
    first = inducer.discover("Explain variance")
    second = inducer.discover("Continue under 2 steps", previous=first)
    request = build_planning_request(second, {"x": 0, "topic": "variance"}, {"x": 1, "topic": "variance"})
    assert "variance" in request.context["entities"]
    assert "inherits_entity" in request.context["context"]
