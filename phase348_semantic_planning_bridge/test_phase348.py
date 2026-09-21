from phase34_reasoning_planning import GoalPlanner, TransitionModel
from phase343_semantic_state import SemanticStateInducer
from phase348_semantic_planning_bridge import (
    build_planning_request,
    execute_planning_request,
)


def make_model():
    model = TransitionModel()
    for x in range(5):
        model.observe({"x": x}, "inc", {"x": x + 1})
    return model


def test_semantic_context_and_world_state_reach_planner_separately():
    semantic = SemanticStateInducer().discover("Explain variance")
    request = build_planning_request(
        semantic,
        {"x": 0},
        {"x": 3},
    )
    assert request is not None
    assert request.context["operations"] == ("explain",)
    assert request.world_state == {"x": 0}
    assert request.goal.requirements == (("x", 3),)


def test_planner_uses_explicit_world_state_not_semantic_entities():
    semantic = SemanticStateInducer().discover("Explain variance")
    request = build_planning_request(semantic, {"x": 0}, {"x": 2})
    plan = execute_planning_request(request, GoalPlanner(make_model()))
    assert plan is not None
    assert plan.actions == ("inc", "inc")


def test_semantic_context_does_not_fabricate_world_keys():
    semantic = SemanticStateInducer().discover("Explain variance")
    request = build_planning_request(semantic, {"x": 0}, {"x": 1})
    assert "variance" not in request.world_state
    assert all(key != "variance" for key, _ in request.goal.requirements)


def test_no_semantic_state_means_no_planning_request():
    assert build_planning_request(None, {"x": 0}, {"x": 1}) is None
