from phase34_reasoning_planning.mirror7_phase34 import GoalPlanner, TransitionModel
from phase343_semantic_state import SemanticStateInducer
from phase348_semantic_planning_bridge import build_planning_request
from phase352_semantic_output_execution import execute_output_aware_planning
from phase353_response_realization_bridge import build_response_realization_request, has_realization_intent

def model():
    m=TransitionModel()
    for x in range(2): m.observe({"x":x,"topic":"variance"},"inc",{"x":x+1,"topic":"variance"})
    return m

def test_builds_realization_request_without_changing_plan():
    req=build_planning_request(SemanticStateInducer().discover("Explain variance"),{"x":0,"topic":"variance"},{"x":1,"topic":"variance"})
    ex=execute_output_aware_planning(req,GoalPlanner(model()))
    out=build_response_realization_request(req,ex,"Explain variance")
    assert out.output.desired_output=="explanation"
    assert out.output.output_mode=="explain"
    assert out.plan_actions==("inc",)

def test_preserves_semantic_context():
    req=build_planning_request(SemanticStateInducer().discover("Explain variance"),{"x":0,"topic":"variance"},{"x":1,"topic":"variance"})
    ex=execute_output_aware_planning(req,GoalPlanner(model()))
    out=build_response_realization_request(req,ex,"Explain variance")
    assert "variance" in out.semantic_context["entities"]

def test_no_plan_means_no_actions():
    req=build_planning_request(SemanticStateInducer().discover("Explain variance"),{"x":0,"topic":"variance"},{"x":9,"topic":"variance"})
    ex=execute_output_aware_planning(req,GoalPlanner(model()))
    out=build_response_realization_request(req,ex,"Explain variance")
    assert out.plan_actions==()
    assert has_realization_intent(out)
