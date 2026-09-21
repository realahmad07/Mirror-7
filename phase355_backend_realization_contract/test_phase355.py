from phase343_semantic_state import SemanticStateInducer
from phase348_semantic_planning_bridge import build_planning_request
from phase34_reasoning_planning.mirror7_phase34 import GoalPlanner, TransitionModel
from phase352_semantic_output_execution import execute_output_aware_planning
from phase353_response_realization_bridge import build_response_realization_request
from phase354_response_realization_policy import derive_response_realization_policy
from phase355_backend_realization_contract import make_backend_realization_contract

def test_contract_carries_only_verified_data():
    m=TransitionModel()
    m.observe({"x":0,"topic":"variance"},"inc",{"x":1,"topic":"variance"})
    req=build_planning_request(SemanticStateInducer().discover("Explain variance"),{"x":0,"topic":"variance"},{"x":1,"topic":"variance"})
    ex=execute_output_aware_planning(req,GoalPlanner(m))
    rr=build_response_realization_request(req,ex,"Explain variance")
    c=make_backend_realization_contract(rr,derive_response_realization_policy(rr))
    assert c.mode=="explain"
    assert c.actions==("inc",)
    assert "variance" in c.context["entities"]
    assert c.observation=="Explain variance"

def test_contract_does_not_add_actions():
    m=TransitionModel()
    m.observe({"x":0},"inc",{"x":1})
    req=build_planning_request(SemanticStateInducer().discover("Explain x"),{"x":0},{"x":1})
    ex=execute_output_aware_planning(req,GoalPlanner(m))
    rr=build_response_realization_request(req,ex,"Explain x")
    c=make_backend_realization_contract(rr,derive_response_realization_policy(rr))
    assert c.actions==("inc",)
