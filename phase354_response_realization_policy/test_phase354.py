from phase343_semantic_state import SemanticStateInducer
from phase348_semantic_planning_bridge import build_planning_request
from phase34_reasoning_planning.mirror7_phase34 import GoalPlanner, TransitionModel
from phase352_semantic_output_execution import execute_output_aware_planning
from phase353_response_realization_bridge import build_response_realization_request
from phase354_response_realization_policy import derive_response_realization_policy, policy_is_non_destructive

def test_explain_policy():
    m=TransitionModel()
    m.observe({"x":0},"inc",{"x":1})
    req=build_planning_request(SemanticStateInducer().discover("Explain x"),{"x":0},{"x":1})
    ex=execute_output_aware_planning(req,GoalPlanner(m))
    p=derive_response_realization_policy(build_response_realization_request(req,ex,"Explain x"))
    assert p.mode=="explain" and p.requires_plan and p.include_context and policy_is_non_destructive(p)

def test_unknown_intent_stays_unknown():
    m=TransitionModel()
    m.observe({"x":0},"inc",{"x":1})
    req=build_planning_request(SemanticStateInducer().discover("x"),{"x":0},{"x":1})
    ex=execute_output_aware_planning(req,GoalPlanner(m))
    p=derive_response_realization_policy(build_response_realization_request(req,ex,"x"))
    assert p.mode is None and not p.requires_plan and policy_is_non_destructive(p)
