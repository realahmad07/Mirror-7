from .mirror7_phase119 import PlanVerifier

def test_valid_plan_reaches_goal():
 r=PlanVerifier().verify([lambda x:x+1,lambda x:x+1],0,lambda x:x==2,3); assert r.valid

def test_invalid_plan_is_rejected_before_execution_when_over_budget():
 r=PlanVerifier().verify([lambda x:x+1]*3,0,lambda x:x==3,2); assert not r.valid and r.checked==0

def test_bad_step_fails_closed():
 r=PlanVerifier().verify([lambda x:1/0],0,lambda x:True,2); assert not r.valid

def test_goal_failure_is_reported():
 r=PlanVerifier().verify([lambda x:x+1],0,lambda x:x==9,2); assert not r.valid
