from phase67_compositional_hidden_state import Phase67Agent
from phase68_counterfactual_planning import Phase68Planner

def model():
    a=Phase67Agent()
    for _ in range(5):
        a.observe("inc",(0,),(1,))
        a.observe("inc",(1,),(2,))
        a.observe("dec",(1,),(0,))
    return a.model

def test_counterfactual_rollout():
    p=Phase68Planner(model(),max_depth=3)
    r=p.counterfactual((0,),("inc","inc"))
    assert r and r["state"]==(2.0,)

def test_bounded_plan_reaches_target():
    p=Phase68Planner(model(),max_depth=3,beam_width=4)
    r=p.plan((0,),("inc","dec"),(2,),goal_tolerance=.01)
    assert r and r.predicted_state==(2.0,) and len(r.sequence)<=3

def test_competing_sequences_are_simulated():
    p=Phase68Planner(model(),max_depth=3)
    r=p.plan((1,),("inc","dec"),(0,),goal_tolerance=.01)
    assert r and r.predicted_state==(0.0,)

def test_unknown_action_fails_closed():
    p=Phase68Planner(model(),max_depth=3)
    assert p.counterfactual((0,),("unknown",)) is None
    assert p.plan((0,),("unknown",),(1,)) is None

def test_depth_budget():
    p=Phase68Planner(model(),max_depth=1)
    r=p.plan((0,),("inc",),(3,),goal_tolerance=.01)
    assert r and r.depth==1 and r.predicted_state==(1.0,)
