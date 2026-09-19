from .mirror7_phase86 import HorizonPlanner, replay_plan

def test_finds_two_step_plan():
    t={((0,),"a"):(1,),((1,),"b"):(2,)}
    r=HorizonPlanner(t,["a","b"],max_depth=2).plan((0,),(2,))
    assert r.actions==("a","b") and r.predicted==(2,)

def test_unreachable_goal_returns_best_supported_prefix():
    t={((0,),"a"):(1,)}
    r=HorizonPlanner(t,["a","b"],max_depth=3).plan((0,),(5,))
    assert r.actions==("a",) and r.predicted==(1,) and not r.failed_closed

def test_node_budget_fails_closed():
    t={((0,),"a"):(1,),((0,),"b"):(2,)}
    r=HorizonPlanner(t,["a","b"],max_depth=3,max_nodes=1).plan((0,),(3,))
    assert r.failed_closed

def test_replay_rejects_unknown_transition():
    assert replay_plan((0,),("a",),{}) is None
