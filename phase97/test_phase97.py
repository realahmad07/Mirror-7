from .mirror7_phase97 import UncertaintyAwarePlanner

def test_chooses_goal_path():
    t={((0,),"a"):((1,),.1),((1,),"b"):((2,),.1),((0,),"bad"):((0,),4.0)}
    p=UncertaintyAwarePlanner(t,["a","b","bad"],max_depth=2).plan((0,),(2,))
    assert p.actions==("a","b") and p.predicted==(2.0,)

def test_high_uncertainty_penalized():
    t={((0,),"a"):((1,),10.0),((0,),"b"):((1,),0.0)}
    p=UncertaintyAwarePlanner(t,["a","b"],max_depth=1).plan((0,),(1,))
    assert p.actions==("b",)

def test_budget_fails_closed():
    t={((0,),"a"):((1,),0),((0,),"b"):((2,),0)}
    p=UncertaintyAwarePlanner(t,["a","b"],node_budget=1).plan((0,),(9,))
    assert p.failed_closed

def test_invalid_state_fails_closed():
    p=UncertaintyAwarePlanner({},["a"]).plan((0,float("nan")),(1,2))
    assert p.failed_closed
