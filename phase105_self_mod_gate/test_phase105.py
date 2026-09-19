from .mirror7_phase105 import SelfImprovementGate

def test_valid_improvement_is_accepted():
    r=SelfImprovementGate(min_gain=.05,max_cost=10).evaluate(.7,.8,.75,[True,True],3)
    assert r.accepted

def test_regression_blocks_adoption():
    r=SelfImprovementGate().evaluate(.7,.9,.8,[True,False],1)
    assert not r.accepted and not r.regression_ok

def test_held_out_blocks_overfit_change():
    r=SelfImprovementGate(min_held_out=.7).evaluate(.7,.95,.6,[True],1)
    assert not r.accepted

def test_resource_limit_blocks_change():
    r=SelfImprovementGate(max_cost=2).evaluate(.7,.8,.8,[True],3)
    assert not r.accepted and not r.resource_ok
