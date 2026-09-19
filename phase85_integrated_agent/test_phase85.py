from .mirror7_phase85 import IntegratedAgent

def test_unknown_action_is_learned():
    a=IntegratedAgent(["inc"])
    r=a.step((0,), lambda _: (1,))
    assert r.action=="inc" and r.observed==(1,) and not r.failed_closed
    assert a.predict((0,),"inc")==(1,)

def test_discrepancy_revises_model():
    a=IntegratedAgent(["x"], tolerance=.1, max_revisions=2)
    a.learn((0,),"x",(1,))
    r=a.step((0,), lambda _: (2,))
    assert r.revised and r.discrepancy==1.0 and a.predict((0,),"x")==(2,)

def test_revision_budget_fails_closed():
    a=IntegratedAgent(["x"], tolerance=.1, max_revisions=0)
    a.learn((0,),"x",(1,))
    r=a.step((0,), lambda _: (2,))
    assert r.failed_closed

def test_invalid_observation_fails_closed():
    a=IntegratedAgent(["x"])
    assert a.step((0,), lambda _: ("bad",)).failed_closed
