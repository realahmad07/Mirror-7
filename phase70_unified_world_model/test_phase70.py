from phase70_unified_world_model import UnifiedWorldModel

def model():
    m=UnifiedWorldModel()
    for _ in range(5):
        m.observe("inc",(0,),(1,)); m.observe("inc",(1,),(2,)); m.observe("dec",(1,),(0,))
    return m

def test_closed_loop_prediction():
    m=model(); p=m.predict("inc",(2,)); assert p and p.state==(3.0,)

def test_counterfactual():
    r=model().counterfactual((0,),("inc","inc")); assert r and r.state==(2.0,)

def test_bounded_plan():
    r=model().plan((0,),("inc","dec"),(2,)); assert r and r.state==(2.0,)

def test_unknown_fails_closed():
    m=model(); assert m.predict("unknown",(0,)) is None
    assert m.counterfactual((0,),("unknown",)) is None

def test_partial_observation():
    m=UnifiedWorldModel()
    for _ in range(4): m.observe("x",(0,None),(1,None))
    p=m.predict("x",(0,None)); assert p and p.state==(1.0,None)
