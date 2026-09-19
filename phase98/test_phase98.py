from .mirror7_phase98 import MixedViewGrounder

def test_numeric_view_identity():
    m=MixedViewGrounder(); assert m.equivalent([0,1,2],[0,1,2])

def test_grid_identity_and_support():
    m=MixedViewGrounder(); assert m.observe([[0,1],[2,3]]).support==1
    assert m.observe([[0,1],[2,3]]).support==2

def test_symbolic_identity():
    m=MixedViewGrounder(); a=m.observe("red red blue"); b=m.observe("red red blue")
    assert a.concept_id==b.concept_id and b.support==2

def test_invalid_view_fails_closed():
    m=MixedViewGrounder()
    try: m.observe([[1],[1,2]]); assert False
    except ValueError: pass
