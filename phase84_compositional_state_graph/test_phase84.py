from phase84_compositional_state_graph import CompositionalStateGraph

def test_composition():
    g=CompositionalStateGraph(); g.add((0,0),"x",(1,0)); g.add((1,0),"y",(1,1)); assert g.predict((1,0),"y")== (1.0,1.0)

def test_isolated_states():
    g=CompositionalStateGraph(); g.add((0,),"a",(1,)); g.add((9,),"b",(8,)); assert g.predict((0,),"b") is None

def test_budget():
    g=CompositionalStateGraph(1); g.add((0,),"a",(1,))
    try: g.add((1,),"b",(2,)); assert False
    except RuntimeError: pass
