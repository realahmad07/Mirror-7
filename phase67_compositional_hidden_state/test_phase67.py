from phase67_compositional_hidden_state import Phase67Agent

def train_additive(a):
    for _ in range(5):
        a.observe("move",(0,0,0),(1,2,0))
        a.observe("move",(0,0,1),(1,2,2))

def test_additive_composition():
    a=Phase67Agent(); train_additive(a)
    assert a.predict("move",(0,0,0))==(1.0,2.0,0.0)

def test_context_interaction():
    a=Phase67Agent()
    for _ in range(6):
        a.observe("mix",(0,0),(1,1))
        a.observe("mix",(1,0),(3,1))
    d=a.predict("mix",(1,0))
    assert d and d[0]>2.0

def test_held_out_composition():
    a=Phase67Agent()
    for _ in range(5):
        a.observe("act",(0,0,0),(1,1,0))
        a.observe("act",(1,0,0),(2,1,0))
        a.observe("act",(0,1,0),(1,2,0))
    assert a.predict("act",(1,1,0))[0] >= 1.0

def test_competing_hypothesis_revision():
    a=Phase67Agent(revision_window=2)
    for _ in range(4): a.observe("x",(0,0),(1,0))
    first=a.infer()
    for _ in range(4): a.observe("x",(0,0),(4,0))
    assert a.infer()!=first and len(a.hypotheses)>=2

def test_partial_state_fail_closed():
    a=Phase67Agent()
    for _ in range(4): a.observe("x",(0,None),(1,None))
    assert a.predict("x",(0,None)) is not None
    assert a.predict("x",(None,0)) is None
    assert a.fail_closed()["components"]>=1
