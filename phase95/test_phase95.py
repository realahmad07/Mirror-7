from .mirror7_phase95 import AssociativeMemory

def test_nearby_retrieval():
    m=AssociativeMemory(); m.add((1,2),(0,0),"a",(1,0),.8)
    r=m.retrieve((1.1,2),(0.1,0),top_k=1)
    assert r and r[0].action=="a"

def test_shape_mismatch_rejects():
    m=AssociativeMemory(); m.add((1,2),(0,0),"a",(1,0))
    assert m.retrieve((1,2,3),(0,0))==()

def test_reinforcement_is_bounded():
    m=AssociativeMemory(max_items=1); x=m.add((0,),(0,),"a",(1,),.5); m.reinforce(x)
    m.add((9,),(9,),"b",(8,),.1)
    assert len(m.items)==1 and m.items[0].action=="a"

def test_invalid_input_fails_closed():
    m=AssociativeMemory()
    try: m.add((float("nan"),),(0,),"a",(1,)); assert False
    except ValueError: pass
