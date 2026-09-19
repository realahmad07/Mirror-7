from .mirror7_phase302 import ImprovementHistory

def test_record_and_read_history():
    h=ImprovementHistory(); h.record("x",.2,.5,True,"v1",1); assert h.records[0].after==.5

def test_fingerprint_is_stable():
    h=ImprovementHistory(); assert h.fingerprint("x","v")==h.fingerprint("x","v")

def test_recent_is_bounded():
    h=ImprovementHistory()
    for i in range(5): h.record("x",i/10,(i+1)/10,True,str(i),i)
    assert len(h.recent("x",2))==2

def test_stagnation_detected():
    h=ImprovementHistory()
    for i in range(3): h.record("x",.5,.5,False,str(i),i)
    assert h.stagnant("x",3)

def test_success_breaks_stagnation():
    h=ImprovementHistory()
    h.record("x",.5,.5,False,"a",1); h.record("x",.5,.5,False,"b",2); h.record("x",.5,.7,True,"c",3)
    assert not h.stagnant("x",3)

def test_capabilities_are_separated():
    h=ImprovementHistory(); h.record("a",.2,.2,False,"x",1); h.record("b",.2,.5,True,"y",1)
    assert h.stagnant("a",1) and not h.stagnant("b",1)

def test_invalid_recent_limit_rejected():
    try: ImprovementHistory().recent("x",0)
    except ValueError: pass
    else: assert False

def test_empty_history_is_safe():
    h=ImprovementHistory(); assert h.records==() and not h.stagnant("x")
