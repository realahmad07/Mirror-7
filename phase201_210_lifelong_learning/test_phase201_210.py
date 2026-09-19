from .mirror7_phase201_210 import *

def E(v,ep,c=1.0,src="s"): return Evidence("k",v,src,ep,c)

def test_201_persistence():
    p=PersistentLearner(); p.observe(E("a",1))
    assert p.recall("k")=="a"

def test_202_confidence_weighting():
    p=PersistentLearner(); p.observe(E("a",1,.2)); p.observe(E("b",2,1.0))
    assert p.recall("k")=="b"

def test_203_contradiction_visible():
    p=PersistentLearner(); p.observe(E("a",1)); p.observe(E("b",1))
    assert p.contradiction("k")

def test_204_abstain_tie():
    p=PersistentLearner(); p.observe(E("a",1)); p.observe(E("b",1))
    assert p.recall("k") is None

def test_205_consolidation():
    p=PersistentLearner(); p.observe(E("a",1)); p.observe(E("a",2))
    assert p.consolidate()["stable"]["k"]=="a"

def test_206_capacity():
    p=PersistentLearner(capacity=2)
    for i in range(5): p.observe(E(str(i),i))
    assert len(p.evidence)==2

def test_207_checkpoint_roundtrip():
    p=PersistentLearner(); p.observe(E("a",1)); cp=p.checkpoint()
    q=PersistentLearner(); q.restore(cp)
    assert q.checkpoint()==cp and q.recall("k")=="a"

def test_208_low_confidence_forgetting():
    p=PersistentLearner(); p.observe(E("a",1,.1)); p.observe(E("b",2,1))
    assert p.forget_low_confidence(.5)==1

def test_209_replay():
    p=PersistentLearner(); p.replay([E("a",1),E("a",2)])
    assert p.support("k","a")==2

def test_210_resource_bound():
    p=PersistentLearner(capacity=4)
    for i in range(20): p.observe(E(str(i),i))
    assert len(p.evidence)<=4 and len(p.evidence)<=p.capacity
