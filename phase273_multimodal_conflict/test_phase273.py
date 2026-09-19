
from .mirror7_phase273 import ConflictResolver, Evidence

def test_progressive_fusion():
    r=ConflictResolver()
    assert r.resolve([Evidence("vision","red",1.0,0),Evidence("text","red",1.0,0)],0)=="red"

def test_conflict_abstention():
    r=ConflictResolver(margin=0.2)
    ev=[Evidence("vision","red",1.0,0),Evidence("audio","blue",0.9,0)]
    assert r.resolve(ev,0) is None

def test_recency_breaks_tie():
    r=ConflictResolver(margin=0.05)
    ev=[Evidence("old","red",1.0,0),Evidence("new","blue",1.0,9)]
    assert r.resolve(ev,10)=="blue"

def test_invalid_confidence_ignored():
    r=ConflictResolver()
    ev=[Evidence("bad","red",3.0,0),Evidence("good","blue",1.0,0)]
    assert r.resolve(ev,0)=="blue"

def test_held_out_modalities():
    r=ConflictResolver()
    ev=[Evidence("sensor_a",1,0.8,4),Evidence("sensor_b",1,0.6,4)]
    assert r.resolve(ev,4)==1

def test_adversarial_equal_evidence_abstains():
    r=ConflictResolver(margin=0.01)
    ev=[Evidence("a","x",0.5,0),Evidence("b","y",0.5,0)]
    assert r.resolve(ev,0) is None

def test_seed_determinism():
    for seed in (1,2,3):
        r=ConflictResolver()
        assert r.resolve([Evidence(str(seed),"ok",0.8,0)],0)=="ok"
