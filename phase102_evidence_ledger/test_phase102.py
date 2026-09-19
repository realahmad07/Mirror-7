from .mirror7_phase102 import Evidence, EvidenceLedger

def test_supported_claim_keeps_provenance():
    l=EvidenceLedger(); l.add(Evidence("sky","blue","sensor",.9));
    assert l.status("sky")=="supported" and l.best("sky").source=="sensor"

def test_conflict_is_explicit():
    l=EvidenceLedger(); l.add(Evidence("x",1,"a",.9)); l.add(Evidence("x",2,"b",.8));
    assert l.status("x")=="conflict"

def test_invalid_evidence_does_not_support():
    l=EvidenceLedger(); l.add(Evidence("x",1,"bad",.99,False));
    assert l.status("x")=="unknown" and l.best("x") is None

def test_memory_bound():
    l=EvidenceLedger(max_items=2)
    for i in range(5): l.add(Evidence(str(i),i,"s",i/10))
    assert len(l.items)<=2
