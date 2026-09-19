from .mirror7_phase109 import ValidatedKnowledge

def test_high_confidence_fact_is_accepted():
    k=ValidatedKnowledge(); r=k.ingest("pi","3.14","user",.95); assert r.accepted

def test_low_confidence_fact_is_not_accepted():
    k=ValidatedKnowledge(); assert not k.ingest("x",1,"u",.4).accepted

def test_conflicting_claim_is_not_overwritten():
    k=ValidatedKnowledge(); k.ingest("x",1,"a",.95); r=k.ingest("x",2,"b",.95); assert not r.accepted and r.conflict

def test_unknown_claim_returns_none():
    assert ValidatedKnowledge().query("missing") is None
