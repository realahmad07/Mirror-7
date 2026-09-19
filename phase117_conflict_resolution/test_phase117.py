from .mirror7_phase117 import ConflictResolver

def test_clear_weighted_winner_resolves():
 r=ConflictResolver().resolve([("a",.9),("b",.2)]); assert r.status=="resolved" and r.value=="a"

def test_close_conflict_abstains():
 r=ConflictResolver().resolve([("a",.6),("b",.55)],margin=.1); assert r.status=="conflict" and r.value is None

def test_no_evidence_is_unknown():
 assert ConflictResolver().resolve([]).status=="unknown"

def test_zero_weights_are_unknown():
 r=ConflictResolver().resolve([("a",0), ("b",0)]); assert r.status=="unknown" and r.value is None
