from .mirror7_phase130 import StateFusion
def test_consensus_wins(): assert StateFusion().fuse([("a",.8),("a",.7),("b",.2)]).value=="a"
def test_empty_is_unknown(): assert StateFusion().fuse([]) is None
def test_zero_support_is_unknown(): assert StateFusion().fuse([("a",0),("b",0)]) is None
