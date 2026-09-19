from .mirror7_phase138 import AbstractReasoner
def test_relation_mapping(): assert AbstractReasoner().infer_relation({"a":1},{"a":2})=={"a":2}
def test_mismatch_abstains(): assert AbstractReasoner().infer_relation({"a":1},{"b":1}) is None
def test_composition(): assert AbstractReasoner().compose([lambda x:x+1,lambda x:x*2])(2)==6
