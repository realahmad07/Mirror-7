from .mirror7_phase132 import WorldMemoryBridge
def test_fact_consistency():
 b=WorldMemoryBridge(); b.remember_fact("x",1); assert b.consistency("x",1) and not b.consistency("x",2)
def test_unknown_is_unknown(): assert WorldMemoryBridge().consistency("x",1) is None
def test_rule_confidence_filter():
 b=WorldMemoryBridge(); b.absorb_rule({},"a",{},.4); b.absorb_rule({},"b",{},.9); assert len(b.supported_rules(.8))==1
