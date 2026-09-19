from .mirror7_phase137 import ConsolidationEngine
def test_repeated_evidence_consolidates():
 c=ConsolidationEngine(); c.observe("x",1); c.observe("x",1); assert c.rules()[0][:2]==("x","1")
def test_noise_alone_not_promoted():
 c=ConsolidationEngine(); c.observe("x",1); assert not c.rules()
def test_rule_bound():
 c=ConsolidationEngine(max_rules=1); [c.observe(str(i),i) or c.observe(str(i),i) for i in range(3)]; assert len(c.rules())==1
