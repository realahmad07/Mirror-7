from .mirror7_phase131 import ConceptSkillBridge
def test_verified_binding():
 b=ConceptSkillBridge(); assert b.bind("move","move_skill",.9,True)
def test_unverified_rejected():
 b=ConceptSkillBridge(); assert not b.bind("x","s",.9,False)
def test_low_confidence_rejected():
 b=ConceptSkillBridge(); assert not b.bind("x","s",.4,True)
