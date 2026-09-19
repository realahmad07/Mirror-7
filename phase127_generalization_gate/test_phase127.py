from .mirror7_phase127 import GeneralizationGate

def test_broadly_successful_skill_passes():
 r=GeneralizationGate().evaluate([.9,.85,.8]); assert r.passed

def test_one_weak_context_blocks_promotion():
 r=GeneralizationGate().evaluate([.95,.9,.2]); assert not r.passed and r.worst==.2

def test_mean_threshold_blocks_weak_overall_skill():
 r=GeneralizationGate(.9,.5).evaluate([.8,.8,.8]); assert not r.passed

def test_empty_evaluation_fails_closed():
 assert not GeneralizationGate().evaluate([]).passed