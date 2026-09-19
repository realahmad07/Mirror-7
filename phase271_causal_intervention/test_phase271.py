
from .mirror7_phase271 import CausalInterventionModel

def build():
    m=CausalInterventionModel()
    for x,y,z in [(0,0,0),(1,2,10),(0,0,0),(1,2,20)]:
        m.observe({"X":x,"Y":y,"Z":z})
    return m

def test_observation_and_intervention_are_distinct():
    m=build()
    assert m.effect("X","Y",0) is None
    m.intervene({"X":0},{"Y":0})
    m.intervene({"X":1},{"Y":2})
    assert m.direct_effect_supported("X","Y",0,1)

def test_three_seeds_same_direct_effect():
    for seed in (2,7,13):
        m=CausalInterventionModel()
        m.intervene({"X":0},{"Y":0})
        m.intervene({"X":1},{"Y":seed})
        assert m.direct_effect_supported("X","Y",0,1)

def test_null_effect_rejected():
    m=CausalInterventionModel()
    m.intervene({"X":0},{"Y":4})
    m.intervene({"X":1},{"Y":4})
    assert not m.direct_effect_supported("X","Y",0,1)

def test_held_out_level_is_absent_not_guessed():
    m=CausalInterventionModel()
    m.intervene({"X":0},{"Y":0})
    m.intervene({"X":2},{"Y":4})
    assert m.effect("X","Y",1) is None

def test_invariance_gate():
    m=CausalInterventionModel()
    for x in (0,1,2):
        m.intervene({"X":x},{"Y":x+1})
    assert not m.invariant_across_contexts("X","Y",(0,1,2))

def test_missing_outcome_abstains():
    m=CausalInterventionModel()
    m.intervene({"X":1},{"Z":3})
    assert m.effect("X","Y",1) is None
