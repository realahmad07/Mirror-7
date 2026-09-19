
from .mirror7_phase269 import AffordanceModel

def test_progressive_affordance_learning():
    m=AffordanceModel()
    a=("shift",(1,))
    m.observe((0,),a,(1,))
    m.observe((1,),a,(2,))
    assert m.effect((5,),a)==(6,)
    assert a in m.valid_actions((0,))

def test_held_out_parameterized_state():
    m=AffordanceModel()
    a=("inc",(1,))
    for x in (0,2,4):
        m.observe((x,),a,(x+1,))
    assert m.effect((100,),a)==(101,)

def test_failed_precondition_not_selected():
    m=AffordanceModel()
    a=("open",())
    m.observe((0,),a,(1,),False)
    m.observe((0,),a,(1,),False)
    assert a not in m.valid_actions((0,))

def test_unknown_action_abstains():
    m=AffordanceModel()
    assert m.effect((0,),("unknown",())) is None

def test_conflicting_effect_abstains():
    m=AffordanceModel()
    a=("flip",(1,))
    m.observe((0,),a,(1,))
    m.observe((2,),a,(1,))
    assert m.effect((0,),a) is None

def test_multi_seed_same_rule():
    for seed in (7,11,19):
        m=AffordanceModel(); a=("inc",(seed,))
        m.observe((0,),a,(seed,))
        assert m.effect((10,),a)==(10+seed,)
