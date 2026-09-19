
from .mirror7_phase277 import Mirror7EndToEnd

def build():
    a=Mirror7EndToEnd()
    a.learn(b"raw-A","increase","INC",(0,),(1,))
    a.learn(b"raw-B","increase","INC",(1,),(2,))
    return a

def test_raw_to_plan_pipeline():
    a=build()
    assert a.plan((0,),"increase",lambda s:s==(2,),3)==["INC","INC"]

def test_held_out_start():
    a=build()
    assert a.plan((5,),"increase",lambda s:s==(7,),3)==["INC","INC"]

def test_unknown_language_fails_closed():
    a=build()
    assert a.plan((0,),"jump",lambda s:s==(1,),2) is None

def test_conflicting_affordance_abstains():
    a=Mirror7EndToEnd()
    a.learn(b"A","increase","INC",(0,),(1,))
    a.learn(b"B","increase","INC",(0,),(2,))
    assert a.plan((0,),"increase",lambda s:s==(1,),2) is None

def test_multiple_seed_regression():
    for seed in (2,5,8):
        a=Mirror7EndToEnd()
        a.learn(bytes([seed]),"up","UP",(0,),(1,))
        a.learn(bytes([seed+1]),"up","UP",(1,),(2,))
        assert a.plan((0,),"up",lambda s:s==(2,),3)==["UP","UP"]

def test_representation_is_executed():
    a=build()
    assert a.representation.history
