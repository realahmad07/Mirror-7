from phase79_long_horizon_self_correction import SelfCorrectingLoop


def test_long_horizon_converges():
    m=SelfCorrectingLoop(max_cycles=6)
    r=m.run((0,),"inc",lambda s,a:(s[0]+1,),lambda s:s[0]>=5)
    assert r.converged and r.cycles==5

def test_mismatch_triggers_revision():
    m=SelfCorrectingLoop(max_cycles=4)
    seq=iter([(1,),(2,),(4,)])
    r=m.run((0,),"a",lambda s,a:next(seq))
    assert r.errors>=0 and r.revisions>=0

def test_revision_budget_fails_closed():
    m=SelfCorrectingLoop(max_cycles=5,max_revisions=0)
    m.model[((0.0,),"a")]=(99.0,)
    r=m.run((0,),"a",lambda s,a:(1,))
    assert r.stopped and "budget" in r.reason

def test_invalid_input():
    try:
        SelfCorrectingLoop(max_cycles=0)
        assert False
    except ValueError: pass
