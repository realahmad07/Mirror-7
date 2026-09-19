from phase83_self_model_consistency import SelfModelConsistency

def test_consistent():
    assert SelfModelConsistency(.1).evaluate((1,2),(1.05,2)).consistent

def test_inconsistent():
    r=SelfModelConsistency(.1).evaluate((1,2),(1.2,2)); assert not r.consistent and r.discrepancy==.2

def test_fail_closed():
    m=SelfModelConsistency()
    for e,a in [((1,),()),((1,),(float("nan"),))]:
        try: m.evaluate(e,a); assert False
        except ValueError: pass
