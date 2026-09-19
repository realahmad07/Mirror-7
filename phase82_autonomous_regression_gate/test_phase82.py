from phase82_autonomous_regression_gate import RegressionGate

def test_green_gate():
    g=RegressionGate(); g.check("state",True); g.check("memory",True); r=g.finish(); assert r.passed and not r.failures

def test_fail_closed():
    g=RegressionGate(); g.check("state",True); g.check("invariant",False); r=g.finish(); assert not r.passed and r.failures==("invariant",)

def test_records_all_checks():
    g=RegressionGate(); [g.check(str(i),True) for i in range(5)]; assert len(g.finish().checks)==5
