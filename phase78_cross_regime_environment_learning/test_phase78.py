from phase78_cross_regime_environment_learning import CrossRegimeEnvironmentLearner


def test_same_structure_transfer():
    m=CrossRegimeEnvironmentLearner()
    m.fit({"a":[1,2,3],"b":[3,2,1]})
    r=m.transfer({"x":[10,20,30],"y":[30,20,10]})
    assert r.accepted and r.score >= .9 and set(r.mapping)=={"a","b"}

def test_mismatch_rejected():
    m=CrossRegimeEnvironmentLearner()
    m.fit({"a":[1,2,3],"b":[3,2,1]})
    r=m.transfer({"x":[1,1,1]})
    assert not r.accepted

def test_unseen_pattern_rejected():
    m=CrossRegimeEnvironmentLearner(min_score=.95)
    m.fit({"a":[0,1,2]})
    r=m.transfer({"x":[0,0.5,0]})
    assert not r.accepted

def test_empty_fail_closed():
    m=CrossRegimeEnvironmentLearner()
    assert not m.transfer({}).accepted
