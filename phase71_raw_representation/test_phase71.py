from phase71_raw_representation import RawRepresentationLearner

def test_segment_discovery():
    r=RawRepresentationLearner(window=2,change_threshold=3,min_support=1)
    s=r.discover((0,0,0,0,10,10,10,10))
    assert len(s)>=2

def test_invented_representation():
    r=RawRepresentationLearner(window=2,change_threshold=3,min_support=1)
    code=r.encode((0,0,10,10))
    assert code and code[0][2]==2

def test_held_out_pattern_reuses_feature():
    r=RawRepresentationLearner(window=2,change_threshold=3,min_support=1)
    r.discover((0,0,10,10)); known=set(r.invented_features())
    r.discover((0,0,10,10)); assert known.issubset(set(r.invented_features()))

def test_noise_does_not_force_many_cuts():
    r=RawRepresentationLearner(window=3,change_threshold=20,min_support=1)
    s=r.discover((1,2,1,2,1,2,1,2,1)); assert len(s)<=1

def test_empty_fail_closed():
    r=RawRepresentationLearner(); assert r.discover(())==()
    assert r.fail_closed()["segments"]==0
