from phase74_representation_revision import RepresentationRevision

def test_initializes():
    r=RepresentationRevision(); x=r.observe((1,1,1)); assert x.representation

def test_stable_does_not_churn():
    r=RepresentationRevision(min_support=2)
    r.observe((1,1,1)); a=r.observe((1.1,1.1,1.1)); assert not a.changed

def test_revision_after_regime_change():
    r=RepresentationRevision(min_support=2,drift_threshold=.2)
    r.observe((1,1,1)); r.observe((8,8,8)); x=r.observe((8,8,8)); assert x.changed and x.regime==1

def test_bounded_feature_memory():
    r=RepresentationRevision(max_features=3,min_support=1,drift_threshold=0)
    for i in range(10): r.observe((i,i,i))
    assert len(r.features)<=3

def test_empty_fails_closed():
    r=RepresentationRevision(); assert r.observe(()).representation==()
