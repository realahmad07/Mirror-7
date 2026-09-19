from phase69_transfer_abstraction import Phase69Transfer

def test_permuted_scaled_environment():
    t=Phase69Transfer()
    m=t.learn_map((1,2,0),(0,4,2))
    assert m and m.source_to_target==(2,1,0)
    assert t.transfer((1,2,0),m)==(0.0,4.0,2.0)

def test_invariant_signature_survives_scale():
    t=Phase69Transfer()
    assert t.invariant((1,2,3))==t.invariant((10,20,30))

def test_novel_environment_transfer():
    t=Phase69Transfer()
    m=t.learn_map((2,1),(6,3))
    assert m and t.transfer((2,1),m)==(6.0,3.0)

def test_bad_mapping_rejected():
    t=Phase69Transfer(min_confidence=.99)
    assert t.learn_map((1,10),(9,2)) is None

def test_dimension_mismatch_fail_closed():
    t=Phase69Transfer()
    assert t.learn_map((1,2),(1,2,3)) is None
    assert t.transfer((1,),None) is None
