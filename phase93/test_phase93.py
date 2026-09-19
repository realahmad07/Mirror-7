from .mirror7_phase93 import StructuralTransfer

def test_permutation_transfer():
    r = StructuralTransfer().match((1, 2, 3), (3, 1, 2))
    assert r.accepted and r.mapping == (1, 2, 0)

def test_affine_transfer():
    r = StructuralTransfer().match((1, 2, 3), (10, 20, 30))
    assert r.accepted and r.confidence >= .9

def test_dimension_rejection():
    assert not StructuralTransfer().match((1, 2), (1, 2, 3)).accepted

def test_large_dimension_rejection():
    s = StructuralTransfer(max_dim=3)
    assert not s.match((1, 2, 3, 4), (1, 2, 3, 4)).accepted
