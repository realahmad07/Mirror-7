from phase75_raw_transfer import RawTransfer

def test_transfer_accepts_same_shape():
    r=RawTransfer(tolerance=.1); x=r.fit([(0,1,2),(1,2,3)],[(10,20,30),(20,30,40)]); assert x.accepted

def test_transfer_rejects_shape_mismatch():
    r=RawTransfer(); assert not r.fit([(0,1)],[(0,1,2)]).accepted

def test_mapping():
    r=RawTransfer(); r.fit([(0,1,2),(1,2,3)],[(10,20,30),(20,30,40)]); assert r.map((30,40,50)).accepted

def test_unseen_nonmatching_rejected():
    r=RawTransfer(tolerance=.05); r.fit([(0,1,2),(1,2,3)],[(10,20,30),(20,30,40)]); assert not r.map((0,100,0)).accepted

def test_empty_fails_closed():
    assert RawTransfer().map((1,2)) is None
