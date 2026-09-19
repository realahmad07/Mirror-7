
from .mirror7_phase272 import TransferMemory

def test_transfer_across_renamed_actions():
    m=TransferMemory()
    m.observe("A",{"x":0},"inc",{"x":1})
    m.observe("B",{"q":10},"step",{"q":11})
    assert m.infer_mapping("A","B")=={"inc":"step"}

def test_transfer_held_out_value():
    m=TransferMemory()
    m.observe("A",{"x":0},"inc",{"x":1})
    m.observe("B",{"q":100},"step",{"q":101})
    assert m.transfer_effect("A","B","inc",{"q":5})=={"q":6.0}

def test_unknown_transfer_abstains():
    m=TransferMemory()
    m.observe("A",{"x":0},"inc",{"x":1})
    assert m.transfer_effect("A","B","inc",{"q":5}) is None

def test_negative_nonmatching_signature():
    m=TransferMemory()
    m.observe("A",{"x":0},"inc",{"x":2})
    m.observe("B",{"q":5},"step",{"q":8})
    assert m.infer_mapping("A","B")=={"inc":"step"} or m.infer_mapping("A","B")=={}

def test_seed_stability():
    for seed in (4,8,12):
        m=TransferMemory()
        m.observe("A",{"x":seed},"inc",{"x":seed+1})
        m.observe("B",{"q":seed*10},"step",{"q":seed*10+1})
        assert m.infer_mapping("A","B")["inc"]=="step"

def test_context_isolation():
    m=TransferMemory()
    m.observe("A",{"x":0},"inc",{"x":1})
    assert m.interference_safe("A",{"x":0},"inc")
    assert not m.interference_safe("B",{"x":0},"inc")
