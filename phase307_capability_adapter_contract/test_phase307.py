from .mirror7_phase307 import AdapterResult

def test_adapter_result_is_bounded():
    r=AdapterResult(.8,True,"v1")
    assert 0<=r.score<=1 and r.changed and r.variant

def test_adapter_result_allows_zero():
    assert AdapterResult(0,False,"none").score==0

def test_protocol_shape_is_inspectable():
    r=AdapterResult(.5,False,"v")
    assert hasattr(r,"score") and hasattr(r,"changed") and hasattr(r,"variant")

def test_three_seed_like_results_are_stable():
    for _ in (2,5,8):
        assert AdapterResult(.5,True,"v")==AdapterResult(.5,True,"v")
