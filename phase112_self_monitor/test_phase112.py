from .mirror7_phase112 import SelfMonitor

def test_insufficient_history_requests_caution():
    r=SelfMonitor().report(.9); assert not r.calibrated

def test_high_confidence_with_good_history_is_allowed():
    m=SelfMonitor(); [m.observe(.9,True) for _ in range(5)]; r=m.report(.9); assert r.calibrated and not r.ask_context

def test_overconfidence_triggers_context_request():
    m=SelfMonitor(); [m.observe(.9,False) for _ in range(5)]; r=m.report(.95); assert r.ask_context

def test_low_confidence_is_cautious():
    m=SelfMonitor(); [m.observe(.4,True) for _ in range(4)]; assert m.report(.4).ask_context
