from .mirror7_phase88 import TemporalAbstraction

def test_variable_length_events():
    m = TemporalAbstraction(window=2, change_threshold=3)
    events = m.discover([0, 0, 10, 10, 10, 0, 0])
    assert len(events) >= 3

def test_repeated_signature_gets_shared_identity():
    m = TemporalAbstraction(window=2, change_threshold=3)
    a = m.discover([0, 0, 10, 10])
    b = m.discover([0, 0, 10, 10])
    assert any(x.event_id == y.event_id for x in a for y in b)
    assert any(x.support >= 2 for x in b)

def test_held_out_pattern():
    m = TemporalAbstraction(window=2, change_threshold=3)
    m.discover([0, 0, 8, 8])
    held = m.discover([0, 0, 12, 12, 12])
    assert held and all(e.event_id >= 0 for e in held)

def test_adversarial_nonfinite_fails_closed():
    m = TemporalAbstraction()
    try:
        m.discover([0, float("nan")])
        assert False
    except ValueError:
        pass
