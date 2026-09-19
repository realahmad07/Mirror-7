from .mirror7_phase90 import OverlappingDelayedEffects

def seed(m):
    for _ in range(3):
        m.learn("A", 1, 0, 2)
        m.learn("B", 2, 0, 3)

def test_overlapping_actions_add():
    m = OverlappingDelayedEffects()
    seed(m)
    assert m.predict((0,), [("A", 1), ("B", 2)]) == (5.0,)

def test_unseen_action_abstains():
    m = OverlappingDelayedEffects()
    seed(m)
    assert m.predict((1,), [("C", 1)]) is None

def test_held_out_dimension():
    m = OverlappingDelayedEffects()
    for _ in range(3):
        m.learn("A", 1, 1, 4)
    assert m.predict((0, 0), [("A", 1)]) == (0.0, 4.0)

def test_invalid_lag_fails_closed():
    m = OverlappingDelayedEffects()
    try:
        m.learn("A", 0, 0, 1)
        assert False
    except ValueError:
        pass
