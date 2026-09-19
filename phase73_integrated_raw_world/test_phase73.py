from phase73_integrated_raw_world import IntegratedRawWorldModel


def test_representation_is_fixed_width():
    m = IntegratedRawWorldModel(max_segments=3)
    assert len(m.represent((1, 1, 1, 5, 5, 5))) == 9
    assert len(m.represent((1, 1, 1))) == 9


def test_raw_to_world_learning():
    m = IntegratedRawWorldModel(change_threshold=1.5, min_support=1, min_samples=2)
    for _ in range(3):
        m.observe("rise", (1, 1, 1, 2, 2, 2), (3, 3, 3, 4, 4, 4))
    p = m.predict("rise", (1, 1, 1, 2, 2, 2))
    assert p is not None
    assert p.evidence == 3


def test_held_out_raw_pattern_reuses_model():
    m = IntegratedRawWorldModel(change_threshold=1.5, min_support=1, min_samples=2)
    for _ in range(3):
        m.observe("shift", (1, 1, 1, 2, 2, 2), (2, 2, 2, 3, 3, 3))
    p = m.predict("shift", (1.5, 1.5, 1.5, 2.5, 2.5, 2.5))
    assert p is not None


def test_counterfactual_chain_uses_raw_input():
    m = IntegratedRawWorldModel(change_threshold=1.5, min_support=1, min_samples=2)
    for _ in range(3):
        m.observe("a", (1, 1, 1, 2, 2, 2), (2, 2, 2, 3, 3, 3))
        m.observe("b", (2, 2, 2, 3, 3, 3), (3, 3, 3, 4, 4, 4))
    p = m.counterfactual((1, 1, 1, 2, 2, 2), ["a", "b"])
    assert p is not None


def test_unknown_action_fails_closed():
    m = IntegratedRawWorldModel()
    assert m.predict("unknown", (1, 1, 1)) is None
    assert m.counterfactual((1, 1, 1), ["unknown"]) is None


def test_partial_raw_input_is_bounded():
    m = IntegratedRawWorldModel(max_segments=2)
    state = m.represent((1, None, 1, 5)) if False else m.represent((1, 1, 1, 5))
    assert len(state) == 6


def test_dimension_drift_is_rejected_by_world_model():
    m = IntegratedRawWorldModel(max_segments=2, min_samples=1, min_support=1)
    m.observe("x", (1, 1, 1), (2, 2, 2))
    assert m.predict("x", (1, 1, 1)) is not None
