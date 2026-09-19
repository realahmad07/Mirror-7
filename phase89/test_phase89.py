from .mirror7_phase89 import HiddenStateInferer

def test_discovers_unlabeled_states():
    m = HiddenStateInferer(merge_threshold=.5)
    a = m.ingest((0, 0)); b = m.ingest((10, 10))
    assert a.state_id != b.state_id and len(m.states) == 2

def test_partial_observation_identifies_state():
    m = HiddenStateInferer(merge_threshold=.5)
    m.ingest((0, 10)); m.ingest((10, 0))
    assert m.infer((0, None)).state_id == 0

def test_noisy_observation_stays_in_cluster():
    m = HiddenStateInferer(merge_threshold=1.0)
    s = m.ingest((5, 5))
    x = m.ingest((5.4, 4.8))
    assert x.state_id == s.state_id

def test_ambiguous_all_hidden_fails_closed():
    m = HiddenStateInferer()
    assert m.ingest((1, None)).state_id == 0
    assert m.infer((None, None)) is None
