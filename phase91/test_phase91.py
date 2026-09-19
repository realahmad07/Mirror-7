from .mirror7_phase91 import HypothesisRevision

def test_merge_repeated_signature():
    m = HypothesisRevision()
    a = m.observe({"x": 1}); b = m.observe({"x": 1.01})
    assert a.hypothesis_id == b.hypothesis_id and b.support == 2

def test_revision_creates_new_hypothesis():
    m = HypothesisRevision(revise_patience=2)
    m.observe({"x": 1}); m.observe({"x": 5}); h = m.observe({"x": 5})
    assert len(m.hypotheses) == 2 and h.hypothesis_id == 1

def test_abstain_on_ambiguity():
    m = HypothesisRevision(merge_threshold=.01)
    m.observe({"x": 0})
    m.hypotheses.append(type(m.hypotheses[0])(1, (("x", 2.0),), 1, 1.0))
    assert m.infer({"x": 1}, ambiguity_margin=2.0) is None

def test_contradiction_control():
    m = HypothesisRevision()
    assert m.contradiction({"x": 0}, {"x": 10})

def test_invalid_signature_fails_closed():
    m = HypothesisRevision()
    try:
        m.observe({"x": float("nan")})
        assert False
    except ValueError:
        pass
