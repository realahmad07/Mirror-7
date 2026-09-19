from .mirror7_phase279 import CapabilityGapDetector, HypothesisGenerator

def _gap(seed):
    evidence = [False, True, True, False, True]
    shift = seed % 2
    return CapabilityGapDetector().detect("planning", 0.60, 0.90, evidence[shift:] + evidence[:shift])

def test_progressive_gap_detection():
    d = CapabilityGapDetector()
    assert d.detect("p", 0.20, 0.90, [True, False]) is not None
    assert d.detect("p", 0.60, 0.90, [True, True, False]) is not None
    assert d.detect("p", 0.89, 0.90, [True, False, True]) is not None

def test_three_seed_gap_consistency():
    gaps = [_gap(seed) for seed in (1, 7, 19)]
    assert all(g is not None for g in gaps)
    assert {round(g.gap, 6) for g in gaps} == {0.30}

def test_no_gap_when_target_already_met():
    assert CapabilityGapDetector().detect("p", 0.95, 0.90, [True, True, True]) is None

def test_no_gap_without_evidence():
    assert CapabilityGapDetector().detect("p", 0.40, 0.90, []) is None

def test_generator_returns_bounded_deterministic_candidates():
    gap = _gap(3)
    space = {"step": [1, 2, 3], "beam": [1, 2]}
    a = HypothesisGenerator().generate(gap, space, max_candidates=4)
    b = HypothesisGenerator().generate(gap, space, max_candidates=4)
    assert a == b
    assert len(a) == 4
    assert [h.ordinal for h in a] == [1, 2, 3, 4]

def test_generator_rejects_empty_space():
    assert HypothesisGenerator().generate(_gap(0), {}) == []

def test_generator_rejects_invalid_limit():
    try:
        HypothesisGenerator().generate(_gap(0), {"x": [1]}, max_candidates=0)
    except ValueError:
        pass
    else:
        assert False

def test_invalid_metrics_fail_closed():
    try:
        CapabilityGapDetector().detect("p", 1.1, 0.9, [True])
    except ValueError:
        pass
    else:
        assert False

def test_adversarial_negative_control_is_not_actionable_when_target_is_met():
    assert CapabilityGapDetector().detect("p", 0.95, 0.90, [False, False, False]) is None
