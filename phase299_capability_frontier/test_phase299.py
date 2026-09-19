from .mirror7_phase299 import CapabilityFrontier, CapabilityTarget

def test_frontier_ranks_largest_weighted_gap():
    f=CapabilityFrontier([CapabilityTarget("a",.2,1,.5),CapabilityTarget("b",.4,1,1)])
    assert f.top_gap().name=="b"

def test_targets_met_are_removed_from_gap_queue():
    f=CapabilityFrontier([CapabilityTarget("a",1,1),CapabilityTarget("b",.2,1)])
    assert [x.name for x in f.ranked_gaps()]==["b"]

def test_update_changes_baseline_only():
    f=CapabilityFrontier([CapabilityTarget("a",.2,.9,2)])
    t=f.update("a",.8)
    assert t.baseline==.8 and t.target==.9 and t.priority==2

def test_three_seed_like_profiles_are_deterministic():
    for _ in (2,5,8):
        f=CapabilityFrontier([CapabilityTarget("a",.1,.9),CapabilityTarget("b",.3,.8)])
        assert f.top_gap().name=="a"

def test_score_is_clamped():
    f=CapabilityFrontier([CapabilityTarget("a",.2,.9)])
    assert f.update("a",2).baseline==1

def test_unknown_update_fails():
    try: CapabilityFrontier([CapabilityTarget("a",.2,.9)]).update("x",.5)
    except KeyError: pass
    else: assert False

def test_empty_frontier_rejected():
    try: CapabilityFrontier([])
    except ValueError: pass
    else: assert False

def test_no_gap_returns_none():
    assert CapabilityFrontier([CapabilityTarget("a",1,1)]).top_gap() is None
