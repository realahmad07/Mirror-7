from phase299_capability_frontier import CapabilityFrontier, CapabilityTarget
from .mirror7_phase304 import FrontierScheduler

def test_scheduler_selects_top_gap():
    f=CapabilityFrontier([CapabilityTarget("a",.2,1,1),CapabilityTarget("b",.5,1,1)])
    assert FrontierScheduler(f).next().capability=="a"

def test_seed_rotation_is_deterministic():
    f=CapabilityFrontier([CapabilityTarget("a",.2,1)])
    s=FrontierScheduler(f,seeds=(7,19))
    assert (s.next().seed,s.next().seed,s.next().seed)==(7,19,7)

def test_completed_frontier_returns_none():
    f=CapabilityFrontier([CapabilityTarget("a",1,1)])
    assert FrontierScheduler(f).next() is None

def test_round_bound_rejected():
    f=CapabilityFrontier([CapabilityTarget("a",.2,1)])
    try: FrontierScheduler(f).next(0)
    except ValueError: pass
    else: assert False

def test_empty_seed_list_rejected():
    try: FrontierScheduler(CapabilityFrontier([CapabilityTarget("a",.2,1)]),seeds=())
    except ValueError: pass
    else: assert False

def test_priority_breaks_ties():
    f=CapabilityFrontier([CapabilityTarget("a",.5,1,1),CapabilityTarget("b",.5,1,2)])
    assert FrontierScheduler(f).next().capability=="b"

def test_three_seed_like_scheduler_runs_stable():
    for _ in (2,5,8):
        f=CapabilityFrontier([CapabilityTarget("a",.2,1)])
        assert FrontierScheduler(f).next().seed==7

def test_gap_preserved_in_schedule():
    f=CapabilityFrontier([CapabilityTarget("a",.3,1)])
    assert abs(FrontierScheduler(f).next().gap-.7)<1e-9
