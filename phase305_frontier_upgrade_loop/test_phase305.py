from phase299_capability_frontier import CapabilityFrontier, CapabilityTarget
from .mirror7_phase305 import FrontierUpgradeLoop, UpgradeOutcome

class Adapter:
    def __init__(self,score=.5,next_score=.8): self.score=score; self.next_score=next_score
    def improve(self,seed,rounds,candidates):
        return UpgradeOutcome(self.next_score,True,"v1",{})

def test_loop_attacks_top_gap():
    f=CapabilityFrontier([CapabilityTarget("a",.2,1,1),CapabilityTarget("b",.5,1,1)])
    e=FrontierUpgradeLoop(f,{"a":Adapter(),"b":Adapter()})
    assert e.step()["capability"]=="a"

def test_verified_gain_updates_frontier():
    f=CapabilityFrontier([CapabilityTarget("a",.2,1)])
    e=FrontierUpgradeLoop(f,{"a":Adapter()}); e.step(); assert f.top_gap().gap==.2

def test_failed_change_does_not_update():
    f=CapabilityFrontier([CapabilityTarget("a",.2,1)])
    class Bad:
        def improve(self,*args): return UpgradeOutcome(.9,False,"bad",{})
    e=FrontierUpgradeLoop(f,{"a":Bad()}); assert not e.step()["accepted"] and f.top_gap().gap==.8

def test_protected_regression_blocks():
    f=CapabilityFrontier([CapabilityTarget("a",.2,1),CapabilityTarget("b",.8,1)])
    class Bad:
        def improve(self,*args): return UpgradeOutcome(.9,True,"v",{"b":.5})
    e=FrontierUpgradeLoop(f,{"a":Bad(),"b":Adapter()})
    r=e.step(); assert not r["accepted"] and f.top_gap().name=="a"

def test_three_seed_like_runs_are_stable():
    for _ in (2,5,8):
        f=CapabilityFrontier([CapabilityTarget("a",.2,1)])
        e=FrontierUpgradeLoop(f,{"a":Adapter()}); assert e.step()["accepted"]

def test_history_records_result():
    f=CapabilityFrontier([CapabilityTarget("a",.2,1)])
    e=FrontierUpgradeLoop(f,{"a":Adapter()}); e.step(); assert len(e.history.records)==1

def test_all_targets_met_stops():
    f=CapabilityFrontier([CapabilityTarget("a",1,1)])
    e=FrontierUpgradeLoop(f,{"a":Adapter()}); assert e.step() is None

def test_missing_adapter_rejected():
    try: FrontierUpgradeLoop(CapabilityFrontier([CapabilityTarget("a",.2,1)]),{})
    except ValueError: pass
    else: assert False

def test_step_bound_forwarded():
    f=CapabilityFrontier([CapabilityTarget("a",.2,1)])
    class Check:
        def improve(self,seed,rounds,candidates): assert (rounds,candidates)==(2,3); return UpgradeOutcome(.7,True,"x",{})
    e=FrontierUpgradeLoop(f,{"a":Check()}); e.step(2,3)

def test_run_is_bounded():
    f=CapabilityFrontier([CapabilityTarget("a",.2,1)])
    e=FrontierUpgradeLoop(f,{"a":Adapter()}); assert len(e.run(steps=2))<=2

def test_invalid_step_bound_rejected():
    f=CapabilityFrontier([CapabilityTarget("a",.2,1)])
    try: FrontierUpgradeLoop(f,{"a":Adapter()}).run(0)
    except ValueError: pass
    else: assert False
