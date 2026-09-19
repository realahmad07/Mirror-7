from .mirror7_phase121 import AutonomousTaskLoop

def test_integrated_loop_executes_to_goal():
    a=AutonomousTaskLoop()
    a.actions.register("one",lambda _:1,postcondition=lambda x:x==1)
    a.actions.register("two",lambda _:2,postcondition=lambda x:x==2)
    r=a.run([("one",{}),("two",{})],{"ship"},lambda s:s==2)
    assert r.completed and r.steps==2

def test_unknown_action_fails_closed():
    a=AutonomousTaskLoop()
    r=a.run([("missing",{})],{"x"},lambda s:True)
    assert not r.completed

def test_budget_is_enforced():
    a=AutonomousTaskLoop(budget=1)
    a.actions.register("x",lambda _:1,postcondition=lambda x:True)
    r=a.run([("x",{}),("x",{})],set(),lambda s:True)
    assert not r.completed and "budget" in r.reason

def test_memory_records_completed_steps():
    a=AutonomousTaskLoop()
    a.actions.register("x",lambda _:1,postcondition=lambda x:True)
    r=a.run([("x",{})],{"x"},lambda s:s==1)
    assert r.completed and r.memory_items==1

def test_recovery_handles_transient_action():
    a=AutonomousTaskLoop()
    n=[0]
    def flaky(_):
        n[0]+=1
        if n[0]==1:
            raise RuntimeError("transient")
        return 7
    a.actions.register("flaky",flaky,postcondition=lambda x:x==7)
    r=a.run([("flaky",{})],{"x"},lambda s:s==7)
    assert r.completed and r.recovered==1

def test_plan_verification_rejects_unverified_action():
    a=AutonomousTaskLoop()
    a.actions.register("bad",lambda _:None,postcondition=lambda x:False)
    r=a.run([("bad",{})],{"x"},lambda s:True)
    assert not r.completed
