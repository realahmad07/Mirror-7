from .mirror7_phase106 import ResourceGovernor

def test_memory_never_exceeds_bound():
    g=ResourceGovernor(memory_limit=2,step_limit=10)
    for i in range(5): g.remember(str(i),i,priority=i)
    assert len(g.memory)<=2 and "4" in g.memory

def test_low_priority_is_evicted_first():
    g=ResourceGovernor(memory_limit=2); g.remember("a",1,0); g.remember("b",2,1); g.remember("c",3,2)
    assert "a" not in g.memory

def test_step_budget_is_hard():
    g=ResourceGovernor(step_limit=1); assert g.tick(); assert not g.tick()

def test_negative_tick_rejected():
    g=ResourceGovernor(); assert not g.tick(-1)
