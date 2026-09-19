import pytest
from .mirror7_phase231_240 import ComputeBudget, ExhaustedBudgetError, BoundedMemory, ScalingAgent, run_scaling_study

def test_compute_budget_halt():
    budget = ComputeBudget(5)
    for _ in range(5):
        budget.step()
    
    with pytest.raises(ExhaustedBudgetError):
        budget.step()

def test_bounded_memory_eviction():
    mem = BoundedMemory(3)
    mem.store({"id": 1})
    mem.store({"id": 2})
    mem.store({"id": 3})
    assert mem.count() == 3
    
    mem.store({"id": 4})
    assert mem.count() == 3
    items = mem.retrieve_all()
    assert items[0]["id"] == 2
    assert items[-1]["id"] == 4

def test_scaling_agent_success():
    agent = ScalingAgent(compute_budget=10, memory_capacity=5)
    assert agent.reason(5) is True
    assert agent.memory.count() == 5

def test_scaling_agent_exhaustion():
    agent = ScalingAgent(compute_budget=5, memory_capacity=10)
    assert agent.reason(10) is False

def test_scaling_study():
    complexities = [2, 5, 8, 12, 15]
    budgets = [1, 5, 10, 20]
    results = run_scaling_study(complexities, budgets)
    
    # Budget 1 can't solve any (except maybe 0, but min is 2)
    assert results[1] == 0.0
    # Budget 5 can solve complexity 2 and 5 (2/5 = 0.4)
    assert results[5] == 0.4
    # Budget 10 can solve 2, 5, 8 (3/5 = 0.6)
    assert results[10] == 0.6
    # Budget 20 can solve all (5/5 = 1.0)
    assert results[20] == 1.0

def test_deterministic_reproducibility():
    agent1 = ScalingAgent(7, 3)
    agent2 = ScalingAgent(7, 3)
    
    res1 = agent1.reason(8)
    res2 = agent2.reason(8)
    
    assert res1 is False
    assert res2 is False
    
    # Both should have exactly the same memory state (last 3 thoughts)
    assert agent1.memory.retrieve_all() == agent2.memory.retrieve_all()
    assert len(agent1.memory.retrieve_all()) == 3
    assert agent1.memory.retrieve_all()[0]["step"] == 5
    assert agent1.memory.retrieve_all()[-1]["step"] == 7
