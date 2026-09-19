import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple

class ExhaustedBudgetError(Exception):
    pass

class ComputeBudget:
    def __init__(self, max_steps: int):
        self.max_steps = max_steps
        self.current_step = 0
        
    def step(self):
        if self.current_step >= self.max_steps:
            raise ExhaustedBudgetError("Compute budget exhausted.")
        self.current_step += 1

    def remaining(self) -> int:
        return max(0, self.max_steps - self.current_step)

class BoundedMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items: List[Dict[str, Any]] = []

    def store(self, item: Dict[str, Any]):
        self.items.append(item)
        if len(self.items) > self.capacity:
            # Deterministic eviction: oldest first
            self.items.pop(0)

    def retrieve_all(self) -> List[Dict[str, Any]]:
        return list(self.items)

    def count(self) -> int:
        return len(self.items)

class ScalingAgent:
    def __init__(self, compute_budget: int, memory_capacity: int):
        self.budget = ComputeBudget(compute_budget)
        self.memory = BoundedMemory(memory_capacity)
        self.state = "INIT"

    def reason(self, problem_complexity: int) -> bool:
        """
        Simulates a reasoning task that requires `problem_complexity` steps.
        Returns True if solved, False if budget is exhausted.
        """
        try:
            for _ in range(problem_complexity):
                self.budget.step()
                # Store intermediate thought in memory
                self.memory.store({"step": self.budget.current_step, "thought": "reasoning"})
            return True
        except ExhaustedBudgetError:
            return False

def run_scaling_study(complexities: List[int], budgets: List[int]) -> Dict[int, float]:
    """
    Returns success rate for each budget across the given problem complexities.
    """
    results = {}
    for budget in budgets:
        successes = 0
        for comp in complexities:
            agent = ScalingAgent(compute_budget=budget, memory_capacity=100)
            if agent.reason(comp):
                successes += 1
        results[budget] = successes / len(complexities)
    return results
