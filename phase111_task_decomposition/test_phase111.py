from .mirror7_phase111 import TaskDecomposer,TaskNode

def test_dependency_order_is_respected():
    o=TaskDecomposer().order([TaskNode("c",("b",)),TaskNode("b",("a",)),TaskNode("a")],10); assert o==("a","b","c")

def test_budget_truncates_safely():
    o=TaskDecomposer().order([TaskNode("a",cost=2),TaskNode("b",("a",),cost=2)],2); assert o==("a",)

def test_cycle_is_rejected():
    try: TaskDecomposer().order([TaskNode("a",("b",)),TaskNode("b",("a",))],5)
    except ValueError as e: assert "cycle" in str(e)
    else: assert False

def test_missing_dependency_is_rejected():
    try: TaskDecomposer().order([TaskNode("a",("x",))],5)
    except ValueError: pass
    else: assert False
