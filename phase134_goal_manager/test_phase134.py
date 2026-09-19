from .mirror7_phase134 import GoalManager
def test_priority_selection():
 g=GoalManager(); g.add("a",1); g.add("b",2); assert g.next().name=="b"
def test_progress_removes_completed():
 g=GoalManager(); g.add("a"); g.update("a",1); assert g.next() is None
def test_goal_bound():
 g=GoalManager(1); assert g.add("a") and not g.add("b")
