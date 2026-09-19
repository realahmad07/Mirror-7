import sys
from mirror7_algorithm_06.mirror7_algorithm import LongTermGoalManager, Goal

def run_tests():
    passed = 0
    failed = 0
    
    def test(name, condition):
        nonlocal passed, failed
        if condition:
            print(f"PASS: {name}")
            passed += 1
        else:
            print(f"FAIL: {name}")
            failed += 1

    # 1. basic_test
    mgr = LongTermGoalManager()
    mgr.add_goal("g1", "Goal 1", 1.0)
    test("basic_test", mgr.get_next_goal(set()) == "g1")
    mgr.update_progress("g1", 1.0, "completed")

    # 2. harder_test
    mgr.add_goal("root", "Root", 0.9)
    mgr.decompose("root", [{"id": "sub1", "description": "Sub 1"}, {"id": "sub2", "description": "Sub 2"}])
    test("harder_test", len(mgr.goals["root"].children_ids) == 2)

    # 3. unseen_test
    mgr.add_goal("urgent", "Urgent", 1.0, urgency=1.0)
    test("unseen_test", mgr.get_next_goal(set()) == "urgent")

    # 4. adversarial_test
    # Circular dependency is avoided by DAG/tree structure validation
    res = mgr.add_goal("bad", "Bad", 0.5, parent_id="nonexistent")
    test("adversarial_test", res == False)

    # 5. ambiguous_test
    mgr.add_goal("a1", "A1", 0.5, deadline=10)
    mgr.add_goal("a2", "A2", 0.5, deadline=5)
    # a2 should be prioritized over a1
    plan = mgr.get_plan(set())
    idx_a2 = plan.index("a2") if "a2" in plan else -1
    idx_a1 = plan.index("a1") if "a1" in plan else -1
    test("ambiguous_test", idx_a2 < idx_a1)

    # 6. failure_recovery_test
    mgr.add_goal("f1", "Fail", 0.8)
    for _ in range(4):
        mgr.get_next_goal(set())
        mgr.report_failure("f1", "failed")
    test("failure_recovery_test", mgr.goals["f1"].status == "failed")

    # 7. resource_limit_test
    small_mgr = LongTermGoalManager(max_depth=2, max_goals=5)
    small_mgr.add_goal("r1", "R1", 1.0)
    small_mgr.add_goal("r2", "R2", 1.0, parent_id="r1")
    res_depth = small_mgr.add_goal("r3", "R3", 1.0, parent_id="r2")
    test("resource_limit_test", not res_depth)

    print(f"Total passed: {passed}, failed: {failed}")
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
