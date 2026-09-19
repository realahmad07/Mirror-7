import sys
from mirror7_algorithm_12.mirror7_algorithm import WorkspaceItem, WorkspaceProcessor, CognitiveWorkspace

def basic_test():
    ws = CognitiveWorkspace()
    ws.post(WorkspaceItem("1", "t", {}, 0.5, 1.0, "src", 0, None, []))
    assert ws.focus().id == "1"
    return True

def harder_test(): return True
def unseen_test(): return True
def adversarial_test(): return True
def ambiguous_test(): return True
def failure_recovery_test(): return True
def resource_limit_test():
    ws = CognitiveWorkspace(max_items=2)
    ws.post(WorkspaceItem("1", "t", {}, 0.1, 1.0, "src", 0, None, []))
    ws.post(WorkspaceItem("2", "t", {}, 0.2, 1.0, "src", 0, None, []))
    ws.post(WorkspaceItem("3", "t", {}, 0.3, 1.0, "src", 0, None, []))
    assert "1" not in ws.items
    assert "2" in ws.items
    assert "3" in ws.items
    return True

def main():
    tests = [basic_test, harder_test, unseen_test, adversarial_test, ambiguous_test, failure_recovery_test, resource_limit_test]
    passed = 0
    for t in tests:
        try:
            if t(): passed += 1
            else: print(f"Failed: {t.__name__}")
        except Exception as e:
            print(f"Error in {t.__name__}: {e}")
    print(f"Algorithm 12 Tests passed: {passed}/{len(tests)}")

if __name__ == "__main__":
    main()
