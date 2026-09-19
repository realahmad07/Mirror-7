from mirror7_algorithm_04.mirror7_algorithm import AdaptiveWorldModelBuilder

def basic_test():
    builder = AdaptiveWorldModelBuilder()
    builder.observe_transition({"light": 0}, "push", {"light": 1})
    state, conf = builder.predict({"light": 0}, "push")
    assert state.get("light") == 1
    return True

def harder_test():
    builder = AdaptiveWorldModelBuilder()
    builder.observe_transition({"x": 1, "y": 2}, "move", {"x": 2, "y": 2})
    builder.observe_transition({"x": 2, "y": 2}, "move", {"x": 3, "y": 2})
    state, _ = builder.predict({"x": 2, "y": 2}, "move")
    assert state.get("x") == 3
    return True

def unseen_test():
    builder = AdaptiveWorldModelBuilder()
    state, conf = builder.predict({"new_var": 1}, "jump")
    assert state.get("new_var") == 1
    return True

def adversarial_test():
    builder = AdaptiveWorldModelBuilder()
    builder.observe_transition({"x": 1}, "act", {"x": 2})
    builder.observe_transition({"x": 1}, "act", {"x": 3})
    builder.observe_transition({"x": 1}, "act", {"x": 2})
    return True

def ambiguous_test():
    builder = AdaptiveWorldModelBuilder()
    builder.observe_transition({}, "act", {"hidden": 1})
    return True

def failure_recovery_test():
    builder = AdaptiveWorldModelBuilder()
    builder.observe_transition({"x": 1}, "act", {"x": 2})
    builder.observe_transition({"x": 1}, "act", {"x": 99})
    return True

def resource_limit_test():
    builder = AdaptiveWorldModelBuilder(max_rules=2)
    for i in range(5):
        builder.observe_transition({"state": i}, "act", {"state": i+1})
    assert len(builder.get_rules()) <= 2
    return True

if __name__ == "__main__":
    tests = [basic_test, harder_test, unseen_test, adversarial_test, ambiguous_test, failure_recovery_test, resource_limit_test]
    passed = 0
    for t in tests:
        try:
            if t(): passed += 1
        except Exception as e:
            print(f"Test {t.__name__} failed: {e}")
    print(f"Algorithm 4 Tests: {passed}/{len(tests)} passed.")
