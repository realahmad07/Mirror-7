from mirror7_algorithm_03.mirror7_algorithm import CausalExperimentDesigner, CausalHypothesis

def basic_test():
    designer = CausalExperimentDesigner()
    h1 = CausalHypothesis("A->B", {"A", "B"}, [("A", "B")], {("A", "B"): 0.9}, 0.5)
    h2 = CausalHypothesis("B->A", {"A", "B"}, [("B", "A")], {("B", "A"): 0.9}, 0.5)
    designer.add_hypothesis(h1)
    designer.add_hypothesis(h2)
    designer.observe({"A": 1, "B": 1})
    assert len(designer.get_posteriors()) == 2
    return True

def harder_test():
    designer = CausalExperimentDesigner()
    for i in range(3):
        designer.add_hypothesis(CausalHypothesis(f"H{i}", {"X", "Y"}, [("X", "Y")], {("X", "Y"): 0.8}, 0.33))
    exp = designer.suggest_experiment(["X", "Y"])
    assert exp == "indistinguishable"
    return True

def unseen_test():
    designer = CausalExperimentDesigner()
    h1 = CausalHypothesis("A->B", {"A", "B"}, [("A", "B")], {("A", "B"): 0.9}, 1.0)
    designer.add_hypothesis(h1)
    designer.observe({"C": 1})
    return True

def adversarial_test():
    designer = CausalExperimentDesigner()
    h1 = CausalHypothesis("A->B", {"A", "B"}, [("A", "B")], {("A", "B"): 0.9}, 1.0)
    designer.add_hypothesis(h1)
    designer.observe({"A": 1, "B": 0})
    return True

def ambiguous_test():
    designer = CausalExperimentDesigner()
    h1 = CausalHypothesis("A->B", {"A", "B"}, [("A", "B")], {("A", "B"): 0.5}, 0.5)
    h2 = CausalHypothesis("none", {"A", "B"}, [], {}, 0.5)
    designer.add_hypothesis(h1)
    designer.add_hypothesis(h2)
    designer.observe({"A": 1, "B": 1})
    assert designer.suggest_experiment(["A", "B"]) == "indistinguishable"
    return True

def failure_recovery_test():
    designer = CausalExperimentDesigner()
    h1 = CausalHypothesis("impossible", {"A", "B"}, [("A", "B")], {("A", "B"): 1.0}, 1.0)
    designer.add_hypothesis(h1)
    designer.observe({"A": 1, "B": 0})
    return True

def resource_limit_test():
    designer = CausalExperimentDesigner(max_hypotheses=2)
    for i in range(5):
        designer.add_hypothesis(CausalHypothesis(f"H{i}", {"A", "B"}, [("A", "B")], {("A", "B"): 0.9}, 0.2))
    assert len(designer.get_posteriors()) == 2
    return True

if __name__ == "__main__":
    tests = [basic_test, harder_test, unseen_test, adversarial_test, ambiguous_test, failure_recovery_test, resource_limit_test]
    passed = 0
    for t in tests:
        try:
            if t(): passed += 1
        except Exception as e:
            print(f"Test {t.__name__} failed: {e}")
    print(f"Algorithm 3 Tests: {passed}/{len(tests)} passed.")
