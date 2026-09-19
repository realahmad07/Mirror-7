import sys
from mirror7_algorithm_10.mirror7_algorithm import AutonomousResearch, KnownFact, Question, ResearchAction

def basic_test():
    ar = AutonomousResearch()
    ar.add_question(Question("math", 0.9, 1.0))
    ar.add_question(Question("history", 0.5, 1.0))
    
    # Epsilon is 0.2, but to avoid flakiness in tests, we can just check if suggestion works
    action = ar.suggest_action()
    assert action is not None
    assert action.target_topic in ["math", "history"]
    print("basic_test: PASS")

def harder_test():
    ar = AutonomousResearch()
    # Math depends on numbers
    ar.add_question(Question("math", 0.9, 1.0, ["numbers"]))
    ar.add_question(Question("numbers", 0.8, 1.0))
    
    ar.epsilon_0 = 0.0 # Force exploitation
    action = ar.suggest_action()
    assert action.target_topic == "numbers"
    print("harder_test: PASS")

def unseen_test():
    ar = AutonomousResearch()
    ar.add_fact(KnownFact("novel_topic", "data", 0.1, 0, "sensor"))
    ar.epsilon_0 = 0.0
    action = ar.suggest_action()
    assert action.target_topic == "novel_topic"
    assert action.action_type == "verify"
    print("unseen_test: PASS")

def adversarial_test():
    ar = AutonomousResearch()
    ar.add_question(Question("topic1", 0.9, 1.0))
    action = ar.suggest_action()
    
    # Report failure repeatedly
    for _ in range(5):
        ar.report_result(action, None)
    
    uncertainty = ar.get_uncertainty_map()["topic1"]
    assert uncertainty == 1.0
    print("adversarial_test: PASS")

def ambiguous_test():
    ar = AutonomousResearch()
    ar.add_question(Question("q1", 0.9, 1.0))
    ar.add_question(Question("q2", 0.9, 10.0))
    ar.epsilon_0 = 0.0
    action = ar.suggest_action()
    # Should pick cheaper one
    assert action.target_topic == "q1"
    print("ambiguous_test: PASS")

def failure_recovery_test():
    ar = AutonomousResearch()
    ar.add_question(Question("q1", 0.9, 1.0))
    action = ar.suggest_action()
    ar.report_result(action, None)
    assert ar.get_uncertainty_map()["q1"] > 0
    print("failure_recovery_test: PASS")

def resource_limit_test():
    ar = AutonomousResearch(max_questions=10)
    for i in range(50):
        ar.add_question(Question(f"q{i}", i/100.0, 1.0))
    assert len(ar.open_questions) <= 10
    print("resource_limit_test: PASS")

if __name__ == "__main__":
    try:
        basic_test()
        harder_test()
        unseen_test()
        adversarial_test()
        ambiguous_test()
        failure_recovery_test()
        resource_limit_test()
        print("ALL TESTS PASSED")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)
