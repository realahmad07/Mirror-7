import json
from mirror7_algorithm_07.mirror7_algorithm import SelfDebuggingAlgorithm, FailureRecord, CandidatePatch

def basic_test():
    print("--- basic_test ---")
    algo = SelfDebuggingAlgorithm(max_records=50, pattern_threshold=3)
    for i in range(10):
        rec = FailureRecord("ComponentA", "TimeoutError", {"param_x": 5}, None, None, None)
        algo.log_failure(rec)
    patterns = algo.detect_patterns()
    assert len(patterns) == 1
    assert patterns[0]['pattern_id'] == "ComponentA::TimeoutError"
    fix = algo.suggest_fix(patterns[0])
    assert fix is not None
    assert fix.patch_type == "input_filter"
    print("basic_test passed")

def harder_test():
    print("--- harder_test ---")
    algo = SelfDebuggingAlgorithm(max_records=50, pattern_threshold=3)
    for i in range(5):
        rec = FailureRecord("ComponentA", "TimeoutError", {"param_x": 5}, None, None, None)
        algo.log_failure(rec)
    for i in range(4):
        rec = FailureRecord("ComponentB", "ValueError", {"param_y": 10}, None, None, None)
        algo.log_failure(rec)
    
    patterns = algo.detect_patterns()
    assert len(patterns) == 2
    print("harder_test passed")

def unseen_test():
    print("--- unseen_test ---")
    algo = SelfDebuggingAlgorithm(max_records=50, pattern_threshold=3)
    rec = FailureRecord("ComponentC", "UnknownError", {"param_z": 1}, None, None, None)
    algo.log_failure(rec)
    patterns = algo.detect_patterns()
    assert len(patterns) == 0 # Threshold is 3
    print("unseen_test passed")

def adversarial_test():
    print("--- adversarial_test ---")
    algo = SelfDebuggingAlgorithm(max_records=50, pattern_threshold=3)
    import random
    for i in range(20):
        rec = FailureRecord(f"Comp{random.randint(1,10)}", "Err", {"p": random.randint(1,10)}, None, None, None)
        algo.log_failure(rec)
    patterns = algo.detect_patterns()
    # Likely 0 or maybe 1 if random clustered, but shouldn't hallucinate a single big pattern
    print(f"adversarial_test detected {len(patterns)} patterns")

def ambiguous_test():
    print("--- ambiguous_test ---")
    algo = SelfDebuggingAlgorithm(max_records=50, pattern_threshold=3)
    for i in range(5):
        # Two features perfectly correlated
        rec = FailureRecord("CompD", "Err", {"f1": "A", "f2": "B"}, None, None, None)
        algo.log_failure(rec)
    patterns = algo.detect_patterns()
    assert len(patterns) == 1
    fix = algo.suggest_fix(patterns[0])
    assert fix is not None
    print("ambiguous_test passed")

def failure_recovery_test():
    print("--- failure_recovery_test ---")
    algo = SelfDebuggingAlgorithm(max_records=50, pattern_threshold=3)
    for i in range(4):
        rec = FailureRecord("CompE", "Err", {"x": 1}, None, None, None)
        algo.log_failure(rec)
    patterns = algo.detect_patterns()
    fix = algo.suggest_fix(patterns[0])
    
    # Validation fails because it breaks a passing case
    test_cases = [
        {'context': {"x": 1}, 'is_failing_case': True},
        {'context': {"x": 1}, 'is_failing_case': False} # Passing case matches filter
    ]
    res = algo.validate_fix(fix, test_cases)
    assert not res.success
    assert fix.status == "rejected"
    print("failure_recovery_test passed")

def resource_limit_test():
    print("--- resource_limit_test ---")
    algo = SelfDebuggingAlgorithm(max_records=50, pattern_threshold=3)
    for i in range(500):
        rec = FailureRecord("CompF", "Err", {"i": i}, None, None, None)
        algo.log_failure(rec)
    assert len(algo.failure_log) == 50
    print("resource_limit_test passed")

def main():
    basic_test()
    harder_test()
    unseen_test()
    adversarial_test()
    ambiguous_test()
    failure_recovery_test()
    resource_limit_test()

if __name__ == "__main__":
    main()
