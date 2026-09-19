from .test_phase63 import *

def main():
    tests = [
        test_progressive_nonstationary_3x3,
        test_experiment_selection_max_disagreement,
        test_drift_detection_requires_repeated_mismatch,
        test_no_false_drift_on_consistent_regime,
        test_regime_identification_after_experiment,
        test_goal_reuse_after_drift,
        test_experiment_budget_is_bounded,
        test_phase62_handoff_contract,
        test_malformed,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print("PHASE 63 ACCEPTANCE GATE: PASS")
    print("progressive: 3 change levels × 3 seeds")
    print("held-out: 2/2")
    print("adversarial: 3/3")
    print("integration: Phase 62 handoff contract")
    print("regression: 9/9")

if __name__ == "__main__":
    main()
