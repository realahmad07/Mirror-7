from .test_phase34 import (
    test_progressive_level_1,
    test_progressive_level_2,
    test_progressive_level_3,
    test_held_out_cases,
    test_adversarial_unsatisfiable_goal,
    test_adversarial_cycle_control,
    test_adversarial_unknown_action_rejected,
    test_exact_evidence_precedes_generalization,
)


def main():
    tests = [
        test_progressive_level_1,
        test_progressive_level_2,
        test_progressive_level_3,
        test_held_out_cases,
        test_adversarial_unsatisfiable_goal,
        test_adversarial_cycle_control,
        test_adversarial_unknown_action_rejected,
        test_exact_evidence_precedes_generalization,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("PHASE 34 ACCEPTANCE GATE: PASS")
    print("progressive: 3/3 task families × 3 seeds")
    print("held-out: 3/3")
    print("adversarial: 3/3")
    print("regression: 8/8")


if __name__ == "__main__":
    main()
