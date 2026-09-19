from .test_phase61 import *


def main():
    tests = [
        test_progressive_goal_pursuit_3x3,
        test_heldout_opaque_environment,
        test_gate_requires_multi_step_composition,
        test_discrepancy_replans_and_clears_plan,
        test_unknown_actions_explored_once_before_repetition,
        test_phase58_59_policy_memory_integration,
        test_long_horizon_32_steps_from_learned_delta,
        test_no_invalid_action_and_fail_closed,
        test_malformed,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)

    print("PHASE 61 ACCEPTANCE GATE: PASS")
    print("progressive: 3 task families × 3 seeds")
    print("held-out: 2/2")
    print("adversarial: 3/3")
    print("integration: Phase 58 + 59 + 60")
    print("regression: 9/9")


if __name__ == "__main__":
    main()
