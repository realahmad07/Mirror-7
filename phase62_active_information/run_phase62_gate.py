from .test_phase62 import *


def main():
    tests = [
        test_progressive_partial_observability_3x3,
        test_heldout_opaque_hidden_configuration,
        test_active_information_selection_prefers_relevant_sensor,
        test_sensor_dropout_uses_remaining_information_path,
        test_contradictory_model_evidence_forces_replan_counter,
        test_unknown_actions_are_not_repeated_at_same_partial_state,
        test_full_state_model_learns_from_revealed_transitions,
        test_phase61_handoff_contract,
        test_malformed,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)

    print("PHASE 62 ACCEPTANCE GATE: PASS")
    print("progressive: 3 task families × 3 seeds")
    print("held-out: 2/2")
    print("adversarial: 3/3")
    print("integration: Phase 61 handoff contract")
    print("regression: 9/9")


if __name__ == "__main__":
    main()
