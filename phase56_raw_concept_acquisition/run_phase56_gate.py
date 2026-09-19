from .test_phase56 import (
    test_adversarial_semantic_swap_changes_relation_graph,
    test_adversarial_single_episode_noise_rejected,
    test_held_out_concept_recombination,
    test_held_out_raw_encoding_seed_change,
    test_malformed_inputs_rejected,
    test_progressive_raw_concept_acquisition,
    test_repeat_determinism,
)


def main():
    tests = [
        test_progressive_raw_concept_acquisition,
        test_held_out_concept_recombination,
        test_held_out_raw_encoding_seed_change,
        test_adversarial_single_episode_noise_rejected,
        test_adversarial_semantic_swap_changes_relation_graph,
        test_malformed_inputs_rejected,
        test_repeat_determinism,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print("PHASE 56 ACCEPTANCE GATE: PASS")
    print("progressive: 3 task families × 3 seeds")
    print("held-out: 2/2")
    print("adversarial: 3/3")
    print("regression: 7/7")


if __name__ == "__main__":
    main()
