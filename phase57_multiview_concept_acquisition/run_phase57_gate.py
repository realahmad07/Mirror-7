from .test_phase57 import (
    test_progressive_multiview_acquisition_3_seeds,
    test_held_out_view_permutation_and_reencoding,
    test_held_out_cross_geometry,
    test_adversarial_noise_only_rejected,
    test_adversarial_semantic_relation_swap_changes_graph,
    test_adversarial_view_dropout_preserves_surviving_structure,
    test_malformed_inputs_rejected,
    test_repeat_determinism,
    test_scaling_guard_small_linear_growth,
    test_graph_view_order_invariance,
)


def main():
    tests = [
        test_progressive_multiview_acquisition_3_seeds,
        test_held_out_view_permutation_and_reencoding,
        test_held_out_cross_geometry,
        test_adversarial_noise_only_rejected,
        test_adversarial_semantic_relation_swap_changes_graph,
        test_adversarial_view_dropout_preserves_surviving_structure,
        test_malformed_inputs_rejected,
        test_repeat_determinism,
        test_scaling_guard_small_linear_growth,
        test_graph_view_order_invariance,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print("PHASE 57 ACCEPTANCE GATE: PASS")
    print("progressive: 3 task families × 3 seeds")
    print("held-out: 2/2")
    print("adversarial: 3/3")
    print("scaling: 1/1")
    print("regression: 10/10")


if __name__ == "__main__":
    main()
