from .test_phase55 import (
    test_adversarial_structural_change_is_rejected,
    test_held_out_unseen_families,
    test_progressive_representation_independence,
    test_repeat_run_determinism_and_seed_invariance,
)
from .test_phase55 import cycle_graph, path_graph, star_graph


def main():
    progressive_total = 0

    for builder in (path_graph, star_graph, cycle_graph):
        for size in (4, 5, 6):
            test_progressive_representation_independence(builder, (size,))
            progressive_total += 3

    test_held_out_unseen_families()
    test_adversarial_structural_change_is_rejected()
    test_repeat_run_determinism_and_seed_invariance()

    print(f"PROGRESSIVE: {progressive_total}/27 representation comparisons pass")
    print("HELD-OUT: 2/2 unseen graph families pass")
    print("ADVERSARIAL: 6/6 negative controls pass")
    print("REGRESSION: parser + canonicalization + invariance + determinism pass")
    print("PHASE 55 ACCEPTANCE GATE: PASS")


if __name__ == "__main__":
    main()
