from .test_phase58 import *

def main():
    tests=[
        test_progressive_3x3,
        test_heldout_new_combination_of_known_composites,
        test_hierarchy_depth_not_memorization,
        test_negative_unrelated_no_hierarchy,
        test_heldout_distractor_does_not_change_vocabulary,
        test_determinism,
        test_malformed,
        test_compression_selects_short_atoms_first,
    ]
    for t in tests:
        t()
        print('PASS',t.__name__)
    print('PHASE 58 ACCEPTANCE GATE: PASS')
    print('progressive: 3 task families × 3 seeds')
    print('held-out: 2/2')
    print('adversarial: 2/2')
    print('regression: 8/8')

if __name__=='__main__':
    main()
