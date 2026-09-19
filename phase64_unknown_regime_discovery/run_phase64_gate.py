from test_phase64 import *

def main():
    tests = [
        test_progressive_unknown_regime_discovery_3x3,
        test_heldout_regime_is_invented_not_selected_from_labels,
        test_experiment_sequence_maximizes_known_disagreement,
        test_repeated_same_regime_does_not_create_spurious_clusters,
        test_irrelevant_state_noise_does_not_split_regime,
        test_active_regime_reidentifies_after_controlled_switch,
        test_experiment_budget_is_hard_bounded,
        test_phase63_handoff_contract,
        test_fail_closed_and_malformed,
    ]
    for test in tests:
        test()
        print('PASS', test.__name__)
    print('PHASE 64 ACCEPTANCE GATE: PASS')
    print('progressive: 3 regime families × 3 seeds')
    print('held-out: 2/2')
    print('adversarial: 3/3')
    print('integration: Phase 63 state/goal contract')
    print('regression: 9/9')

if __name__ == '__main__':
    main()
