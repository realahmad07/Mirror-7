from .test_phase60 import *

def main():
    tests=[
        test_progressive_long_horizon_3x3,
        test_heldout_state_generalization,
        test_heldout_long_horizon_recombination,
        test_discrepancy_and_online_correction,
        test_conflict_abstention,
        test_unknown_action_fails_closed,
        test_phase58_59_integration,
        test_malformed,
    ]
    for t in tests:
        t()
        print('PASS',t.__name__)
    print('PHASE 60 ACCEPTANCE GATE: PASS')
    print('progressive: 3 horizons × 3 seeds')
    print('held-out: 2/2')
    print('adversarial: 2/2')
    print('integration: Phase 58 + Phase 59 + Phase 60')
    print('regression: 8/8')

if __name__=='__main__':
    main()
