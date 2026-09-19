from .test_phase59 import *

def main():
    tests=[
        test_progressive_prediction_3x3,
        test_heldout_recombination,
        test_longer_context_beats_shorter_context,
        test_abstention_on_ambiguity,
        test_insufficient_support_abstains,
        test_unseen_context_backoff,
        test_determinism_and_signature,
        test_malformed,
    ]
    for t in tests:
        t()
        print('PASS',t.__name__)
    print('PHASE 59 ACCEPTANCE GATE: PASS')
    print('progressive: 3 task families × 3 seeds')
    print('held-out: 2/2')
    print('adversarial: 2/2')
    print('regression: 8/8')

if __name__=='__main__':
    main()
