"""
Test suite for Phase 31.4: Cross-Encoding Invariance
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mirror7_representation import discover_representation
from mirror7_state import StructuralState, extract_state
from mirror7_invariance import (
    permute_vocabulary,
    verify_cross_encoding_identity,
    verify_structural_divergence,
)


def test_prompt_canonical_example():
    enc1 = bytes([65, 65, 66, 67, 67, 66])
    enc2 = bytes([9, 9, 7, 8, 8, 7])
    enc3 = bytes([20, 20, 30, 40, 40, 30])

    ok, state_hash = verify_cross_encoding_identity([enc1, enc2, enc3])
    assert ok, f"Cross-encoding failure: {state_hash}"

    adv_near = bytes([65, 66, 66, 67, 67, 66])
    assert verify_structural_divergence(enc1, adv_near)
    print("  [PASS] test_prompt_canonical_example")


def test_multi_seed_vocabulary_permutations():
    seeds = [1001, 2002, 3003, 4004]
    for s in seeds:
        rng = random.Random(s)
        template = [0, 0, 1, 2, 1, 2, 2, 3]
        all_symbols = list(range(256))
        rng.shuffle(all_symbols)

        vocab_a = all_symbols[0:4]
        vocab_b = all_symbols[50:54]
        vocab_c = all_symbols[120:124]

        obs_a = bytes([vocab_a[i] for i in template])
        obs_b = bytes([vocab_b[i] for i in template])
        obs_c = bytes([vocab_c[i] for i in template])

        ok, _ = verify_cross_encoding_identity([obs_a, obs_b, obs_c])
        assert ok, f"Seed {s} permutation invariance failed"
    print(f"  [PASS] test_multi_seed_vocabulary_permutations ({len(seeds)} seeds)")


def test_progressively_harder_isomorphic_structures():
    t1 = [0]*5 + [1]*3 + [2]*4
    obs1_a = bytes([10 + i for i in t1])
    obs1_b = bytes([200 + i for i in t1])
    assert verify_cross_encoding_identity([obs1_a, obs1_b])[0]

    t2 = [0, 1, 2] * 4
    obs2_a = bytes([50 + i for i in t2])
    obs2_b = bytes([100 + i*10 for i in t2])
    assert verify_cross_encoding_identity([obs2_a, obs2_b])[0]

    t3 = [0, 0, 1, 2, 3, 3, 2, 1, 0, 0]
    obs3_a = bytes([150 + i for i in t3])
    obs3_b = bytes([10 + i*7 for i in t3])
    assert verify_cross_encoding_identity([obs3_a, obs3_b])[0]
    print("  [PASS] test_progressively_harder_isomorphic_structures")


def test_held_out_cross_encoding():
    pattern = [0, 1, 2, 2, 3, 4, 3, 4, 0, 1]
    enc_alpha = bytes([1, 2, 3, 3, 4, 5, 4, 5, 1, 2])
    enc_beta = bytes([254, 253, 200, 200, 100, 101, 100, 101, 254, 253])

    ok, h = verify_cross_encoding_identity([enc_alpha, enc_beta])
    assert ok
    print("  [PASS] test_held_out_cross_encoding")


def test_adversarial_divergence_categories():
    base = bytes([10, 10, 20, 30, 30, 20])

    alt_run = bytes([10, 20, 20, 30, 30, 20])
    assert verify_structural_divergence(base, alt_run)

    alt_trans = bytes([10, 10, 20, 20, 30, 30])
    assert verify_structural_divergence(base, alt_trans)

    m_base = bytes([1, 2, 1, 2])
    m_alt = bytes([1, 2, 2, 1])
    assert verify_structural_divergence(m_base, m_alt)

    rev = bytes(reversed(base))
    assert verify_structural_divergence(base, rev)
    print("  [PASS] test_adversarial_divergence_categories")


def test_deterministic_output():
    obs1 = bytes([5, 5, 6, 7, 7, 6])
    obs2 = bytes([50, 50, 60, 70, 70, 60])
    res = [verify_cross_encoding_identity([obs1, obs2]) for _ in range(30)]
    assert all(r[0] is True and r[1] == res[0][1] for r in res)
    print("  [PASS] test_deterministic_output")


def run_all_step4_tests():
    print("=" * 60)
    print("PHASE 31.4 ACCEPTANCE SUITE")
    print("=" * 60)
    test_prompt_canonical_example()
    test_multi_seed_vocabulary_permutations()
    test_progressively_harder_isomorphic_structures()
    test_held_out_cross_encoding()
    test_adversarial_divergence_categories()
    test_deterministic_output()
    print("=" * 60)
    print("PHASE_31_4_CROSS_ENCODING_INVARIANCE_PASS")
    print("=" * 60)


if __name__ == '__main__':
    run_all_step4_tests()
