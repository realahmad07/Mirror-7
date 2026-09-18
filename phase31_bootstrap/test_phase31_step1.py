"""
Test suite for Phase 31.1: Raw Observation -> Structural Representation
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mirror7_representation import (
    DiscoveredRepresentation,
    discover_representation,
    are_structurally_equivalent,
)


def test_determinism():
    raw = b'ABCABCAABBAACC'
    rep1 = discover_representation(raw)
    rep2 = discover_representation(raw)
    assert rep1 == rep2
    assert rep1.checksum == rep2.checksum
    print("  [PASS] test_determinism")


def test_cross_vocabulary_invariance():
    raw_v1 = bytes([65, 65, 66, 67, 67, 66])
    raw_v2 = bytes([9, 9, 7, 8, 8, 7])
    raw_v3 = bytes([200, 200, 10, 55, 55, 10])

    rep1 = discover_representation(raw_v1)
    rep2 = discover_representation(raw_v2)
    rep3 = discover_representation(raw_v3)

    assert rep1.canonical_tokens == (0, 0, 1, 2, 2, 1)
    assert rep2.canonical_tokens == (0, 0, 1, 2, 2, 1)
    assert rep3.canonical_tokens == (0, 0, 1, 2, 2, 1)
    assert rep1.runs == ((0, 2), (1, 1), (2, 2), (1, 1))
    assert rep1.checksum == rep2.checksum == rep3.checksum
    assert are_structurally_equivalent(rep1, rep2)
    assert are_structurally_equivalent(rep2, rep3)
    print("  [PASS] test_cross_vocabulary_invariance")


def test_progressively_harder_structures():
    l1 = bytes([10, 10, 10, 20, 20, 30])
    rep_l1 = discover_representation(l1)
    assert rep_l1.vocab_size == 3
    assert rep_l1.runs == ((0, 3), (1, 2), (2, 1))

    l2 = bytes([1, 2, 1, 2, 1, 2, 3, 4])
    rep_l2 = discover_representation(l2)
    assert rep_l2.vocab_size == 4
    motif_map = {m: c for m, c in rep_l2.motifs}
    assert motif_map.get((0, 1)) == 3
    assert motif_map.get((1, 0)) == 2

    l3 = bytes([10, 20, 30, 30, 20, 10, 10, 20, 30, 30, 20, 10])
    rep_l3 = discover_representation(l3)
    assert rep_l3.vocab_size == 3
    has_motif = any(m == (0, 1, 2, 2) and c >= 2 for m, c in rep_l3.motifs)
    assert has_motif
    print("  [PASS] test_progressively_harder_structures")


def test_multi_seed_variants():
    seeds = [42, 101, 2024, 7777]
    for s in seeds:
        rng = random.Random(s)
        alphabet = rng.sample(range(256), k=5)
        raw = []
        for _ in range(4):
            sym = rng.choice(alphabet)
            count = rng.randint(1, 4)
            raw.extend([sym] * count)
        raw_bytes = bytes(raw)
        rep_a = discover_representation(raw_bytes)
        rep_b = discover_representation(raw_bytes)
        assert rep_a.checksum == rep_b.checksum
        assert rep_a.raw_length == len(raw_bytes)
    print("  [PASS] test_multi_seed_variants")


def test_adversarial_negative_controls():
    base = bytes([1, 1, 2, 3, 3, 2])
    rep_base = discover_representation(base)

    adv1 = bytes([1, 2, 3, 3, 2])
    rep_adv1 = discover_representation(adv1)
    assert not are_structurally_equivalent(rep_base, rep_adv1)

    adv_true = bytes([1, 2, 2, 3, 3, 2])
    rep_adv_true = discover_representation(adv_true)
    assert rep_adv_true.canonical_tokens == (0, 1, 1, 2, 2, 1)
    assert not are_structurally_equivalent(rep_base, rep_adv_true)

    adv3 = bytes([2, 3, 3, 2, 1, 1])
    rep_adv3 = discover_representation(adv3)
    assert rep_adv3.canonical_tokens == (0, 1, 1, 0, 2, 2)
    assert not are_structurally_equivalent(rep_base, rep_adv3)
    print("  [PASS] test_adversarial_negative_controls")


def test_held_out_unseen():
    held_out_raw = bytes([170, 85, 170, 4, 16, 16, 32, 48, 238, 255])
    rep = discover_representation(held_out_raw)
    assert rep.raw_length == 10
    assert rep.vocab_size == 8
    assert rep.canonical_tokens[0] == 0
    assert rep.canonical_tokens[2] == 0
    assert len(rep.runs) == 9
    print("  [PASS] test_held_out_unseen")


def test_invalid_input_rejection():
    invalid_cases = [None, "plain string", 12345, {"dict": "val"}, [1, 2, "str"], [1, 2, 256], [1, -1, 5], [1.5, 2.0]]
    for case in invalid_cases:
        try:
            discover_representation(case)
            assert False, f"Expected rejection for {case}"
        except (TypeError, ValueError):
            pass

    empty_rep = discover_representation(b'')
    assert empty_rep.raw_length == 0
    assert empty_rep.vocab_size == 0
    print("  [PASS] test_invalid_input_rejection")


def run_all_step1_tests():
    print("=" * 60)
    print("PHASE 31.1 ACCEPTANCE SUITE")
    print("=" * 60)
    test_determinism()
    test_cross_vocabulary_invariance()
    test_progressively_harder_structures()
    test_multi_seed_variants()
    test_adversarial_negative_controls()
    test_held_out_unseen()
    test_invalid_input_rejection()
    print("=" * 60)
    print("PHASE_31_1_DISCOVERED_REPRESENTATION_PASS")
    print("=" * 60)

if __name__ == '__main__':
    run_all_step1_tests()
