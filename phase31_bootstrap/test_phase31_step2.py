"""
Test suite for Phase 31.2: Representation -> Stable State
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mirror7_representation import discover_representation
from mirror7_state import StructuralState, extract_state, are_states_identical


def test_group_a_basic_state():
    seeds = [11, 22, 33]
    for s in seeds:
        rng = random.Random(s)
        sym = rng.randint(0, 255)
        length = rng.randint(5, 15)
        raw = bytes([sym] * length)

        rep = discover_representation(raw)
        state = extract_state(rep)

        assert state.length == length
        assert state.vocab_size == 1
        assert state.num_runs == 1
        assert state.max_run_length == length
        assert state.run_profile == (length,)
        assert state.state_id.startswith("STATE_")
    print("  [PASS] test_group_a_basic_state")


def test_group_b_progressively_harder():
    seeds = [101, 102, 103]
    for s in seeds:
        rng = random.Random(s)
        a, b = rng.sample(range(256), 2)
        l1 = bytes([a, b] * 4)
        s1 = extract_state(discover_representation(l1))
        assert s1.vocab_size == 2
        assert s1.num_runs == 8

        c = rng.choice([x for x in range(256) if x not in (a, b)])
        l2 = bytes([a, a, b, c, c, c])
        s2 = extract_state(discover_representation(l2))
        assert s2.vocab_size == 3
        assert s2.num_runs == 3
        assert s2.max_run_length == 3

        l3 = bytes([a, b, c, a, b, c, a, b, c])
        s3 = extract_state(discover_representation(l3))
        assert s3.vocab_size == 3
        assert len(s3.motif_signature) > 0

        assert s1.state_hash != s2.state_hash
        assert s2.state_hash != s3.state_hash
    print("  [PASS] test_group_b_progressively_harder")


def test_group_c_complex_repeated():
    seeds = [201, 202, 203]
    for s in seeds:
        rng = random.Random(s)
        alphabet = rng.sample(range(256), 4)
        motif = [alphabet[0], alphabet[1], alphabet[2], alphabet[3]]
        noise = alphabet[0]
        raw = []
        for _ in range(3):
            raw.extend(motif)
            raw.append(noise)
        raw_bytes = bytes(raw)

        rep = discover_representation(raw_bytes)
        state = extract_state(rep)

        assert state.length == len(raw_bytes)
        assert state.vocab_size == 4
        assert len(state.motif_signature) >= 1
    print("  [PASS] test_group_c_complex_repeated")


def test_held_out_structure():
    packet1 = [0xAA, 0x01, 0x10, 0x20]
    packet2 = [0xAA, 0x02, 0x10, 0x20]
    raw = bytes(packet1 + packet2)

    rep = discover_representation(raw)
    state = extract_state(rep)

    assert state.length == 8
    assert state.vocab_size == 5
    assert state.state_id.startswith("STATE_")
    print("  [PASS] test_held_out_structure")


def test_cross_vocabulary_state_invariance():
    v1 = bytes([65, 65, 66, 67, 67, 66])
    v2 = bytes([10, 10, 20, 30, 30, 20])
    v3 = bytes([250, 250, 1, 99, 99, 1])

    s1 = extract_state(discover_representation(v1))
    s2 = extract_state(discover_representation(v2))
    s3 = extract_state(discover_representation(v3))

    assert s1 == s2 == s3
    assert s1.state_hash == s2.state_hash == s3.state_hash
    assert are_states_identical(s1, s2)
    assert are_states_identical(s2, s3)
    print("  [PASS] test_cross_vocabulary_state_invariance")


def test_adversarial_distinction():
    base = bytes([1, 1, 2, 3, 3, 2])
    adv1 = bytes([1, 2, 2, 3, 3, 2])
    adv2 = bytes([1, 1, 2, 2, 3, 3])

    s_base = extract_state(discover_representation(base))
    s_adv1 = extract_state(discover_representation(adv1))
    s_adv2 = extract_state(discover_representation(adv2))

    assert s_base != s_adv1
    assert s_base != s_adv2
    assert s_adv1 != s_adv2
    assert s_base.state_hash != s_adv1.state_hash
    assert s_base.state_hash != s_adv2.state_hash
    print("  [PASS] test_adversarial_distinction")


def test_empty_and_minimal_input():
    s_empty = extract_state(discover_representation(b""))
    assert s_empty.length == 0
    assert s_empty.vocab_size == 0
    assert s_empty.num_runs == 0

    s_single = extract_state(discover_representation(b"ÿ"))
    assert s_single.length == 1
    assert s_single.vocab_size == 1
    assert s_single.num_runs == 1

    assert s_empty.state_hash != s_single.state_hash
    print("  [PASS] test_empty_and_minimal_input")


def test_invalid_representation_rejection():
    invalids = [None, "string", 12345, b"raw_bytes_not_rep", [1, 2, 3], {"dict": 1}]
    for inv in invalids:
        try:
            extract_state(inv)
            assert False, f"Expected rejection for {inv}"
        except TypeError:
            pass
    print("  [PASS] test_invalid_representation_rejection")


def test_deterministic_repeated_execution():
    raw = bytes([10, 20, 10, 20, 30, 30, 40, 50, 40, 50])
    rep = discover_representation(raw)

    hashes = set()
    for _ in range(50):
        st = extract_state(rep)
        hashes.add(st.state_hash)

    assert len(hashes) == 1
    print("  [PASS] test_deterministic_repeated_execution")


def run_all_step2_tests():
    print("=" * 60)
    print("PHASE 31.2 ACCEPTANCE SUITE")
    print("=" * 60)
    test_group_a_basic_state()
    test_group_b_progressively_harder()
    test_group_c_complex_repeated()
    test_held_out_structure()
    test_cross_vocabulary_state_invariance()
    test_adversarial_distinction()
    test_empty_and_minimal_input()
    test_invalid_representation_rejection()
    test_deterministic_repeated_execution()
    print("=" * 60)
    print("PHASE_31_2_REPRESENTATION_TO_STATE_PASS")
    print("=" * 60)

if __name__ == '__main__':
    run_all_step2_tests()
