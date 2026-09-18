"""
Test suite for Phase 31.3: Temporal State Identity
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mirror7_representation import discover_representation
from mirror7_state import StructuralState, extract_state
from mirror7_temporal import (
    TemporalRelation,
    StateTransition,
    TemporalStateTracker,
)


def _state_from_bytes(b: bytes) -> StructuralState:
    return extract_state(discover_representation(b))


def test_repeated_and_unchanged_state():
    tracker = TemporalStateTracker()
    s_a = _state_from_bytes(b"AABBCC")

    t0 = tracker.step(s_a)
    assert t0.relation == TemporalRelation.INITIAL
    assert t0.from_state_id is None
    assert t0.to_state_id == s_a.state_id

    t1 = tracker.step(s_a)
    assert t1.relation == TemporalRelation.UNCHANGED
    assert t1.from_state_id == s_a.state_id
    assert t1.to_state_id == s_a.state_id

    t2 = tracker.step(s_a)
    assert t2.relation == TemporalRelation.UNCHANGED

    assert tracker.is_repeated_state(s_a.state_id)
    assert not tracker.has_returned_to(s_a.state_id)
    print("  [PASS] test_repeated_and_unchanged_state")


def test_return_to_previous_state():
    tracker = TemporalStateTracker()
    s_a = _state_from_bytes(b"AAAA")
    s_b = _state_from_bytes(b"BBBBCCCC")
    s_c = _state_from_bytes(b"DDDD")

    t0 = tracker.step(s_a)
    assert t0.relation == TemporalRelation.INITIAL

    t1 = tracker.step(s_b)
    assert t1.relation == TemporalRelation.TRANSITION_NOVEL
    assert t1.from_state_id == s_a.state_id
    assert t1.to_state_id == s_b.state_id

    t2 = tracker.step(s_a)
    assert t2.relation == TemporalRelation.TRANSITION_RETURN
    assert t2.from_state_id == s_b.state_id
    assert t2.to_state_id == s_a.state_id

    assert tracker.is_repeated_state(s_a.state_id)
    assert tracker.has_returned_to(s_a.state_id)
    assert not tracker.has_returned_to(s_b.state_id)
    print("  [PASS] test_return_to_previous_state")


def test_progressively_harder_temporal_sequences():
    seeds = [111, 222, 333]
    for seed in seeds:
        rng = random.Random(seed)

        s_a = _state_from_bytes(bytes([rng.randint(0, 50)] * 4))
        s_b = _state_from_bytes(bytes([rng.randint(51, 100)] * 8))
        t1 = TemporalStateTracker()
        for i in range(6):
            t1.step(s_a if i % 2 == 0 else s_b)
        assert t1.step_count == 6
        assert t1.has_returned_to(s_a.state_id)
        assert t1.has_returned_to(s_b.state_id)

        s_c = _state_from_bytes(bytes([10, 20, 10, 20]))
        t2 = TemporalStateTracker()
        t2.step(s_a)
        t2.step(s_a)
        t2.step(s_b)
        t2.step(s_b)
        t2.step(s_c)
        ret_trans = t2.step(s_a)
        assert ret_trans.relation == TemporalRelation.TRANSITION_RETURN
        assert t2.has_returned_to(s_a.state_id)

        states_pool = [
            _state_from_bytes(bytes([x] * (x + 2))) for x in range(5)
        ]
        t3 = TemporalStateTracker()
        for _ in range(15):
            chosen = rng.choice(states_pool)
            t3.step(chosen)
        assert t3.step_count == 15
        assert len(t3.transitions) == 15
    print("  [PASS] test_progressively_harder_temporal_sequences")


def test_held_out_temporal_sequence():
    tracker = TemporalStateTracker()
    s1 = _state_from_bytes(bytes([1, 1, 1, 1]))
    s2 = _state_from_bytes(bytes([1, 2, 1, 2]))
    s3 = _state_from_bytes(bytes([1, 2, 3, 4]))
    s4 = _state_from_bytes(bytes([1, 1, 2, 2, 3, 3]))

    sequence = [s1, s2, s3, s4, s1, s2]
    for s in sequence:
        tracker.step(s)

    assert tracker.step_count == 6
    assert tracker.transitions[4].relation == TemporalRelation.TRANSITION_RETURN
    assert tracker.transitions[4].to_state_id == s1.state_id
    assert tracker.transitions[5].relation == TemporalRelation.TRANSITION_RETURN
    assert tracker.transitions[5].to_state_id == s2.state_id
    print("  [PASS] test_held_out_temporal_sequence")


def test_adversarial_near_match_temporal():
    tracker = TemporalStateTracker()
    s1 = _state_from_bytes(bytes([1, 1, 1, 1]))
    s2 = _state_from_bytes(bytes([1, 1, 1]))

    t0 = tracker.step(s1)
    t1 = tracker.step(s2)

    assert t1.relation != TemporalRelation.UNCHANGED
    assert t1.relation == TemporalRelation.TRANSITION_NOVEL
    assert t0.to_state_id != t1.to_state_id
    print("  [PASS] test_adversarial_near_match_temporal")


def test_deterministic_replay():
    states = [
        _state_from_bytes(bytes([x, x, y]))
        for x, y in [(1, 2), (1, 2), (3, 4), (1, 2), (5, 6)]
    ]

    tracker1 = TemporalStateTracker.replay(states)
    tracker2 = TemporalStateTracker.replay(states)

    assert tracker1.temporal_signature() == tracker2.temporal_signature()
    assert len(tracker1.transitions) == len(tracker2.transitions)
    for tr1, tr2 in zip(tracker1.transitions, tracker2.transitions):
        assert tr1 == tr2
    print("  [PASS] test_deterministic_replay")


def test_invalid_input_rejection():
    tracker = TemporalStateTracker()
    invalids = [None, "STATE_123", 12345, b"bytes_not_state", [1, 2]]
    for inv in invalids:
        try:
            tracker.step(inv)
            assert False, f"Expected rejection for {inv}"
        except TypeError:
            pass
    print("  [PASS] test_invalid_input_rejection")


def run_all_step3_tests():
    print("=" * 60)
    print("PHASE 31.3 ACCEPTANCE SUITE")
    print("=" * 60)
    test_repeated_and_unchanged_state()
    test_return_to_previous_state()
    test_progressively_harder_temporal_sequences()
    test_held_out_temporal_sequence()
    test_adversarial_near_match_temporal()
    test_deterministic_replay()
    test_invalid_input_rejection()
    print("=" * 60)
    print("PHASE_31_3_TEMPORAL_STATE_IDENTITY_PASS")
    print("=" * 60)


if __name__ == '__main__':
    run_all_step3_tests()
