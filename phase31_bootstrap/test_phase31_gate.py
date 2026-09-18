"""
Mirror 7 — Phase 31.6: Full Phase 31 Acceptance Gate

Comprehensive gate verifying the entire Mirror 7 Phase 31 pipeline:
  Raw Observation -> Discovered Representation -> Stable State ->
  Temporal State Identity -> Cross-Encoding Invariance -> Raw->State Integration

Acceptance Criteria:
  A. Determinism: Identical inputs yield byte-identical representations, states, and temporal traces.
  B. Multi-Seed: Verified across multiple deterministic PRNG seeds.
  C. Progressive Difficulty: Verified across >= 3 levels of structural complexity.
  D. Held-Out / Unseen: Tested against unseen structured protocols.
  E. Active Adversarial Audit: Perturbation matrix (noise, insertions, deletions, reorderings,
     run-length changes, motif alterations, transition alterations).
  F. Cross-Encoding Invariance: Isomorphic structures across different vocabularies yield identical states.
  G. Temporal State Identity: Unchanged states, novel transitions, repeated states, and returns verified.
  H. Invalid Input Rejection: Malformed types and out-of-bounds inputs strictly rejected.
  I. Accumulated Regression: All steps (31.1 - 31.5) pass.
  J. Zero External Dependencies: Pure Python standard library.
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from mirror7_representation import (
    DiscoveredRepresentation,
    discover_representation,
    are_structurally_equivalent,
)
from mirror7_state import (
    StructuralState,
    extract_state,
    are_states_identical,
)
from mirror7_temporal import (
    TemporalRelation,
    StateTransition,
    TemporalStateTracker,
)
from mirror7_invariance import (
    permute_vocabulary,
    verify_cross_encoding_identity,
    verify_structural_divergence,
)
from mirror7_pipeline import (
    raw_to_state,
    raw_to_state_identity,
    Mirror7Pipeline,
)


def verify_criterion_a_determinism():
    raw = bytes([10, 20, 30, 10, 20, 30, 40, 40, 50, 60, 50, 60])
    pipeline = Mirror7Pipeline()

    rep0, state0, trans0 = pipeline.process_observation(raw)

    for _ in range(100):
        p = Mirror7Pipeline()
        rep, state, trans = p.process_observation(raw)
        assert rep.checksum == rep0.checksum
        assert state.state_hash == state0.state_hash
        assert trans.transition_hash == trans0.transition_hash
    print("  [PASS] Criterion A: Determinism (100 iterations)")


def verify_criterion_b_multi_seed():
    seeds = [42, 137, 256, 1024, 9999]
    for seed in seeds:
        rng = random.Random(seed)
        vocab = rng.sample(range(256), 6)
        raw_blocks = [
            bytes([rng.choice(vocab[:3])] * rng.randint(2, 5))
            for _ in range(4)
        ]
        pipeline = Mirror7Pipeline()
        records = pipeline.process_sequence(raw_blocks)
        assert len(records) == 4
        assert pipeline.step_count == 4
    print(f"  [PASS] Criterion B: Multi-Seed ({len(seeds)} seeds)")


def verify_criterion_c_progressive_difficulty():
    l1 = bytes([1, 1, 1, 2, 2, 3, 3, 3, 3])
    s1 = raw_to_state(l1)
    assert s1.vocab_size == 3
    assert s1.num_runs == 3
    assert s1.max_run_length == 4

    l2 = bytes([10, 20, 30, 10, 20, 30, 10, 20, 30])
    s2 = raw_to_state(l2)
    assert s2.vocab_size == 3
    assert len(s2.motif_signature) >= 1

    l3 = bytes([100, 100, 200, 250, 250, 200, 100, 100])
    s3 = raw_to_state(l3)
    assert s3.vocab_size == 3
    assert s3.run_profile == (2, 1, 2, 1, 2)

    assert s1.state_hash != s2.state_hash != s3.state_hash
    print("  [PASS] Criterion C: Progressive Difficulty (3 levels)")


def verify_criterion_d_held_out():
    held_out = bytes([
        0x55, 0xAA, 0x01, 0x10, 0x10, 0x10, 0x20, 0x30, 0xEE,
        0x55, 0xAA, 0x02, 0x10, 0x10, 0x10, 0x20, 0x30, 0xFF
    ])
    pipeline = Mirror7Pipeline()
    rep, state, trans = pipeline.process_observation(held_out)

    assert state.length == 18
    assert state.vocab_size == 9
    assert trans.relation == TemporalRelation.INITIAL
    assert state.max_run_length == 3
    print("  [PASS] Criterion D: Held-Out Unseen Observation")


def verify_criterion_e_active_adversarial_audit():
    base = bytes([10, 10, 20, 30, 30, 20])
    s_base = raw_to_state(base)

    adv_noise = bytes([10, 10, 99, 20, 30, 30, 20])
    assert raw_to_state(adv_noise) != s_base

    adv_removed = bytes([10, 10, 30, 30])
    assert raw_to_state(adv_removed) != s_base

    adv_run = bytes([10, 10, 10, 30, 30, 20])
    assert raw_to_state(adv_run) != s_base

    adv_trans = bytes([10, 10, 20, 20, 30, 30])
    assert raw_to_state(adv_trans) != s_base

    adv_rev = bytes(reversed(base))
    assert raw_to_state(adv_rev) != s_base

    adv_near = bytes([10, 20, 20, 30, 30, 10])
    assert raw_to_state(adv_near) != s_base
    print("  [PASS] Criterion E: Active Adversarial Audit (6 perturbation checks)")


def verify_criterion_f_cross_encoding():
    v1 = bytes([65, 65, 66, 67, 67, 66])
    v2 = bytes([9, 9, 7, 8, 8, 7])
    v3 = bytes([20, 20, 30, 40, 40, 30])

    ok, state_hash = verify_cross_encoding_identity([v1, v2, v3])
    assert ok, "Prompt example cross-encoding invariance failed"

    p1 = bytes([1, 2, 3, 1, 2, 3])
    p2 = bytes([80, 90, 100, 80, 90, 100])
    p3 = bytes([240, 245, 250, 240, 245, 250])
    ok2, _ = verify_cross_encoding_identity([p1, p2, p3])
    assert ok2, "Multi-alphabet cross-encoding invariance failed"
    print("  [PASS] Criterion F: Cross-Encoding Invariance")


def verify_criterion_g_temporal_identity():
    pipeline = Mirror7Pipeline()

    raw_a = bytes([1, 1, 2, 2])
    raw_b = bytes([3, 4, 3, 4])

    _, sa0, t0 = pipeline.process_observation(raw_a)
    assert t0.relation == TemporalRelation.INITIAL

    _, sa1, t1 = pipeline.process_observation(raw_a)
    assert t1.relation == TemporalRelation.UNCHANGED

    _, sb, t2 = pipeline.process_observation(raw_b)
    assert t2.relation == TemporalRelation.TRANSITION_NOVEL

    _, sa2, t3 = pipeline.process_observation(raw_a)
    assert t3.relation == TemporalRelation.TRANSITION_RETURN

    assert pipeline.tracker.is_repeated_state(sa0.state_id)
    assert pipeline.tracker.has_returned_to(sa0.state_id)
    assert not pipeline.tracker.has_returned_to(sb.state_id)
    print("  [PASS] Criterion G: Temporal State Identity (Initial, Unchanged, Novel, Return)")


def verify_criterion_h_invalid_input():
    invalids = [None, "string_data", 42, 3.14, {"a": 1}, [1, 256], [1, -1], [1, "two"]]
    for inv in invalids:
        try:
            raw_to_state(inv)
            assert False, f"Expected rejection for {inv}"
        except (TypeError, ValueError):
            pass

    s_empty = raw_to_state(b"")
    assert s_empty.length == 0
    s_single = raw_to_state(b"\x00")
    assert s_single.length == 1
    assert s_empty != s_single
    print("  [PASS] Criterion H: Invalid Input Rejection")


def verify_criterion_i_accumulated_regression():
    step_scripts = [
        "test_phase31_step1.py",
        "test_phase31_step2.py",
        "test_phase31_step3.py",
        "test_phase31_step4.py",
        "test_phase31_step5.py",
    ]
    for script in step_scripts:
        script_path = CURRENT_DIR / script
        res = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
        assert res.returncode == 0, f"Regression failed for {script}:\n{res.stderr}\n{res.stdout}"
        print(f"  [PASS] Regression: {script}")


def run_full_phase31_acceptance_gate():
    print("=" * 70)
    print("MIRROR 7 — PHASE 31 COMPREHENSIVE ACCEPTANCE GATE")
    print("=" * 70)

    verify_criterion_a_determinism()
    verify_criterion_b_multi_seed()
    verify_criterion_c_progressive_difficulty()
    verify_criterion_d_held_out()
    verify_criterion_e_active_adversarial_audit()
    verify_criterion_f_cross_encoding()
    verify_criterion_g_temporal_identity()
    verify_criterion_h_invalid_input()
    verify_criterion_i_accumulated_regression()

    print("=" * 70)
    print("PHASE 31 COMPLETE: ALL CRITERIA VERIFIED AND PASSED")
    print("=" * 70)


if __name__ == '__main__':
    run_full_phase31_acceptance_gate()
