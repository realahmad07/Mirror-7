"""Automated Stage S1 tests: Typed State Frames, Partial ObservationMask,
Append-Only SQLite EpisodeLedger, Separate Parser-Only Harness, and AST Isolation.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from research.cpu_learning.evaluation.parser_harness import (
    build_parser_factories,
    evaluate_parser_on_seed,
    generate_s1_parser_suite,
)
from research.cpu_learning.evaluation.run_s1_experiments import (
    audit_ast_import_isolation,
    run_s1_experiments,
    verify_s1_committed_manifests,
)
from research.cpu_learning.frames.frame_parser import TypedRoleFrameParser
from research.cpu_learning.frames.state_frame import (
    FactEpistemicStatus,
    TypedStateFrame,
    assert_no_hidden_fields,
    extract_masked_sparse_delta,
    verify_unobserved_preservation,
)
from research.cpu_learning.ledger.episode_ledger import (
    EpisodeLedger,
    LedgerAppendError,
    LedgerFrozenError,
)
from research.cpu_learning.simulator.generators import (
    DEFAULT_DEMOS_PER_ACTION,
    DEFAULT_SEEDS,
    generate_training_episodes,
)
from research.cpu_learning.simulator.state_graph import (
    Entity,
    ObservationMask,
    StateGraph,
)


def test_s1_ast_strict_learner_isolation_zero_violations():
    audit = audit_ast_import_isolation()
    assert audit["passed"] is True, f"AST isolation violations: {audit['violations']}"
    assert audit["scanned_s1_learner_files"] >= 4
    assert audit["violations"] == []


def test_s1_hidden_field_guard_rejects_evaluator_leakage():
    with pytest.raises(ValueError):
        assert_no_hidden_fields({"expected_transition_kind": "applied"})
    with pytest.raises(ValueError):
        assert_no_hidden_fields({"nested": {"gold_after_state": {}}})
    assert_no_hidden_fields({"public_key": "ok", "split_id": "train"})


def test_s1_partial_observation_mask_unobserved_is_not_false():
    ents = (
        ("a1", Entity("a1", "agent")),
        ("o1", Entity("o1", "object", (("portable", True),))),
        ("s1", Entity("s1", "surface")),
        ("s2", Entity("s2", "surface")),
    )
    before = StateGraph(
        entities=ents,
        facts=(
            ("o1", "located_on", "s1"),
            ("o1", "owned_by", "a1"),
        ),
    )
    after = StateGraph(
        entities=ents,
        facts=(
            ("o1", "located_on", "s2"),
            ("o1", "owned_by", "a1"),
        ),
    )
    # Mask hides predicate 'owned_by' in after-state: must be UNOBSERVED, NEVER deleted!
    mask_before = ObservationMask(fully_observed=True)
    mask_after = ObservationMask(
        observed_entities=frozenset({"a1", "o1", "s1", "s2"}),
        observed_predicates=frozenset({"located_on"}),
        fully_observed=False,
    )
    f_before = TypedStateFrame.from_state_graph(before, mask_before)
    f_after = TypedStateFrame.from_state_graph(after, mask_after)

    assert f_before.query_fact_status(("o1", "owned_by", "a1")) == FactEpistemicStatus.OBSERVED_PRESENT
    assert f_after.query_fact_status(("o1", "owned_by", "a1")) == FactEpistemicStatus.UNOBSERVED

    delta_rep = extract_masked_sparse_delta(f_before, f_after)
    assert ("o1", "located_on", "s2") in delta_rep.sparse_delta.added_facts
    assert ("o1", "located_on", "s1") in delta_rep.sparse_delta.deleted_facts
    assert ("o1", "owned_by", "a1") not in delta_rep.sparse_delta.deleted_facts
    assert ("o1", "owned_by", "a1") in delta_rep.occluded_in_after_facts
    assert verify_unobserved_preservation(before, delta_rep.sparse_delta, mask_after) is True


def test_s1_episode_ledger_append_only_hash_chain_freeze_and_audit(tmp_path: Path):
    db_path = tmp_path / "s1_test_ledger.sqlite3"
    train_eps = generate_training_episodes(101, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    parser = TypedRoleFrameParser(use_state_types=True)

    with EpisodeLedger(db_path) as ledger:
        recs = [ledger.ingest_training_episode(ep, parser=parser) for ep in train_eps[:6]]
        assert ledger.count() == 6
        integrity = ledger.verify_integrity(strict=True)
        assert integrity.ok is True
        assert integrity.checked_records == 6
        assert integrity.triggers_present is True

        # Duplicate episode ID must be rejected
        with pytest.raises(LedgerAppendError):
            ledger.ingest_training_episode(train_eps[0], parser=parser)

        # SQLite append-only triggers must block UPDATE and DELETE directly
        with pytest.raises(sqlite3.DatabaseError):
            ledger._conn.execute("UPDATE episodes SET split_id = 'tampered'")
        with pytest.raises(sqlite3.DatabaseError):
            ledger._conn.execute("DELETE FROM episodes")

        # Freeze blocks all further appends and records test-time memory accesses
        ledger.freeze()
        assert ledger.is_frozen is True
        with pytest.raises(LedgerFrozenError):
            ledger.ingest_training_episode(train_eps[6], parser=parser)

        hook_events = []
        ledger.register_access_hook(hook_events.append)
        with ledger.audit_task("probe_exact", caller_tag="test") as scope:
            hits = ledger.lookup_exact_state_action(recs[0].state_action_key, caller_tag="test")
            assert len(hits) == 1
        assert scope.result().exact_key_hits == 1
        assert len(hook_events) >= 1

    # Reload from disk in read-only mode and verify integrity
    with EpisodeLedger(db_path, read_only=True) as reloaded:
        assert reloaded.count() == 6
        assert reloaded.verify_integrity(strict=True).ok is True


def test_s1_parser_disambiguation_ambiguity_and_ood_abstention():
    suite = generate_s1_parser_suite(101, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    by_name = {fn().name: fn for fn in build_parser_factories()}
    full_res = evaluate_parser_on_seed(by_name["s1_typed_frame_parser_full"], 101, suite=suite)
    no_type_res = evaluate_parser_on_seed(by_name["s1_typed_frame_parser_no_types"], 101, suite=suite)
    ccg_res = evaluate_parser_on_seed(by_name["existing_mirror_bilingual_ccg"], 101, suite=suite)

    assert full_res["per_split"]["s0_unseen_entities_and_compositions"]["exact_role_alignment_accuracy"] >= 0.99
    assert full_res["per_split"]["s1_unseen_paraphrases"]["exact_role_alignment_accuracy"] >= 0.95
    assert full_res["per_split"]["s1_state_typed_disambiguation"]["unambiguous_exact_role_accuracy"] == 1.0
    assert no_type_res["per_split"]["s1_state_typed_disambiguation"]["unambiguous_exact_role_accuracy"] == 0.0
    assert full_res["per_split"]["s1_genuinely_ambiguous"]["ambiguity_rate"] == 1.0
    assert full_res["per_split"]["s1_genuinely_ambiguous"]["top_k_candidate_recall"] == 1.0
    assert full_res["per_split"]["s1_unparseable_ood"]["abstention_rate"] == 1.0
    assert (
        full_res["per_split"]["s0_unseen_entities_and_compositions"]["exact_role_alignment_accuracy"]
        - ccg_res["per_split"]["s0_unseen_entities_and_compositions"]["exact_role_alignment_accuracy"]
        >= 0.70
    )


def test_s1_committed_manifests_and_end_to_end_gate_pass(tmp_path: Path):
    ok, mismatches = verify_s1_committed_manifests(
        seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION
    )
    assert ok is True, f"S1 manifest drift detected: {mismatches}"

    summary = run_s1_experiments(
        seeds=(101, 202),
        demos_per_action=DEFAULT_DEMOS_PER_ACTION,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        verify_s0_manifests=False,
    )
    for k, passed in summary["s1_gate"]["checks"].items():
        if k != "five_distinct_fixed_seeds_present":
            assert passed is True, f"S1 check failed: {k}"
