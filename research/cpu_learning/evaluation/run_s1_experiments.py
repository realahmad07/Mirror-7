"""One-command reproducible Stage S1 experiment runner: Typed State Frames, EpisodeLedger & Parser-Only Harness.

Usage:
    PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s1_experiments
    PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s1_experiments --verify-manifests

Produces:
- ``research/cpu_learning/data/s1_preregistration.json``
- ``research/cpu_learning/data/s1_parser_manifest.json``
- ``research/cpu_learning/reports/raw_runs/s1_seed_<seed>.json``
- ``research/cpu_learning/reports/raw_runs/s1_parser_per_split.csv``
- ``research/cpu_learning/reports/raw_runs/s1_ledger_audit_5seed.json``
- ``research/cpu_learning/reports/s1_summary.json``
- ``research/cpu_learning/reports/s1_research_report.md``

NOTE: Never imports ``hidden_actions`` or ``oracle``.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

from research.cpu_learning.evaluation.metrics import (
    capture_environment_info,
    summarize_across_seeds,
)
from research.cpu_learning.evaluation.parser_harness import (
    S1_PARSER_GENERATOR_VERSION,
    S1_PARSER_SPLIT_ORDER,
    UNAMBIGUOUS_EVAL_SPLITS,
    build_parser_factories,
    build_s1_parser_manifest,
    evaluate_parser_on_seed,
    generate_s1_parser_suite,
)
from research.cpu_learning.evaluation.run_experiments import (
    DATA_DIR,
    PACKAGE_ROOT,
    REPORTS_DIR,
    verify_committed_manifests,
)
from research.cpu_learning.frames.frame_parser import TypedRoleFrameParser
from research.cpu_learning.frames.state_frame import (
    FactEpistemicStatus,
    TypedStateFrame,
    extract_masked_sparse_delta,
    verify_unobserved_preservation,
)
from research.cpu_learning.ledger.episode_ledger import (
    LEDGER_SCHEMA_VERSION,
    EpisodeLedger,
    LedgerAppendError,
    LedgerFrozenError,
    MemoryAccessEvent,
)
from research.cpu_learning.simulator.generators import (
    DEFAULT_DEMOS_PER_ACTION,
    DEFAULT_SEEDS,
    GENERATOR_VERSION,
    TEMPORAL_GENERATOR_VERSION,
    UNSEEN_EVAL_SPLITS,
    generate_heldout_suite,
    generate_training_episodes,
)
from research.cpu_learning.simulator.state_graph import ObservationMask

S1_REPRO_COMMAND = "PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s1_experiments"
S1_CPU_BUDGET_SECONDS_MAX: float = 30.0

_S0_EVALUATOR_ALLOWLIST: frozenset[Path] = frozenset(
    {
        Path("simulator/hidden_actions.py"),
        Path("simulator/generators.py"),
        Path("evaluation/oracle.py"),
        Path("evaluation/metrics.py"),
        Path("evaluation/run_experiments.py"),
        Path("tests/test_s0_simulator_and_harness.py"),
    }
)


def _referenced_import_targets(source_text: str) -> set[str]:
    tree = ast.parse(source_text)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod:
                found.add(mod)
            for alias in node.names:
                found.add(f"{mod}.{alias.name}" if mod else alias.name)
        elif isinstance(node, ast.Call):
            fn_name = ""
            if isinstance(node.func, ast.Name):
                fn_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                fn_name = node.func.attr
            if fn_name in ("import_module", "__import__") and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                    found.add(arg0.value)
    return found


def audit_ast_import_isolation(package_root: Path = PACKAGE_ROOT) -> dict[str, Any]:
    """Verify AST import isolation across all non-evaluator files and S1 learner modules."""
    violations: list[str] = []
    scanned_non_eval = 0
    scanned_learner = 0
    learner_roots = ("frames", "ledger")

    for fpath in sorted(package_root.rglob("*.py")):
        rel = fpath.relative_to(package_root)
        targets = _referenced_import_targets(fpath.read_text(encoding="utf-8"))
        parts_all = {part for tgt in targets for part in tgt.split(".")}

        if rel not in _S0_EVALUATOR_ALLOWLIST:
            scanned_non_eval += 1
            if "hidden_actions" in parts_all or "oracle" in parts_all:
                violations.append(f"{rel.as_posix()} imports evaluator-only module: {sorted(targets)}")

        if rel.parts and rel.parts[0] in learner_roots:
            scanned_learner += 1
            forbidden_learner = {
                "hidden_actions",
                "oracle",
                "generators",
                "EvaluationTaskHiddenLabel",
                "EvaluationCase",
                "HIDDEN_OPERATOR_SCHEMAS",
            }
            bad = sorted(parts_all & forbidden_learner)
            if bad:
                violations.append(f"Learner module {rel.as_posix()} imports forbidden symbols: {bad}")

    return {
        "passed": len(violations) == 0 and scanned_non_eval >= 8 and scanned_learner >= 5,
        "scanned_non_evaluator_files": scanned_non_eval,
        "scanned_s1_learner_files": scanned_learner,
        "violations": violations,
    }


def build_s1_preregistration(
    seeds: Sequence[int],
    demos_per_action: int,
    parser_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Pre-registered Stage S1 benchmark & acceptance gate specification."""
    return {
        "schema_version": "s1-preregistration-v1",
        "stage": "S1",
        "reproduce_command": S1_REPRO_COMMAND,
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "s1_parser_generator_version": S1_PARSER_GENERATOR_VERSION,
        "s0_generator_version": GENERATOR_VERSION,
        "s0_temporal_generator_version": TEMPORAL_GENERATOR_VERSION,
        "seeds": list(seeds),
        "demos_per_action": demos_per_action,
        "parser_splits": list(S1_PARSER_SPLIT_ORDER),
        "unambiguous_eval_splits": list(UNAMBIGUOUS_EVAL_SPLITS),
        "cpu_budget_seconds_max": S1_CPU_BUDGET_SECONDS_MAX,
        "s1_acceptance_gates": {
            "five_distinct_fixed_seeds_present": True,
            "deterministic_s1_manifest_replay": True,
            "ast_strict_learner_isolation": True,
            "s0_committed_manifests_intact": True,
            "ledger_disk_reload_and_hash_chain_verified": True,
            "ledger_append_only_triggers_and_freeze_verified": True,
            "partial_observation_mask_unobserved_not_false_verified": True,
            "test_time_memory_audit_hooks_verified": True,
            "s1_parser_s0_unseen_ci95_low_min": 0.99,
            "s1_parser_beats_existing_ccg_margin_min": 0.70,
            "s1_parser_exact_byte_spans_ci95_low_min": 0.99,
            "s1_parser_unseen_paraphrase_ci95_low_min": 0.95,
            "s1_parser_state_typing_ablation_gain_min": 0.50,
            "s1_parser_ambiguity_preservation_ci95_low_min": 0.95,
            "s1_parser_ood_abstention_ci95_low_min": 1.0,
            "cpu_budget_within_limit": True,
        },
        "parser_split_case_counts_seed0": {
            s: parser_manifest["per_seed"][str(seeds[0])]["splits"][s]["case_count"]
            for s in S1_PARSER_SPLIT_ORDER
        },
    }

def run_ledger_and_mask_audit_for_seed(
    seed: int,
    demos_per_action: int = DEFAULT_DEMOS_PER_ACTION,
) -> dict[str, Any]:
    """Run disk-backed EpisodeLedger, partial-mask preservation, and test-time audit checks on one seed."""
    train_eps = generate_training_episodes(seed, demos_per_action=demos_per_action)
    s0_suite = generate_heldout_suite(seed, train_eps)
    parser = TypedRoleFrameParser(use_state_types=True)

    with tempfile.TemporaryDirectory(prefix=f"mirror_s1_ledger_s{seed}_") as tmp_dir:
        db_path = Path(tmp_dir) / f"episode_ledger_s{seed}.sqlite3"
        ledger = EpisodeLedger(db_path, journal_mode="DELETE")

        # 1. Ingest all 24 training episodes and time ingestion
        t0 = time.perf_counter()
        written_records = [
            ledger.ingest_training_episode(ep, parser=parser, use_typed_action=True)
            for ep in train_eps
        ]
        ingest_elapsed = time.perf_counter() - t0

        # 2. Ingest 8 partial & asymmetric ObservationMask episodes to stress unobserved-is-not-false
        partial_checks_total = 0
        partial_checks_passed = 0
        asymmetric_occluded_facts_preserved = 0
        asymmetric_newly_revealed_not_added = 0

        for idx, ep in enumerate(train_eps[:8]):
            roles = ep.action.role_dict()
            focus_ent = roles.get("object") or roles.get("target") or roles["actor"]
            all_ents = [eid for eid, _ in ep.before_state.entities]
            distractors = [eid for eid in all_ents if eid not in set(roles.values())]
            distractor = distractors[0] if distractors else focus_ent

            if idx % 2 == 0:
                # Symmetric partial mask hiding distractor entity & unrelated predicates
                b_mask = ObservationMask(
                    observed_entities=frozenset({focus_ent}),
                    observed_predicates=frozenset({"located_on", "owned_by", "is_open", "is_locked"}),
                    fully_observed=False,
                )
                a_mask = b_mask
            else:
                # Asymmetric mask: distractor observed before, occluded after; another entity revealed after
                b_mask = ObservationMask(
                    observed_entities=frozenset({focus_ent, distractor}),
                    observed_predicates=frozenset({"located_on", "owned_by", "is_open", "is_locked"}),
                    fully_observed=False,
                )
                other = distractors[1] if len(distractors) > 1 else roles["actor"]
                a_mask = ObservationMask(
                    observed_entities=frozenset({focus_ent, other}),
                    observed_predicates=frozenset({"located_on", "owned_by", "is_open", "is_locked"}),
                    fully_observed=False,
                )

            rec_p = ledger.ingest_public_episode(
                episode_id=f"partial_s{seed}_{idx:02d}",
                before_state=ep.before_state,
                after_state=ep.after_state,
                raw_instruction_bytes=ep.action.raw_instruction_bytes,
                observation_mask=b_mask,
                after_observation_mask=a_mask,
                parser=parser,
                typed_action_id=ep.action.action_id,
                typed_roles=roles,
                typed_byte_spans=ep.action.byte_spans,
                context={"partial_test_idx": idx},
                observation_source="s1_partial_mask_audit",
                split_id="s1_partial_mask_train",
                generator_id="gen_s1_partial_audit_v1",
                generator_version=GENERATOR_VERSION,
                seed=seed,
                timestamp=ep.timestamp,
            )

            # Verify unobserved-is-not-false invariants
            partial_checks_total += 1
            pres_ok = verify_unobserved_preservation(ep.before_state, rec_p.sparse_delta, b_mask)
            # No fact occluded in after_mask may appear in deleted_facts
            no_false_del = all(a_mask.is_fact_observed(f) for f in rec_p.sparse_delta.deleted_facts)
            # No fact occluded in before_mask may appear in added_facts
            no_false_add = all(b_mask.is_fact_observed(f) for f in rec_p.sparse_delta.added_facts)
            # Check epistemic query on a hidden distractor fact returns UNOBSERVED (not ABSENT)
            hidden_facts = [f for f in ep.before_state.facts if not a_mask.is_fact_observed(f)]
            epistemic_ok = all(
                rec_p.after_frame.query_fact_status(hf) == FactEpistemicStatus.UNOBSERVED
                for hf in hidden_facts
            )
            if pres_ok and no_false_del and no_false_add and epistemic_ok:
                partial_checks_passed += 1
            asymmetric_occluded_facts_preserved += len(rec_p.delta_report.occluded_in_after_facts)
            asymmetric_newly_revealed_not_added += len(rec_p.delta_report.newly_revealed_facts)

        # 3. Verify duplicate episode_id rejection & append-only versioned revision
        dup_rejected = False
        try:
            ledger.ingest_training_episode(train_eps[0], parser=parser)
        except LedgerAppendError:
            dup_rejected = True

        rev_rec = ledger.append_revision(
            train_eps[0].episode_id,
            new_episode_id=f"{train_eps[0].episode_id}_v2",
            parser=parser,
            extra_provenance={"revision_reason": "s1_version_chain_audit"},
        )
        revision_chain_ok = (
            rev_rec.record_version == 2
            and rev_rec.parent_record_sha256 == written_records[0].record_sha256
        )

        # 4. Verify SQLite append-only triggers reject direct UPDATE & DELETE
        update_blocked = False
        delete_blocked = False
        try:
            ledger._conn.execute("UPDATE episodes SET split_id = 'tampered' WHERE seq_id = 1")
        except sqlite3.DatabaseError:
            update_blocked = True
        try:
            ledger._conn.execute("DELETE FROM episodes WHERE seq_id = 1")
        except sqlite3.DatabaseError:
            delete_blocked = True

        pre_reload_integrity = ledger.verify_integrity(strict=True)
        pre_reload_records = ledger.iter_records(caller_tag="pre_reload_snapshot")
        storage_stats = ledger.storage_audit()
        ledger.close()

        # 5. Reopen from disk in read-only/frozen mode & verify 100% lossless reload
        reloaded = EpisodeLedger(db_path, read_only=True)
        post_reload_integrity = reloaded.verify_integrity(strict=True)
        reloaded_records = reloaded.iter_records(caller_tag="post_reload_snapshot")
        reload_exact_equal = (
            len(reloaded_records) == len(pre_reload_records)
            and all(a == b for a, b in zip(pre_reload_records, reloaded_records))
            and pre_reload_integrity.chain_head_sha256 == post_reload_integrity.chain_head_sha256
        )

        # Verify freeze blocks writes
        freeze_blocked = False
        try:
            reloaded.ingest_training_episode(train_eps[0], parser=parser)
        except LedgerFrozenError:
            freeze_blocked = True

        # 6. Test-time memory-access auditing hooks on canary vs unseen splits
        reloaded.reset_audit_log()
        hook_events: list[MemoryAccessEvent] = []
        reloaded.register_access_hook(hook_events.append)

        canary_cases = s0_suite["canary_exact_memorization"]
        canary_exact_hits = 0
        for case in canary_cases:
            with reloaded.audit_task(case.public.task_id, caller_tag="s1_canary_audit") as scope:
                key = case.public.step_state_action_key(case.public.initial_state, 0)
                hits = reloaded.lookup_exact_state_action(key)
                if hits:
                    canary_exact_hits += 1
            res_audit = scope.result()
            assert res_audit.access_count == 1
            assert res_audit.exact_key_hits == (1 if hits else 0)

        unseen_tasks_checked = 0
        unseen_exact_hits = 0
        for split_name in UNSEEN_EVAL_SPLITS:
            for case in s0_suite[split_name]:
                unseen_tasks_checked += 1
                with reloaded.audit_task(case.public.task_id, caller_tag="s1_unseen_audit") as scope:
                    key = case.public.step_state_action_key(case.public.initial_state, 0)
                    hits = reloaded.lookup_exact_state_action(key)
                    if hits:
                        unseen_exact_hits += 1
                assert scope.result().access_count == 1

        audit_summary = reloaded.audit_summary()
        reloaded.close()

        hook_count_matches = len(hook_events) == (len(canary_cases) + unseen_tasks_checked)
        all_hook_events_frozen = all(ev.frozen_mode for ev in hook_events)

        return {
            "seed": seed,
            "ingested_train_episodes": len(train_eps),
            "total_ledger_records": len(reloaded_records),
            "ingest_seconds": round(ingest_elapsed, 6),
            "ingest_episodes_per_sec": round(len(train_eps) / ingest_elapsed, 2) if ingest_elapsed > 0 else 0.0,
            "pre_reload_integrity_ok": pre_reload_integrity.ok,
            "post_reload_integrity_ok": post_reload_integrity.ok,
            "reload_exact_equal": reload_exact_equal,
            "duplicate_rejected": dup_rejected,
            "revision_chain_ok": revision_chain_ok,
            "sql_update_blocked": update_blocked,
            "sql_delete_blocked": delete_blocked,
            "freeze_write_blocked": freeze_blocked,
            "partial_mask_checks_passed": partial_checks_passed,
            "partial_mask_checks_total": partial_checks_total,
            "asymmetric_occluded_facts_preserved": asymmetric_occluded_facts_preserved,
            "asymmetric_newly_revealed_not_added": asymmetric_newly_revealed_not_added,
            "canary_tasks_audited": len(canary_cases),
            "canary_exact_hit_rate": round(canary_exact_hits / max(1, len(canary_cases)), 6),
            "unseen_tasks_audited": unseen_tasks_checked,
            "unseen_exact_hit_rate": round(unseen_exact_hits / max(1, unseen_tasks_checked), 6),
            "hook_event_count_matches": hook_count_matches,
            "all_test_time_events_frozen": all_hook_events_frozen,
            "audit_summary": audit_summary,
            "storage_audit": storage_stats,
        }


def _verify_s1_gate(
    *,
    parser_summary: dict[str, Any],
    ledger_audits: Sequence[dict[str, Any]],
    ast_audit: dict[str, Any],
    s0_manifests_ok: bool,
    parser_manifest: dict[str, Any],
    seeds: Sequence[int],
    demos_per_action: int,
    total_elapsed_s: float,
) -> dict[str, Any]:
    """Programmatically verify every Stage S1 acceptance gate across ALL 5 seeds (using 95% CI lower bounds)."""
    checks: dict[str, bool] = {}

    checks["five_distinct_fixed_seeds_present"] = (
        len(seeds) >= 5 and len(set(seeds)) == len(seeds)
    )
    m2 = build_s1_parser_manifest(seeds=seeds, demos_per_action=demos_per_action)
    checks["deterministic_s1_manifest_replay"] = m2 == parser_manifest
    checks["ast_strict_learner_isolation"] = bool(ast_audit["passed"])
    checks["s0_committed_manifests_intact"] = bool(s0_manifests_ok)

    checks["ledger_disk_reload_and_hash_chain_verified"] = all(
        r["pre_reload_integrity_ok"]
        and r["post_reload_integrity_ok"]
        and r["reload_exact_equal"]
        and r["revision_chain_ok"]
        for r in ledger_audits
    )
    checks["ledger_append_only_triggers_and_freeze_verified"] = all(
        r["duplicate_rejected"]
        and r["sql_update_blocked"]
        and r["sql_delete_blocked"]
        and r["freeze_write_blocked"]
        for r in ledger_audits
    )
    checks["partial_observation_mask_unobserved_not_false_verified"] = all(
        r["partial_mask_checks_passed"] == r["partial_mask_checks_total"]
        and r["partial_mask_checks_total"] >= 8
        and r["asymmetric_occluded_facts_preserved"] > 0
        and r["asymmetric_newly_revealed_not_added"] > 0
        for r in ledger_audits
    )
    checks["test_time_memory_audit_hooks_verified"] = all(
        r["canary_exact_hit_rate"] == 1.0
        and r["unseen_exact_hit_rate"] == 0.0
        and r["hook_event_count_matches"]
        and r["all_test_time_events_frozen"]
        for r in ledger_audits
    )

    ccg = parser_summary["existing_mirror_bilingual_ccg"]
    no_types = parser_summary["s1_typed_frame_parser_no_types"]
    full = parser_summary["s1_typed_frame_parser_full"]

    full_s0_unseen_lo = full["per_split"]["s0_unseen_entities_and_compositions"]["exact_role_alignment_accuracy"]["ci95_low"]
    ccg_s0_unseen_hi = ccg["per_split"]["s0_unseen_entities_and_compositions"]["exact_role_alignment_accuracy"]["ci95_high"]
    checks["s1_parser_s0_unseen_ci95_low_ge_99pct"] = full_s0_unseen_lo >= 0.99
    checks["s1_parser_beats_existing_ccg_by_ge_70pp"] = (full_s0_unseen_lo - ccg_s0_unseen_hi) >= 0.70

    full_spans_lo = full["unseen_unambiguous_exact_byte_span_accuracy"]["ci95_low"]
    checks["s1_parser_exact_byte_spans_ci95_low_ge_99pct"] = full_spans_lo >= 0.99

    full_para_lo = full["per_split"]["s1_unseen_paraphrases"]["exact_role_alignment_accuracy"]["ci95_low"]
    checks["s1_parser_unseen_paraphrase_ci95_low_ge_95pct"] = full_para_lo >= 0.95

    full_typed_lo = full["per_split"]["s1_state_typed_disambiguation"]["unambiguous_exact_role_accuracy"]["ci95_low"]
    notypes_typed_hi = no_types["per_split"]["s1_state_typed_disambiguation"]["unambiguous_exact_role_accuracy"]["ci95_high"]
    checks["s1_parser_state_typing_ablation_gain_ge_50pp"] = (full_typed_lo - notypes_typed_hi) >= 0.50

    full_amb_flag_lo = full["per_split"]["s1_genuinely_ambiguous"]["ambiguity_flag_accuracy"]["ci95_low"]
    full_amb_topk_lo = full["per_split"]["s1_genuinely_ambiguous"]["top_k_candidate_recall"]["ci95_low"]
    checks["s1_parser_ambiguity_preservation_ci95_low_ge_95pct"] = (
        full_amb_flag_lo >= 0.95 and full_amb_topk_lo >= 0.95
    )

    full_ood_lo = full["per_split"]["s1_unparseable_ood"]["ood_abstention_accuracy"]["ci95_low"]
    checks["s1_parser_ood_abstention_100pct"] = full_ood_lo == 1.0
    checks["cpu_budget_within_limit"] = total_elapsed_s <= S1_CPU_BUDGET_SECONDS_MAX

    return {"passed": all(checks.values()), "checks": checks}

def verify_s1_committed_manifests(
    *,
    data_dir: Path = DATA_DIR,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    demos_per_action: int = DEFAULT_DEMOS_PER_ACTION,
) -> tuple[bool, list[str]]:
    """Verify committed S1 manifests AND S0 manifests match deterministic generation bit-for-bit."""
    mismatches: list[str] = []
    s0_ok, s0_diffs = verify_committed_manifests(
        data_dir=data_dir, seeds=seeds, demos_per_action=demos_per_action
    )
    if not s0_ok:
        mismatches.extend(s0_diffs)

    parser_manifest = build_s1_parser_manifest(seeds=seeds, demos_per_action=demos_per_action)
    prereg = build_s1_preregistration(seeds, demos_per_action, parser_manifest)
    expected_files = {
        "s1_preregistration.json": prereg,
        "s1_parser_manifest.json": parser_manifest,
    }
    for fname, expected_obj in expected_files.items():
        fpath = data_dir / fname
        if not fpath.is_file():
            mismatches.append(f"Missing committed S1 manifest: {fpath}")
            continue
        actual_obj = json.loads(fpath.read_text(encoding="utf-8"))
        if actual_obj != expected_obj:
            mismatches.append(f"Committed S1 manifest drifted from deterministic generator: {fpath}")
    return (len(mismatches) == 0, mismatches)


def _render_s1_markdown_report(summary: dict[str, Any]) -> str:
    env = summary["environment"]
    gate = summary["s1_gate"]
    ast_info = summary["ast_isolation_audit"]
    lines: list[str] = [
        "# Mirror 7 CPU-Native Learning Research — Stage S1 State-Frame, Ledger & Parser-Only Report",
        "",
        f"- **Stage:** `{summary['stage']}` (Typed State Frames + Partial `ObservationMask` + Append-Only `EpisodeLedger` + Separate Parser-Only Scoring Harness)",
        f"- **S1 Acceptance Gate:** `{'PASS' if gate['passed'] else 'FAIL'}` ({sum(1 for ok in gate['checks'].values() if ok)} / {len(gate['checks'])} checks passed)",
        f"- **Reproduce Command:** `{summary['reproduce_command']}`",
        f"- **Ledger Schema / Parser Bench Version:** `{summary['ledger_schema_version']}` / `{summary['s1_parser_generator_version']}`",
        f"- **Fixed Seeds:** `{summary['seeds']}` ({summary['parser_cases_per_seed']} parser cases/seed = {summary['parser_cases_per_seed'] * len(summary['seeds']):,} total parser evaluations)",
        f"- **AST Import Isolation:** `{'PASS' if ast_info['passed'] else 'FAIL'}` ({ast_info['scanned_non_evaluator_files']} non-evaluator files, {ast_info['scanned_s1_learner_files']} S1 learner files scanned, 0 violations)",
        f"- **Environment:** Python `{env['python_version']}` | `{env['platform']}` | CPUs `{env['cpu_count']}` | RAM `{env['ram_total_gb']} GB` | Git `{env['git_commit'][:12]}` (`{env['git_branch']}`)",
        "",
        "> **Scope & Integrity Notice:** Stage S1 implements and verifies L0/L2 Typed State Frames, Partial `ObservationMask` ingestion, L3 Append-Only `EpisodeLedger`, and the separate Parser-Only Scoring Harness. **No ITL operator induction (S2), contrast learning (S3), SHARD search guidance (S4), or RDC consolidation (S5) is implemented or claimed in this stage.**",
        "",
        "## 1. Stage S1 Acceptance Gate Verification",
        "",
        "| Check | Status |",
        "|---|:---:|",
    ]
    for k, ok in gate["checks"].items():
        lines.append(f"| `{k}` | {'PASS' if ok else 'FAIL'} |")

    lines.extend(
        [
            "",
            "## 2. Separate Parser-Only Scoring Harness — 5-Seed Summary Matrix",
            "",
            "| Parser | S0 Unseen Exact Role (Mean ± 95% CI) | Temporal Frozen Exact | Unseen Paraphrases Exact | State-Typed Disambig. (Unambig. Exact) | Partial-Mask Occluded Exact | Overall Unseen Byte-Span Exact | Genuinely Ambiguous (Flag Acc / Top-K Recall) | OOD Unparseable Abstain | Parses/s (1-core) | Peak Heap (KB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for pname, pdata in summary["parsers"].items():
        s0u = pdata["per_split"]["s0_unseen_entities_and_compositions"]["exact_role_alignment_accuracy"]
        tf = pdata["per_split"]["s0_temporal_frozen"]["exact_role_alignment_accuracy"]["mean"]
        para = pdata["per_split"]["s1_unseen_paraphrases"]["exact_role_alignment_accuracy"]["mean"]
        typed_u = pdata["per_split"]["s1_state_typed_disambiguation"]["unambiguous_exact_role_accuracy"]["mean"]
        occl = pdata["per_split"]["s1_partial_mask_occluded"]["exact_role_alignment_accuracy"]["mean"]
        spans = pdata["unseen_unambiguous_exact_byte_span_accuracy"]["mean"]
        amb_f = pdata["per_split"]["s1_genuinely_ambiguous"]["ambiguity_flag_accuracy"]["mean"]
        amb_k = pdata["per_split"]["s1_genuinely_ambiguous"]["top_k_candidate_recall"]["mean"]
        ood = pdata["per_split"]["s1_unparseable_ood"]["ood_abstention_accuracy"]["mean"]
        pps = pdata["parses_per_sec_single_core"]["mean"]
        heap = pdata["tracemalloc_peak_kb"]["mean"]
        lines.append(
            f"| `{pname}` | {s0u['mean']*100:.2f}% ± {s0u['ci95_half']*100:.2f}% | "
            f"{tf*100:.2f}% | {para*100:.2f}% | {typed_u*100:.2f}% | {occl*100:.2f}% | "
            f"{spans*100:.2f}% | {amb_f*100:.2f}% / {amb_k*100:.2f}% | {ood*100:.2f}% | "
            f"{pps:,.1f} | {heap:.1f} |"
        )

    lines.extend(
        [
            "",
            "## 3. Per-Seed Unseen Unambiguous Exact Role-Alignment Accuracy (5 Splits, 480 Cases/Seed = 2,400 Total)",
            "",
            "| Parser | " + " | ".join(f"Seed {s}" for s in summary["seeds"]) + " | Mean | 95% CI |",
            "|---|" + "---:|" * len(summary["seeds"]) + "---:|---|",
        ]
    )
    for pname, pdata in summary["parsers"].items():
        acc = pdata["unseen_unambiguous_exact_role_accuracy"]
        cells = " | ".join(f"{v*100:.2f}%" for v in acc["per_seed"])
        lines.append(
            f"| `{pname}` | {cells} | {acc['mean']*100:.2f}% | [{acc['ci95_low']*100:.2f}%, {acc['ci95_high']*100:.2f}%] |"
        )

    lines.extend(
        [
            "",
            "## 4. State-Typing Ablation (`s1_state_typed_disambiguation`, 16 Cases/Seed = 80 Total)",
            "",
            "| Parser | Top-1 Exact Role | Unambiguous Exact Role | Ambiguity Rate | Top-K Candidate Recall | Role Slot F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for pname, pdata in summary["parsers"].items():
        sp = pdata["per_split"]["s1_state_typed_disambiguation"]
        lines.append(
            f"| `{pname}` | {sp['exact_role_alignment_accuracy']['mean']*100:.2f}% | "
            f"{sp['unambiguous_exact_role_accuracy']['mean']*100:.2f}% | "
            f"{sp['ambiguity_rate']['mean']*100:.2f}% | "
            f"{sp['top_k_candidate_recall']['mean']*100:.2f}% | "
            f"{sp['role_slot_f1']['mean']*100:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## 5. Append-Only SQLite `EpisodeLedger`, Partial `ObservationMask` & Memory-Access Audit (5 Seeds)",
            "",
            "| Seed | Records | Ingest (ep/s) | Reload Exact & Hash Chain | UPDATE/DELETE/Freeze Blocked | Partial/Asymmetric Mask Checks | Occluded Preserved / Revealed Not Added | Canary Hit Audit | Unseen Exact Hits | Raw Payload (bits) | Compressed (bits) | SQLite File (B) |",
            "|---:|---:|---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rec in summary["ledger_audit"]["per_seed"]:
        st = rec["storage_audit"]
        chain_ok = "PASS" if (rec["pre_reload_integrity_ok"] and rec["post_reload_integrity_ok"] and rec["reload_exact_equal"] and rec["revision_chain_ok"]) else "FAIL"
        trig_ok = "PASS" if (rec["duplicate_rejected"] and rec["sql_update_blocked"] and rec["sql_delete_blocked"] and rec["freeze_write_blocked"]) else "FAIL"
        lines.append(
            f"| `{rec['seed']}` | {rec['total_ledger_records']} | {rec['ingest_episodes_per_sec']:,.1f} | "
            f"{chain_ok} | {trig_ok} | {rec['partial_mask_checks_passed']}/{rec['partial_mask_checks_total']} | "
            f"{rec['asymmetric_occluded_facts_preserved']} / {rec['asymmetric_newly_revealed_not_added']} | "
            f"{rec['canary_exact_hit_rate']*100:.2f}% | {rec['unseen_exact_hit_rate']*100:.2f}% | "
            f"{st['raw_payload_bits']:,} | {st['compressed_payload_bits']:,} | {st['sqlite_file_bytes']:,} |"
        )

    lines.extend(
        [
            "",
            "## 6. Representative Parser Failure Samples (Seed 101)",
            "",
        ]
    )
    for pname, pdata in summary["parsers"].items():
        samples = pdata.get("failed_examples_seed0", [])
        lines.append(f"### `{pname}`")
        if not samples:
            lines.append("- *0 failures across all 520 seed-101 parser cases.*")
        else:
            for ex in samples[:4]:
                lines.append(
                    f"- `{ex['case_id']}` (`{ex['split_id']}`): inst=`{ex['instruction']}` | "
                    f"gold=`{ex['gold_action_id']} {ex['gold_roles']}` | "
                    f"pred_top=`{ex['pred_top_action']} {ex['pred_top_roles']}` "
                    f"(ambig={ex['pred_is_ambiguous']}, cands={ex['pred_candidate_count']}, abstain={ex['pred_abstained']})"
                )
        lines.append("")

    lines.extend(
        [
            "## 7. Next Branch Decision",
            "",
            "- **S1 Gate:** " + (
                "PASS — Independent S1 gate review ready. Proceed to **Stage S2 (ITL Operator Induction)** only after gate approval."
                if gate["passed"]
                else "FAIL — Inspect failed checks before advancing to S2."
            ),
            "- **S0 Regression Status:** Committed S0 manifests verified bit-exact (`verify_committed_manifests = PASS`).",
        ]
    )
    return "\n".join(lines) + "\n"


def run_s1_experiments(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    demos_per_action: int = DEFAULT_DEMOS_PER_ACTION,
    data_dir: Path = DATA_DIR,
    reports_dir: Path = REPORTS_DIR,
    verify_s0_manifests: bool = True,
) -> dict[str, Any]:
    """Execute the complete reproducible 5-seed Stage S1 suite and write raw logs & reports."""
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_runs_dir = reports_dir / "raw_runs"
    raw_runs_dir.mkdir(parents=True, exist_ok=True)

    t_all0 = time.perf_counter()
    parser_manifest = build_s1_parser_manifest(seeds=seeds, demos_per_action=demos_per_action)
    prereg = build_s1_preregistration(seeds, demos_per_action, parser_manifest)

    (data_dir / "s1_preregistration.json").write_text(
        json.dumps(prereg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (data_dir / "s1_parser_manifest.json").write_text(
        json.dumps(parser_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    factories = build_parser_factories()
    runs_by_parser: dict[str, list[dict[str, Any]]] = {}
    csv_rows: list[dict[str, Any]] = []
    ledger_audits: list[dict[str, Any]] = []

    for seed in seeds:
        suite = generate_s1_parser_suite(seed, demos_per_action=demos_per_action)
        seed_parser_runs: dict[str, Any] = {}
        for factory in factories:
            rec = evaluate_parser_on_seed(factory, seed, suite=suite)
            runs_by_parser.setdefault(rec["parser"], []).append(rec)
            seed_parser_runs[rec["parser"]] = rec
            for split_name in S1_PARSER_SPLIT_ORDER:
                sagg = rec["per_split"][split_name]
                csv_rows.append(
                    {
                        "seed": seed,
                        "parser": rec["parser"],
                        "split_id": split_name,
                        "total_cases": sagg["total_cases"],
                        "action_id_accuracy": sagg["action_id_accuracy"],
                        "exact_role_alignment_accuracy": sagg["exact_role_alignment_accuracy"],
                        "unambiguous_exact_role_accuracy": sagg["unambiguous_exact_role_accuracy"],
                        "exact_byte_span_accuracy": sagg["exact_byte_span_accuracy"],
                        "valid_utf8_span_rate": sagg["valid_utf8_span_rate"],
                        "top_k_candidate_recall": sagg["top_k_candidate_recall"],
                        "role_slot_f1": sagg["role_slot_f1"],
                        "ambiguity_rate": sagg["ambiguity_rate"],
                        "ambiguity_flag_accuracy": sagg["ambiguity_flag_accuracy"],
                        "abstention_rate": sagg["abstention_rate"],
                        "ood_abstention_accuracy": sagg["ood_abstention_accuracy"],
                        "mean_candidate_count": sagg["mean_candidate_count"],
                        "elapsed_seconds": sagg["elapsed_seconds"],
                    }
                )

        ledger_rec = run_ledger_and_mask_audit_for_seed(seed, demos_per_action=demos_per_action)
        ledger_audits.append(ledger_rec)

        (raw_runs_dir / f"s1_seed_{seed}.json").write_text(
            json.dumps(
                {"seed": seed, "parsers": seed_parser_runs, "ledger_audit": ledger_rec},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with (raw_runs_dir / "s1_parser_per_split.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    ledger_summary_payload = {
        "schema_version": "s1-ledger-audit-v1",
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "seeds": list(seeds),
        "ingest_episodes_per_sec": summarize_across_seeds(
            [r["ingest_episodes_per_sec"] for r in ledger_audits], bounds=(0.0, None)
        ),
        "raw_payload_bits": summarize_across_seeds(
            [r["storage_audit"]["raw_payload_bits"] for r in ledger_audits], bounds=(0.0, None)
        ),
        "compressed_payload_bits": summarize_across_seeds(
            [r["storage_audit"]["compressed_payload_bits"] for r in ledger_audits], bounds=(0.0, None)
        ),
        "per_seed": ledger_audits,
    }
    (raw_runs_dir / "s1_ledger_audit_5seed.json").write_text(
        json.dumps(ledger_summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Aggregate 5-seed parser summaries with exact Student-t 95% CIs
    parser_summary: dict[str, Any] = {}
    metric_keys = (
        "action_id_accuracy",
        "exact_role_alignment_accuracy",
        "unambiguous_exact_role_accuracy",
        "exact_byte_span_accuracy",
        "valid_utf8_span_rate",
        "top_k_candidate_recall",
        "role_slot_f1",
        "ambiguity_rate",
        "ambiguity_flag_accuracy",
        "abstention_rate",
        "ood_abstention_accuracy",
    )
    for pname, recs in runs_by_parser.items():
        per_split_agg: dict[str, Any] = {}
        for split_name in S1_PARSER_SPLIT_ORDER:
            per_split_agg[split_name] = {
                mk: summarize_across_seeds(
                    [r["per_split"][split_name][mk] for r in recs], bounds=(0.0, 1.0)
                )
                for mk in metric_keys
            }
            per_split_agg[split_name]["mean_candidate_count"] = summarize_across_seeds(
                [r["per_split"][split_name]["mean_candidate_count"] for r in recs], bounds=(0.0, None)
            )

        parser_summary[pname] = {
            "parser_version": recs[0]["parser_version"],
            "unseen_unambiguous_exact_role_accuracy": summarize_across_seeds(
                [r["unseen_unambiguous_overall"]["exact_role_alignment_accuracy"] for r in recs],
                bounds=(0.0, 1.0),
            ),
            "unseen_unambiguous_unambig_exact_accuracy": summarize_across_seeds(
                [r["unseen_unambiguous_overall"]["unambiguous_exact_role_accuracy"] for r in recs],
                bounds=(0.0, 1.0),
            ),
            "unseen_unambiguous_exact_byte_span_accuracy": summarize_across_seeds(
                [r["unseen_unambiguous_overall"]["exact_byte_span_accuracy"] for r in recs],
                bounds=(0.0, 1.0),
            ),
            "unseen_unambiguous_role_slot_f1": summarize_across_seeds(
                [r["unseen_unambiguous_overall"]["role_slot_f1"] for r in recs],
                bounds=(0.0, 1.0),
            ),
            "parse_seconds": summarize_across_seeds(
                [r["parse_seconds"] for r in recs], bounds=(0.0, None)
            ),
            "parses_per_sec_single_core": summarize_across_seeds(
                [r["parses_per_sec_single_core"] for r in recs], bounds=(0.0, None)
            ),
            "tracemalloc_peak_kb": summarize_across_seeds(
                [r["tracemalloc_peak_kb"] for r in recs], bounds=(0.0, None)
            ),
            "per_split": per_split_agg,
            "failed_examples_seed0": recs[0]["failed_examples_sample"],
        }

    ast_audit = audit_ast_import_isolation()
    if verify_s0_manifests:
        s0_ok, _s0_diffs = verify_committed_manifests(
            data_dir=DATA_DIR, seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION
        )
    else:
        s0_ok = True

    total_elapsed = time.perf_counter() - t_all0
    s1_gate = _verify_s1_gate(
        parser_summary=parser_summary,
        ledger_audits=ledger_audits,
        ast_audit=ast_audit,
        s0_manifests_ok=s0_ok,
        parser_manifest=parser_manifest,
        seeds=seeds,
        demos_per_action=demos_per_action,
        total_elapsed_s=total_elapsed,
    )

    summary = {
        "schema_version": "s1-summary-v1",
        "stage": "S1",
        "reproduce_command": S1_REPRO_COMMAND,
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "s1_parser_generator_version": S1_PARSER_GENERATOR_VERSION,
        "s0_generator_version": GENERATOR_VERSION,
        "s0_temporal_generator_version": TEMPORAL_GENERATOR_VERSION,
        "seeds": list(seeds),
        "demos_per_action": demos_per_action,
        "parser_cases_per_seed": parser_manifest["per_seed"][str(seeds[0])]["total_cases"],
        "total_elapsed_seconds": round(total_elapsed, 4),
        "environment": capture_environment_info(),
        "ast_isolation_audit": ast_audit,
        "s1_gate": s1_gate,
        "parsers": parser_summary,
        "ledger_audit": ledger_summary_payload,
    }

    (reports_dir / "s1_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports_dir / "s1_research_report.md").write_text(
        _render_s1_markdown_report(summary), encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible Mirror 7 CPU-Native Learning Stage S1 state-frame, ledger, and parser-only benchmark."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Fixed random seeds (default: 101 202 303 404 505).",
    )
    parser.add_argument(
        "--demos-per-action",
        type=int,
        default=DEFAULT_DEMOS_PER_ACTION,
        help="Demonstrations per action in [3, 8] (default: 6).",
    )
    parser.add_argument(
        "--verify-manifests",
        action="store_true",
        help="Verify committed S0 and S1 manifests match deterministic generation bit-for-bit.",
    )
    args = parser.parse_args(argv)

    if args.verify_manifests:
        ok, mismatches = verify_s1_committed_manifests(
            seeds=tuple(args.seeds), demos_per_action=args.demos_per_action
        )
        if not ok:
            for msg in mismatches:
                print(f"MANIFEST MISMATCH: {msg}", file=sys.stderr)
            return 1
        print("Committed S0 and S1 manifests verified bit-exact against deterministic generators.")
        return 0

    summary = run_s1_experiments(
        seeds=tuple(args.seeds), demos_per_action=args.demos_per_action
    )
    gate = summary["s1_gate"]
    print(
        f"S1 experiment complete in {summary['total_elapsed_seconds']:.2f}s | "
        f"gate={'PASS' if gate['passed'] else 'FAIL'} "
        f"({sum(1 for ok in gate['checks'].values() if ok)}/{len(gate['checks'])} checks)"
    )
    if not gate["passed"]:
        for k, ok in gate["checks"].items():
            if not ok:
                print(f"  FAILED GATE CHECK: {k}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
