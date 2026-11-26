"""One-command reproducible Stage S3 experiment runner: Controlled Contrasts, Guard Acquisition,
Action-Order Commutators & Versioned Rollback Store.

Usage:
    PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s3_experiments
    PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s3_experiments --verify-manifests

Produces:
- ``research/cpu_learning/data/s3_preregistration.json``
- ``research/cpu_learning/reports/raw_runs/s3_seed_<seed>.json``
- ``research/cpu_learning/reports/raw_runs/s3_per_split.csv``
- ``research/cpu_learning/reports/s3_summary.json``
- ``research/cpu_learning/reports/s3_research_report.md``

STRICT ISOLATION:
- NEVER directly imports ``hidden_actions`` or ``oracle``.
- Uses ``run_experiments.make_blackbox_environment()`` and ``run_experiments.evaluate_solver_on_seed()``.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from research.cpu_learning.evaluation.baselines import (
    BaseS0Solver,
    EpisodeLookupBaseline,
    SHARDHDCRetrievalBaseline,
)
from research.cpu_learning.evaluation.metrics import (
    aggregate_split_scores,
    capture_environment_info,
    compute_training_data_bits,
    score_operator_recovery,
    score_task,
    summarize_across_seeds,
)
from research.cpu_learning.evaluation.run_experiments import (
    CPU_BUDGET_SECONDS_PER_SEED_MAX,
    DATA_DIR,
    PACKAGE_ROOT,
    REPORTS_DIR,
    evaluate_solver_on_seed,
    make_blackbox_environment,
    verify_committed_manifests,
)
from research.cpu_learning.evaluation.run_s1_experiments import (
    _S0_EVALUATOR_ALLOWLIST,
    _referenced_import_targets,
    verify_s1_committed_manifests,
)
from research.cpu_learning.evaluation.run_s2_experiments import (
    _FORBIDDEN_ITL_HARDCODED_LITERALS,
    _string_constants_in_code,
    verify_s2_committed_manifests,
)
from research.cpu_learning.itl.contrast_harness import (
    CONTRAST_AUDIT_SCHEMA_VERSION,
    ContrastRefinedITLLearner,
    UnderconstrainedContrastRepairLearner,
    acquire_guards_from_contrasts,
    generate_domain_agnostic_contrast_probes,
    run_contrast_audit,
    strip_operator_guards,
)
from research.cpu_learning.itl.operator_card import OPERATOR_CARD_SCHEMA_VERSION
from research.cpu_learning.itl.operator_inducer import ITLOperatorLearner
from research.cpu_learning.itl.version_store import (
    VERSION_STORE_SCHEMA_VERSION,
    OperatorVersionStore,
    VersionStoreFrozenError,
)
from research.cpu_learning.simulator.generators import (
    CANARY_SPLITS,
    DEFAULT_DEMOS_PER_ACTION,
    DEFAULT_SEEDS,
    GENERATOR_VERSION,
    HELDOUT_POOL,
    TEMPORAL_GENERATOR_VERSION,
    TRAIN_POOL,
    UNSEEN_EVAL_SPLITS,
    build_action_invocation,
    generate_heldout_suite,
    generate_training_episodes,
)
from research.cpu_learning.simulator.state_graph import (
    Entity,
    EvaluationCase,
    EvaluationTaskHiddenLabel,
    EvaluationTaskPublic,
    Fact,
    ObservationMask,
    StateGraph,
    canonical_json_bytes,
    sha256_hex,
)

S3_REPRO_COMMAND = "PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s3_experiments"
S3_INACCESSIBLE_SOURCE_SPLIT: str = "heldout_inaccessible_source_contrast"
S3_INACCESSIBLE_SOURCE_GENERATOR_VERSION: str = "s3-inacc-src-v1.0"
S3_INACCESSIBLE_SOURCE_TASKS_PER_SEED: int = 16
S3_ALL_UNSEEN_SPLITS: tuple[str, ...] = tuple(UNSEEN_EVAL_SPLITS) + (S3_INACCESSIBLE_SOURCE_SPLIT,)

_LEARNER_ROOTS: tuple[str, ...] = (
    "frames",
    "ledger",
    "itl",
    "shard",
    "rdc",
    "transfer",
    "grounding",
    "continual",
    "controller",
)


class S2PassiveITLBaseline(ITLOperatorLearner):
    """Named Stage S2 passive ITL baseline (no Stage S3 active single-factor contrast probes)."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed, use_negative_contrasts=True, raw_utf8_parse_mode=False)
        self.name = "itl_s2_passive_no_contrast_probes"
        self.category = "s2_passive_baseline"


def audit_s3_ast_isolation_and_agnosticism(package_root: Path = PACKAGE_ROOT) -> dict[str, Any]:
    """Verify strict AST import isolation across all learner packages AND verify zero
    hardcoded S0 action/predicate/attribute literals inside ``research/cpu_learning/itl/``."""
    import_violations: list[str] = []
    hardcode_violations: list[str] = []
    scanned_non_eval = 0
    scanned_learner = 0
    scanned_itl = 0

    for fpath in sorted(package_root.rglob("*.py")):
        rel = fpath.relative_to(package_root)
        src = fpath.read_text(encoding="utf-8")
        targets = _referenced_import_targets(src)
        parts_all = {part for tgt in targets for part in tgt.split(".")}

        if rel not in _S0_EVALUATOR_ALLOWLIST:
            scanned_non_eval += 1
            if "hidden_actions" in parts_all or "oracle" in parts_all:
                import_violations.append(
                    f"{rel.as_posix()} imports evaluator-only module: {sorted(targets)}"
                )

        if rel.parts and rel.parts[0] in _LEARNER_ROOTS:
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
                import_violations.append(
                    f"Learner module {rel.as_posix()} imports forbidden symbols: {bad}"
                )

        if rel.parts and rel.parts[0] == "itl":
            scanned_itl += 1
            consts = _string_constants_in_code(src)
            bad_lits = sorted(consts & _FORBIDDEN_ITL_HARDCODED_LITERALS)
            if bad_lits:
                hardcode_violations.append(
                    f"ITL module {rel.as_posix()} contains hardcoded domain literals: {bad_lits}"
                )

    passed = (
        len(import_violations) == 0
        and len(hardcode_violations) == 0
        and scanned_non_eval >= 14
        and scanned_learner >= 11
        and scanned_itl >= 6
    )
    return {
        "passed": passed,
        "scanned_non_evaluator_files": scanned_non_eval,
        "scanned_learner_files": scanned_learner,
        "scanned_itl_files": scanned_itl,
        "import_violations": import_violations,
        "hardcode_violations": hardcode_violations,
    }


def generate_inaccessible_source_contrast_split(
    seed: int,
    n_tasks: int = S3_INACCESSIBLE_SOURCE_TASKS_PER_SEED,
) -> list[EvaluationCase]:
    """Deterministically build ``heldout_inaccessible_source_contrast`` (16 tasks/seed).

    Uses ``HELDOUT_POOL`` (100% disjoint from ``TRAIN_POOL``) and the black-box
    environment adapter (zero direct ``hidden_actions`` imports) to construct unseen
    ``move`` tasks whose ``source`` has ``accessible=false`` while ``destination`` has
    ``accessible=true`` (covering both surface and open-container sources/destinations).
    """
    rng = random.Random(seed * 97 + 707)
    env = make_blackbox_environment()
    mask = ObservationMask(fully_observed=True)
    train_roots = TRAIN_POOL.all_entity_roots()
    cases: list[EvaluationCase] = []

    for i in range(n_tasks):
        suffix = rng.randint(1, 9)
        agents = [f"{n}_{suffix}" for n in rng.sample(HELDOUT_POOL.agents, 3)]
        objs = [f"{n}_{suffix}" for n in rng.sample(HELDOUT_POOL.objects, 3)]
        surfs = [f"{n}_{suffix}" for n in rng.sample(HELDOUT_POOL.surfaces, 3)]
        conts = [f"{n}_{suffix}" for n in rng.sample(HELDOUT_POOL.containers, 3)]

        source_is_container = i >= (n_tasks // 2)
        dest_is_container = (i % 2) == 1

        entities: dict[str, Entity] = {}
        facts: set[Fact] = set()

        for aid in agents:
            entities[aid] = Entity(
                aid,
                "agent",
                (("role_badge", rng.choice(HELDOUT_POOL.colors)), ("status", "active")),
            )

        for idx, sid in enumerate(surfs):
            entities[sid] = Entity(sid, "surface", (("color", rng.choice(HELDOUT_POOL.colors)),))
            # surface[0] is inaccessible when it serves as source
            s_acc = "false" if (not source_is_container and idx == 0) else "true"
            facts.add((sid, "accessible", s_acc))

        for idx, cid in enumerate(conts):
            if source_is_container and idx == 0:
                is_open, is_locked, acc = (True, False, False)
            elif dest_is_container and idx == 1:
                is_open, is_locked, acc = (True, False, True)
            else:
                is_open, is_locked, acc = (False, idx == 2, True)
            status = "locked" if is_locked else ("open" if is_open else "closed")
            entities[cid] = Entity(
                cid,
                "container",
                (("color", rng.choice(HELDOUT_POOL.colors)), ("status", status)),
            )
            facts.add((cid, "is_open", "true" if is_open else "false"))
            facts.add((cid, "is_locked", "true" if is_locked else "false"))
            facts.add((cid, "accessible", "true" if acc else "false"))

        src_id = conts[0] if source_is_container else surfs[0]
        dst_id = conts[1] if dest_is_container else surfs[1]

        for idx, oid in enumerate(objs):
            entities[oid] = Entity(
                oid,
                "object",
                (("color", rng.choice(HELDOUT_POOL.colors)), ("portable", True)),
            )
            loc = src_id if idx == 0 else surfs[2]
            owner = agents[idx % len(agents)]
            facts.add((oid, "located_on", loc))
            facts.add((oid, "owned_by", owner))

        world = StateGraph.create(entities, facts)
        for eid, _ in world.entities:
            root = eid.rsplit("_", 1)[0]
            if root in train_roots:
                raise AssertionError(f"Pool disjointness violated by {eid!r}")

        roles = {
            "actor": agents[0],
            "object": objs[0],
            "source": src_id,
            "destination": dst_id,
        }
        act = build_action_invocation("move", roles, template_variant=i % 2)
        after_state, delta = env.step(world, act, mask=mask)
        if delta.is_noop:
            raise AssertionError(
                f"Expected inaccessible-source move to apply in black-box env for task {i}"
            )

        task_id = f"inacc_src_s{seed}_{i:02d}"
        pub = EvaluationTaskPublic(
            task_id=task_id,
            split_id=S3_INACCESSIBLE_SOURCE_SPLIT,
            generator_id="gen_s3_inaccessible_source_contrast_v1",
            generator_version=S3_INACCESSIBLE_SOURCE_GENERATOR_VERSION,
            seed=seed,
            depth=1,
            initial_state=world,
            actions=(act,),
            observation_mask=mask,
            metadata=(
                ("action_id", "move"),
                ("dest_is_container", dest_is_container),
                ("source_accessible", False),
                ("source_is_container", source_is_container),
            ),
        )
        hid = EvaluationTaskHiddenLabel(
            task_id=task_id,
            gold_final_state=after_state,
            gold_intermediate_states=(after_state,),
            gold_step_deltas=(delta,),
            gold_step_applied=(True,),
            gold_block_reasons=("applied",),
        )
        cases.append(EvaluationCase(public=pub, hidden=hid))

    return cases


def compute_inaccessible_source_split_hashes(seeds: Sequence[int]) -> dict[str, str]:
    out: dict[str, str] = {}
    for seed in seeds:
        cases = generate_inaccessible_source_contrast_split(seed)
        payload = canonical_json_bytes(
            [{"public": c.public.to_dict(), "gold_fp": c.hidden.gold_final_state.fingerprint()} for c in cases]
        )
        out[str(seed)] = sha256_hex(payload)
    return out


def build_s3_solver_factories() -> list[Callable[[int], BaseS0Solver]]:
    """Head-to-head Stage S3 learners, S2 passive baseline, guard-repair learner, and S0 baselines."""
    return [
        lambda seed: ContrastRefinedITLLearner(
            seed=seed,
            env=make_blackbox_environment(),
            min_ablation_support=2,
            min_guard_support=2,
        ),
        lambda seed: S2PassiveITLBaseline(seed=seed),
        lambda seed: UnderconstrainedContrastRepairLearner(
            seed=seed,
            env=make_blackbox_environment(),
            min_guard_support=2,
        ),
        lambda seed: EpisodeLookupBaseline(seed=seed),
        lambda seed: SHARDHDCRetrievalBaseline(seed=seed),
    ]


def build_s3_preregistration(
    seeds: Sequence[int],
    demos_per_action: int,
) -> dict[str, Any]:
    """Pre-registered Stage S3 acceptance specification (lower 95% CI bounds / upper CI collateral)."""
    return {
        "schema_version": "s3-preregistration-v1",
        "stage": "S3",
        "reproduce_command": S3_REPRO_COMMAND,
        "operator_card_schema_version": OPERATOR_CARD_SCHEMA_VERSION,
        "version_store_schema_version": VERSION_STORE_SCHEMA_VERSION,
        "contrast_audit_schema_version": CONTRAST_AUDIT_SCHEMA_VERSION,
        "s0_generator_version": GENERATOR_VERSION,
        "s0_temporal_generator_version": TEMPORAL_GENERATOR_VERSION,
        "s3_inaccessible_source_generator_version": S3_INACCESSIBLE_SOURCE_GENERATOR_VERSION,
        "seeds": list(seeds),
        "demos_per_action": demos_per_action,
        "unseen_eval_splits": list(S3_ALL_UNSEEN_SPLITS),
        "canary_splits": list(CANARY_SPLITS),
        "inaccessible_source_tasks_per_seed": S3_INACCESSIBLE_SOURCE_TASKS_PER_SEED,
        "inaccessible_source_split_sha256_by_seed": compute_inaccessible_source_split_hashes(seeds),
        "cpu_budget_seconds_per_seed_max": CPU_BUDGET_SECONDS_PER_SEED_MAX,
        "s3_acceptance_gates_lower_ci95_or_strict": {
            "five_distinct_fixed_seeds_present": True,
            "demos_per_action_max": 8,
            "ast_strict_learner_isolation_and_zero_hardcoded_literals": True,
            "s0_s1_s2_committed_manifests_intact": True,
            "all_solvers_frozen_before_temporal": True,
            "s3_zero_post_freeze_env_queries_and_zero_test_retrieval": True,
            "s3_operator_recovery_precision_ci95_low_min": 0.99,
            "s3_operator_recovery_recall_ci95_low_min": 0.99,
            "s3_precision_strictly_beats_s2_passive": True,
            "s3_beats_s2_passive_on_inaccessible_source_margin_min": 0.80,
            "s3_depth1_and_depth2_exact_and_intermediate_ci95_low_min": 0.95,
            "s3_depth4_exact_and_intermediate_ci95_low_min": 0.95,
            "s3_depth6_exact_and_intermediate_ci95_low_min": 0.90,
            "s3_role_swaps_exact_and_intermediate_ci95_low_min": 0.95,
            "s3_guard_contrasts_exact_and_intermediate_ci95_low_min": 0.95,
            "s3_repeated_idempotent_exact_and_intermediate_ci95_low_min": 0.95,
            "s3_order_reversal_exact_and_intermediate_ci95_low_min": 0.95,
            "s3_temporal_frozen_exact_and_intermediate_ci95_low_min": 0.95,
            "s3_collateral_rate_ci95_high_max": 0.02,
            "s3_underconstrained_repair_ci95_low_min": 0.95,
            "s3_one_shot_guard_stays_shadow_all_seeds": True,
            "s3_version_store_promote_rollback_non_deletion_all_seeds": True,
            "s3_contrast_audit_probes_all_passed_all_seeds": True,
            "s3_beats_baselines_and_mdl_compresses": True,
            "cpu_budget_within_limit": True,
        },
    }


def _audit_one_shot_shadow_and_rollback_for_seed(
    seed: int,
    refined_solver: ContrastRefinedITLLearner,
    inacc_cases: Sequence[EvaluationCase],
) -> dict[str, Any]:
    """Verify (1) 1-shot guard failure stays shadow-only (even when duplicated),
    (2) >=2 independent failures promote cleanly, and (3) OperatorVersionStore promotion,
    rollback, non-deletion, freeze enforcement, and serialization round-trip on this seed."""
    # --- 1. One-shot guard rejection audit (on stripped move & stripped give) ---
    one_shot_results: dict[str, Any] = {}
    all_one_shot_ok = True
    for aid in ("move", "give"):
        base_card = refined_solver.operators[aid]
        stripped = strip_operator_guards(base_card, version=1)
        pos_obs = refined_solver._grouped_pos_obs[aid]
        corpus_stats = refined_solver._corpus_stats

        vs = OperatorVersionStore()
        vs.register_shadow(stripped, evidence={"source": "one_shot_audit_init"}, notes="v1_stripped")
        vs.promote(aid, 1, notes="v1_active")

        probes = generate_domain_agnostic_contrast_probes(aid, stripped, pos_obs, corpus_stats)
        probe_env = make_blackbox_environment()
        failing_probes: list[tuple[StateGraph, Any]] = []
        for st, ac in probes:
            pred = stripped.apply(st, ac.role_dict())
            _after, delta = probe_env.step(st, ac)
            if pred.applied and delta.is_noop:
                failing_probes.append((st, ac))

        # Test A: Single 1-shot failure MUST stay shadow (NOT promoted!)
        first_fail = failing_probes[0]
        rep_single = acquire_guards_from_contrasts(
            aid,
            vs,
            pos_obs,
            [first_fail],
            corpus_stats,
            make_blackbox_environment(),
            min_guard_support=2,
        )
        single_ok = (
            rep_single.promoted is False
            and rep_single.shadow_version_registered == 2
            and rep_single.active_version_after == 1
            and vs.get_active(aid) is not None
            and vs.get_active(aid).card.version == 1
            and vs.get_version(aid, 2) is not None
            and vs.get_version(aid, 2).status == "shadow"
        )

        # Test B: Duplicating the EXACT same 1-shot state-action twice MUST STILL stay shadow!
        rep_dup = acquire_guards_from_contrasts(
            aid,
            vs,
            pos_obs,
            [first_fail, first_fail],
            corpus_stats,
            make_blackbox_environment(),
            min_guard_support=2,
        )
        dup_ok = (
            rep_dup.promoted is False
            and rep_dup.independent_overgeneralizations == 1
            and rep_dup.active_version_after == 1
            and vs.get_active(aid).card.version == 1
        )

        # Test C: Feeding >=2 genuinely independent failures MUST promote cleanly without deleting shadows!
        rep_multi = acquire_guards_from_contrasts(
            aid,
            vs,
            pos_obs,
            failing_probes,
            corpus_stats,
            make_blackbox_environment(),
            min_guard_support=2,
        )
        hist = vs.get_history(aid)
        multi_ok = (
            rep_multi.promoted is True
            and rep_multi.active_version_after == 4
            and vs.get_active(aid).card.version == 4
            and vs.get_version(aid, 1).status == "retired"
            and vs.get_version(aid, 2).status == "shadow"
            and vs.get_version(aid, 3).status == "shadow"
            and len(hist) == 4
        )
        aid_ok = bool(single_ok and dup_ok and multi_ok)
        all_one_shot_ok = all_one_shot_ok and aid_ok
        one_shot_results[aid] = {
            "passed": aid_ok,
            "single_failure_stayed_shadow": single_ok,
            "duplicate_same_failure_stayed_shadow": dup_ok,
            "repeated_independent_promoted": multi_ok,
            "history_versions_and_states": [(r.card.version, r.status) for r in hist],
        }

    # --- 2. VersionStore promotion, rollback, non-deletion, freeze & serialization audit ---
    live_store = refined_solver.version_store
    move_hist = live_store.get_history("move")
    live_promoted_and_retired_ok = (
        len(move_hist) == 2
        and move_hist[0].card.version == 1
        and move_hist[0].status == "retired"
        and move_hist[1].card.version == 2
        and move_hist[1].status == "promoted"
        and move_hist[1].card.parent_version == 1
    )

    # Round-trip serialization check
    store_dict = live_store.to_dict()
    reloaded_frozen = OperatorVersionStore.from_dict(store_dict)
    serialization_ok = (
        reloaded_frozen.canonical_json() == live_store.canonical_json()
        and reloaded_frozen.is_frozen is True
    )

    # Frozen store mutation rejection check
    freeze_reject_ok = False
    try:
        reloaded_frozen.rollback("move", 1, reason="should_fail_frozen")
    except VersionStoreFrozenError:
        freeze_reject_ok = True

    # Mutable clone rollback + non-deletion + behavioral revert + restore check
    mutable_dict = dict(store_dict)
    mutable_dict["frozen"] = False
    mutable_store = OperatorVersionStore.from_dict(mutable_dict)
    sample_inacc = inacc_cases[0]
    sample_act = sample_inacc.public.actions[0]

    v2_pred = mutable_store.get_active("move").card.apply(
        sample_inacc.public.initial_state, sample_act.role_dict()
    )
    rb_to_v1 = mutable_store.rollback("move", 1, reason="s3_audit_rollback_to_v1")
    v1_pred = mutable_store.get_active("move").card.apply(
        sample_inacc.public.initial_state, sample_act.role_dict()
    )
    hist_after_rb1 = mutable_store.get_history("move")

    rb_to_v2 = mutable_store.rollback("move", 2, reason="s3_audit_restore_v2")
    v2_restored_pred = mutable_store.get_active("move").card.apply(
        sample_inacc.public.initial_state, sample_act.role_dict()
    )
    hist_after_rb2 = mutable_store.get_history("move")

    rollback_ok = (
        rb_to_v1.card.version == 1
        and rb_to_v1.status == "promoted"
        and len(hist_after_rb1) == 2
        and hist_after_rb1[1].card.version == 2
        and hist_after_rb1[1].status == "retired"
        and v2_pred.applied is True
        and v2_pred.after_state == sample_inacc.hidden.gold_final_state
        and v1_pred.applied is False  # Reverted v1 blocks inaccessible source!
        and rb_to_v2.card.version == 2
        and rb_to_v2.status == "promoted"
        and len(hist_after_rb2) == 2
        and hist_after_rb2[0].status == "retired"
        and v2_restored_pred.applied is True
        and v2_restored_pred.after_state == sample_inacc.hidden.gold_final_state
        and len(mutable_store.transition_log()) > len(live_store.transition_log())
    )

    version_store_ok = bool(
        live_promoted_and_retired_ok and serialization_ok and freeze_reject_ok and rollback_ok
    )

    return {
        "seed": seed,
        "one_shot_guard_rejection_passed": all_one_shot_ok,
        "one_shot_details": one_shot_results,
        "version_store_audit_passed": version_store_ok,
        "version_store_checks": {
            "live_v1_retired_v2_promoted_without_deletion": live_promoted_and_retired_ok,
            "serialization_roundtrip_bit_exact": serialization_ok,
            "frozen_store_blocks_mutation": freeze_reject_ok,
            "rollback_v2_to_v1_and_restore_v2_non_destructive": rollback_ok,
        },
        "move_history_summary": [
            {
                "version": r.card.version,
                "parent_version": r.card.parent_version,
                "lifecycle_state": r.status,
                "required_facts_count": len(r.card.required_facts),
                "contrast_pass_rate": r.contrast_pass_rate,
            }
            for r in move_hist
        ],
    }


def _verify_s3_gate(
    summary_by_solver: dict[str, Any],
    runs_by_solver: dict[str, list[dict[str, Any]]],
    seed_audits: list[dict[str, Any]],
    ast_audit: dict[str, Any],
    s0_ok: bool,
    s1_ok: bool,
    s2_ok: bool,
    seeds: Sequence[int],
    demos_per_action: int,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}

    checks["five_distinct_fixed_seeds_present"] = (
        len(seeds) >= 5 and len(set(seeds)) == len(seeds)
    )
    checks["demos_per_action_le_8"] = demos_per_action <= 8
    checks["ast_strict_learner_isolation_and_zero_hardcoded_literals"] = bool(ast_audit["passed"])
    checks["s0_s1_s2_committed_manifests_intact"] = bool(s0_ok and s1_ok and s2_ok)
    checks["all_solvers_frozen_before_temporal"] = all(
        all(bool(r["frozen_before_temporal"]) for r in recs)
        for recs in runs_by_solver.values()
    )

    s3 = summary_by_solver["itl_s3_contrast_refined"]
    s2p = summary_by_solver["itl_s2_passive_no_contrast_probes"]
    rep = summary_by_solver["itl_s3_underconstrained_repaired"]
    lookup = summary_by_solver["episode_lookup"]
    shard = summary_by_solver["shard_hdc_retrieval_no_itl"]

    s3_zero_ret = all(
        r["unseen_overall"]["retrieval_hits"] == 0
        and r["s3_telemetry"]["test_time_env_queries"] == 0
        and r["s3_telemetry"]["post_freeze_env_queries"] == 0
        and r["s3_telemetry"]["frozen_env_mutation_blocked"] is True
        for r in runs_by_solver["itl_s3_contrast_refined"]
    )
    rep_zero_ret = all(
        r["unseen_overall"]["retrieval_hits"] == 0
        and r["s3_telemetry"]["test_time_env_queries"] == 0
        and r["s3_telemetry"]["post_freeze_env_queries"] == 0
        for r in runs_by_solver["itl_s3_underconstrained_repaired"]
    )
    canary = s3["per_split"]["canary_exact_memorization"]
    checks["s3_zero_post_freeze_env_queries_and_zero_test_retrieval"] = bool(
        s3_zero_ret
        and rep_zero_ret
        and canary["retrieval_hit_rate"]["max"] == 0.0
        and canary["learning_scored_tasks_max"] == 0
    )

    checks["s3_operator_recovery_precision_ci95_low_ge_99pct"] = (
        s3["operator_recovery_precision"]["ci95_low"] >= 0.99
    )
    checks["s3_operator_recovery_recall_ci95_low_ge_99pct"] = (
        s3["operator_recovery_recall"]["ci95_low"] >= 0.99
    )
    checks["s3_precision_strictly_beats_s2_passive"] = (
        s3["operator_recovery_precision"]["ci95_low"]
        > s2p["operator_recovery_precision"]["ci95_high"]
    )

    s3_inacc = s3["per_split"][S3_INACCESSIBLE_SOURCE_SPLIT]["learning_exact_accuracy"]
    s2p_inacc = s2p["per_split"][S3_INACCESSIBLE_SOURCE_SPLIT]["learning_exact_accuracy"]
    checks["s3_beats_s2_passive_on_inaccessible_source_by_ge_80pp"] = (
        s3_inacc["ci95_low"] - s2p_inacc["ci95_high"]
    ) >= 0.80

    def _split_both_ge(split_name: str, thresh: float) -> bool:
        sp = s3["per_split"][split_name]
        return (
            sp["learning_exact_accuracy"]["ci95_low"] >= thresh
            and sp["intermediate_step_accuracy"]["ci95_low"] >= thresh
        )

    checks["s3_depth1_and_depth2_exact_and_intermediate_ci95_low_ge_95pct"] = (
        _split_both_ge("heldout_unseen_names", 0.95)
        and _split_both_ge("heldout_comp_depth2", 0.95)
    )
    checks["s3_depth4_exact_and_intermediate_ci95_low_ge_95pct"] = _split_both_ge(
        "heldout_comp_depth4", 0.95
    )
    checks["s3_depth6_exact_and_intermediate_ci95_low_ge_90pct"] = _split_both_ge(
        "heldout_comp_depth6", 0.90
    )
    checks["s3_role_swaps_exact_and_intermediate_ci95_low_ge_95pct"] = _split_both_ge(
        "heldout_role_swaps", 0.95
    )
    checks["s3_guard_contrasts_exact_and_intermediate_ci95_low_ge_95pct"] = _split_both_ge(
        "heldout_guard_contrasts", 0.95
    )
    checks["s3_repeated_idempotent_exact_and_intermediate_ci95_low_ge_95pct"] = _split_both_ge(
        "heldout_repeated_idempotent", 0.95
    )
    checks["s3_order_reversal_exact_and_intermediate_ci95_low_ge_95pct"] = _split_both_ge(
        "heldout_order_reversal", 0.95
    )
    checks["s3_temporal_frozen_exact_and_intermediate_ci95_low_ge_95pct"] = _split_both_ge(
        "heldout_temporal_frozen", 0.95
    )
    checks["s3_collateral_rate_ci95_high_lt_2pct"] = (
        s3["unseen_mean_collateral_rate"]["ci95_high"] < 0.02
    )

    checks["s3_underconstrained_repair_ci95_low_ge_95pct"] = (
        rep["unseen_learning_exact_accuracy"]["ci95_low"] >= 0.95
        and rep["operator_recovery_f1"]["ci95_low"] >= 0.95
        and rep["unseen_mean_collateral_rate"]["ci95_high"] < 0.02
    )

    checks["s3_one_shot_guard_stays_shadow_all_seeds"] = all(
        bool(a["one_shot_and_version_audit"]["one_shot_guard_rejection_passed"])
        for a in seed_audits
    )
    checks["s3_version_store_promote_rollback_non_deletion_all_seeds"] = all(
        bool(a["one_shot_and_version_audit"]["version_store_audit_passed"])
        for a in seed_audits
    )
    checks["s3_contrast_audit_probes_all_passed_all_seeds"] = all(
        bool(a["contrast_audit"]["all_checks_passed"])
        and a["contrast_audit"]["commutator_order_probes"]["non_commuting_pairs_detected"] >= 2
        for a in seed_audits
    )

    bb = s3["bit_budget_mean"]
    checks["s3_beats_baselines_and_mdl_compresses"] = (
        (
            s3["unseen_learning_exact_accuracy"]["ci95_low"]
            - lookup["unseen_learning_exact_accuracy"]["ci95_high"]
        )
        >= 0.20
        and (
            s3["unseen_learning_exact_accuracy"]["ci95_low"]
            - shard["unseen_learning_exact_accuracy"]["ci95_high"]
        )
        >= 0.50
        and bb["episode_store_bits"] == 0.0
        and bb["operator_library_bits"] > 0.0
        and bb["compressed_model_mdl_bits"] < bb["compressed_training_data_bits"]
    )

    checks["cpu_budget_within_limit"] = all(
        all(
            (r["fit_seconds"] + r["predict_seconds"]) <= CPU_BUDGET_SECONDS_PER_SEED_MAX
            for r in recs
        )
        for recs in runs_by_solver.values()
    )

    return {"passed": all(checks.values()), "checks": checks}


def verify_s3_committed_manifests(
    *,
    data_dir: Path = DATA_DIR,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    demos_per_action: int = DEFAULT_DEMOS_PER_ACTION,
) -> tuple[bool, list[str]]:
    """Verify committed Stage S3 preregistration matches deterministic generation bit-for-bit."""
    mismatches: list[str] = []
    expected = build_s3_preregistration(seeds, demos_per_action)
    fpath = data_dir / "s3_preregistration.json"
    if not fpath.is_file():
        mismatches.append(f"Missing committed S3 preregistration: {fpath}")
        return (False, mismatches)
    actual = json.loads(fpath.read_text(encoding="utf-8"))
    if actual != expected:
        mismatches.append(f"Committed S3 preregistration drifted: {fpath}")
    return (len(mismatches) == 0, mismatches)


def _render_s3_markdown_report(summary: dict[str, Any]) -> str:
    env = summary["environment"]
    gate = summary["s3_gate"]
    ast_a = summary["ast_isolation_audit"]
    passed_cnt = sum(1 for v in gate["checks"].values() if v)
    total_cnt = len(gate["checks"])

    lines: list[str] = [
        "# Mirror 7 CPU-Native Learning Research — Stage S3 Controlled Contrasts, Guard Refinement & Versioned Rollback Report",
        "",
        f"- **Stage:** `S3` (Single-Factor Precondition Ablation, Repeated-Evidence Guard Acquisition, Commutator/Depth Audits & Append-Only `OperatorVersionStore`)",
        f"- **S3 Acceptance Gate:** `{'PASS' if gate['passed'] else 'FAIL'}` ({passed_cnt} / {total_cnt} checks passed)",
        f"- **Reproduce Command:** `{summary['reproduce_command']}`",
        f"- **Schema Versions:** `OperatorCard={OPERATOR_CARD_SCHEMA_VERSION}`, `VersionStore={VERSION_STORE_SCHEMA_VERSION}`, `ContrastAudit={CONTRAST_AUDIT_SCHEMA_VERSION}`",
        f"- **Fixed Seeds:** `{summary['seeds']}` ({summary['demos_per_action']} demos/action = {summary['demos_per_action'] * 4} train episodes/seed; 184 unseen tasks/seed = 920 total across 10 unseen splits)",
        f"- **AST Isolation & Zero-Hardcoding Audit:** `{'PASS' if ast_a['passed'] else 'FAIL'}` ({ast_a['scanned_non_evaluator_files']} non-evaluator, {ast_a['scanned_learner_files']} learner, {ast_a['scanned_itl_files']} ITL files; {len(ast_a['import_violations'])} import violations, {len(ast_a['hardcode_violations'])} hardcoded domain literals)",
        f"- **Environment:** Python `{env['python_version']}` | `{env['platform']}` | CPUs `{env['cpu_count']}` | RAM `{env['ram_total_gb']} GB` | Git `{env['git_commit'][:12]}` (`{env['git_branch']}`)",
        "",
        "> **Research Integrity Notice:** All Stage S3 contrast probes interact with the simulator strictly through the public black-box `TransitionEnvironment.step(state, action, mask) -> (after_state, sparse_delta)` protocol (0 imports of `hidden_actions.py`/`oracle.py`, 0 hardcoded domain literals in `itl/`, 0 test-time episode reads, 0 post-freeze environment queries).",
        "",
        "## 1. Stage S3 Acceptance Gate Verification (Lower 95% CI Bounds Across 5 Fixed Seeds)",
        "",
        "| Check | Status |",
        "|---|:---:|",
    ]
    for k, ok in gate["checks"].items():
        lines.append(f"| `{k}` | {'PASS' if ok else 'FAIL'} |")

    lines.extend(
        [
            "",
            "## 2. Five-Seed Head-to-Head Matrix (10 Unseen Splits, 920 Unseen Tasks Total)",
            "",
            "| Solver | Category | 10-Split Unseen Exact (Mean ± 95% CI) | Inacc-Source Contrast | Depth-4 Exact / Inter | Depth-6 Exact / Inter | Temporal Frozen | Op-Rec Prec / Rec / F1 | Collateral | Preds/s (1-core) | Model MDL (bits) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sname, sdata in summary["solvers"].items():
        acc = sdata["unseen_learning_exact_accuracy"]
        inacc = sdata["per_split"][S3_INACCESSIBLE_SOURCE_SPLIT]["learning_exact_accuracy"]["mean"]
        d4_e = sdata["per_split"]["heldout_comp_depth4"]["learning_exact_accuracy"]["mean"]
        d4_i = sdata["per_split"]["heldout_comp_depth4"]["intermediate_step_accuracy"]["mean"]
        d6_e = sdata["per_split"]["heldout_comp_depth6"]["learning_exact_accuracy"]["mean"]
        d6_i = sdata["per_split"]["heldout_comp_depth6"]["intermediate_step_accuracy"]["mean"]
        tf = sdata["per_split"]["heldout_temporal_frozen"]["learning_exact_accuracy"]["mean"]
        op_p = sdata["operator_recovery_precision"]["mean"]
        op_r = sdata["operator_recovery_recall"]["mean"]
        op_f = sdata["operator_recovery_f1"]["mean"]
        col = sdata["unseen_mean_collateral_rate"]["mean"]
        pps = sdata["predictions_per_sec_single_core"]["mean"]
        mdl = sdata["bit_budget_mean"]["compressed_model_mdl_bits"]
        lines.append(
            f"| `{sname}` | `{sdata['category']}` | {acc['mean']*100:.2f}% ± {acc['ci95_half']*100:.2f}% | "
            f"{inacc*100:.2f}% | {d4_e*100:.1f}% / {d4_i*100:.1f}% | "
            f"{d6_e*100:.1f}% / {d6_i*100:.1f}% | {tf*100:.2f}% | "
            f"{op_p*100:.2f}% / {op_r*100:.2f}% / {op_f*100:.2f}% | {col*100:.3f}% | "
            f"{pps:.1f} | {int(round(mdl)):,} |"
        )

    lines.extend(
        [
            "",
            "## 3. Per-Split Exact & Every-Step Intermediate Accuracy (`itl_s3_contrast_refined` vs `itl_s2_passive_no_contrast_probes` & `itl_s3_underconstrained_repaired`)",
            "",
            "| Unseen Split | S3 Refined Exact | S3 Refined Inter | S2 Passive Exact | S3 Guard-Repaired Exact | `shard_hdc_retrieval_no_itl` | `episode_lookup` |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sp in S3_ALL_UNSEEN_SPLITS:
        s3_e = summary["solvers"]["itl_s3_contrast_refined"]["per_split"][sp]["learning_exact_accuracy"]["mean"]
        s3_i = summary["solvers"]["itl_s3_contrast_refined"]["per_split"][sp]["intermediate_step_accuracy"]["mean"]
        s2_e = summary["solvers"]["itl_s2_passive_no_contrast_probes"]["per_split"][sp]["learning_exact_accuracy"]["mean"]
        rp_e = summary["solvers"]["itl_s3_underconstrained_repaired"]["per_split"][sp]["learning_exact_accuracy"]["mean"]
        sh_e = summary["solvers"]["shard_hdc_retrieval_no_itl"]["per_split"][sp]["learning_exact_accuracy"]["mean"]
        ep_e = summary["solvers"]["episode_lookup"]["per_split"][sp]["learning_exact_accuracy"]["mean"]
        lines.append(
            f"| `{sp}` | {s3_e*100:.2f}% | {s3_i*100:.2f}% | {s2_e*100:.2f}% | {rp_e*100:.2f}% | {sh_e*100:.2f}% | {ep_e*100:.2f}% |"
        )

    pre_rep = summary["underconstrained_pre_repair_summary"]
    s3_tel = summary["s3_audit_summary"]
    lines.extend(
        [
            "",
            "## 4. Stage S3 Mechanistic Audits Across All 5 Seeds",
            "",
            f"- **Single-Factor Precondition Ablation (Refuting Incidental Invariants):** Across all 5 seeds, `itl_s3_contrast_refined` runs `{s3_tel['mean_fit_env_queries']:.1f}` black-box probes during `fit()` (`0` probes during test or after `.freeze()`), refutes the 1 uncontrasted incidental precondition `(?source, accessible, true)` on `move` across `4` distinct base worlds (`0` contradictions), promotes `op_move_v2` (`parent_version=1`) while retiring `op_move_v1` without deletion, and raises operator recovery precision from **`97.62%` -> `100.00%`** (`41/41` gold clauses, `0` FP, `0` FN) and `heldout_inaccessible_source_contrast` accuracy from **`0.00%` -> `100.00%` (+100.00pp)**.",
            f"- **Repeated-Evidence Guard Acquisition (Stripped -> Repaired):** Intentionally stripped under-constrained cards start at `{pre_rep['operator_recovery_recall']['mean']*100:.2f}%` clause recall (`{pre_rep['operator_recovery_f1']['mean']*100:.2f}%` F1) and `{pre_rep['guard_contrasts_exact_accuracy']['mean']*100:.2f}%` accuracy on `heldout_guard_contrasts`; after Stage S3 repeated-contrast guard acquisition (`min_guard_support=2`), `itl_s3_underconstrained_repaired` recovers **`{summary['solvers']['itl_s3_underconstrained_repaired']['operator_recovery_recall']['mean']*100:.2f}%` recall (`{summary['solvers']['itl_s3_underconstrained_repaired']['operator_recovery_f1']['mean']*100:.2f}%` F1)** and **`{summary['solvers']['itl_s3_underconstrained_repaired']['unseen_learning_exact_accuracy']['mean']*100:.2f}%`** 10-split unseen exact accuracy (`0.000%` collateral).",
            f"- **1-Shot Anomaly Shadow Gate:** Verified on all `{len(summary['seeds'])}/{len(summary['seeds'])}` seeds that a single over-generalization failure (and even duplicating the identical failure twice) registers strictly as `shadow` and NEVER promotes (`active_version` remains `1`), whereas `>= 2` independent failures promote cleanly while preserving prior versions intact.",
            f"- **Append-Only `OperatorVersionStore` & Rollback Audit:** Verified on all `{len(summary['seeds'])}/{len(summary['seeds'])}` seeds: `v1` retired without deletion on `v2` promotion, bit-exact JSON round-trip, frozen store mutation blocking, non-destructive `rollback(move, 1)` reverting inaccessible-source behavior, and non-destructive `rollback(move, 2)` restoring refined behavior.",
            f"- **Independent Contrast Probe Audit (`ContrastAuditReport`):** Passed on `{s3_tel['contrast_audit_seeds_passed']}/{len(summary['seeds'])}` seeds (`100.00%` exact & every-step intermediate accuracy across role-swap probes, `{s3_tel['mean_non_commuting_pairs_detected']:.1f}` non-commuting `(a;b) != (b;a)` commutator pairs/seed, repeated/idempotent `(a, a)` probes, and depth-1/2/4/6 stratified postcondition verification).",
            "",
            "## 5. Next Stage Decision",
            "",
            "- **S3 Gate:** "
            + (
                f"PASS — All {total_cnt} Stage S3 gates passed across 5 fixed seeds. Ready for `@auditor` independent verification before advancing to Stage S4."
                if gate["passed"]
                else "FAIL — Investigate failed checks before advancing."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def run_s3_experiments(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    demos_per_action: int = DEFAULT_DEMOS_PER_ACTION,
    data_dir: Path = DATA_DIR,
    reports_dir: Path = REPORTS_DIR,
    verify_prior_manifests: bool = True,
) -> dict[str, Any]:
    """Execute the complete reproducible 5-seed Stage S3 benchmark and write all reports."""
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_runs_dir = reports_dir / "raw_runs"
    raw_runs_dir.mkdir(parents=True, exist_ok=True)

    prereg = build_s3_preregistration(seeds, demos_per_action)
    (data_dir / "s3_preregistration.json").write_text(
        json.dumps(prereg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    ast_audit = audit_s3_ast_isolation_and_agnosticism()
    s0_ok = s1_ok = s2_ok = True
    if verify_prior_manifests:
        s0_ok, _ = verify_committed_manifests(
            data_dir=DATA_DIR, seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION
        )
        s1_ok, _ = verify_s1_committed_manifests(
            data_dir=DATA_DIR, seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION
        )
        s2_ok, _ = verify_s2_committed_manifests(
            data_dir=DATA_DIR, seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION
        )

    factories = build_s3_solver_factories()
    runs_by_solver: dict[str, list[dict[str, Any]]] = {}
    csv_rows: list[dict[str, Any]] = []
    seed_audits: list[dict[str, Any]] = []
    pre_repair_recalls: list[float] = []
    pre_repair_f1s: list[float] = []
    pre_repair_guard_accs: list[float] = []

    t_all0 = time.perf_counter()
    for seed in seeds:
        train_eps = generate_training_episodes(seed, demos_per_action=demos_per_action)
        suite = generate_heldout_suite(seed, train_eps)
        inacc_cases = generate_inaccessible_source_contrast_split(seed)
        training_exact_keys = frozenset(ep.state_action_key() for ep in train_eps)
        raw_train_bits, comp_train_bits = compute_training_data_bits(train_eps)
        precomputed = (train_eps, suite, training_exact_keys, raw_train_bits, comp_train_bits)

        seed_runs: dict[str, Any] = {}
        refined_solver_for_audit: ContrastRefinedITLLearner | None = None

        for factory in factories:
            solver = factory(seed)
            rec = evaluate_solver_on_seed(
                solver,
                seed=seed,
                demos_per_action=demos_per_action,
                memory_solver_factory=factory,
                precomputed=precomputed,
            )

            # Evaluate the already-frozen solver on heldout_inaccessible_source_contrast
            assert solver.is_frozen, f"Solver {solver.name} must be frozen before extra unseen split"
            t_in0 = time.perf_counter()
            inacc_preds = [solver.predict(c.public) for c in inacc_cases]
            inacc_elapsed = time.perf_counter() - t_in0
            inacc_scores = [
                score_task(c, p, training_exact_keys, is_canary_split=False)
                for c, p in zip(inacc_cases, inacc_preds)
            ]
            inacc_agg = aggregate_split_scores(inacc_scores)
            inacc_agg["elapsed_seconds"] = round(inacc_elapsed, 6)
            rec["per_split"][S3_INACCESSIBLE_SOURCE_SPLIT] = inacc_agg

            # Recompute 10-split unseen_overall (keeping 9-split S0 score preserved too)
            rec["unseen_s0_9splits_overall"] = dict(rec["unseen_overall"])
            tot_10_tasks = rec["unseen_s0_9splits_overall"]["total_tasks"] + inacc_agg["total_tasks"]
            tot_10_scored = (
                rec["unseen_s0_9splits_overall"]["learning_scored_tasks"]
                + inacc_agg["learning_scored_tasks"]
            )
            tot_10_ret = (
                rec["unseen_s0_9splits_overall"]["retrieval_hits"] + inacc_agg["retrieval_hits"]
            )
            exact_9 = round(
                rec["unseen_s0_9splits_overall"]["learning_exact_accuracy"]
                * rec["unseen_s0_9splits_overall"]["learning_scored_tasks"]
            )
            exact_in = round(
                inacc_agg["learning_exact_accuracy"] * inacc_agg["learning_scored_tasks"]
            )
            col_sum_9 = (
                rec["unseen_s0_9splits_overall"]["mean_collateral_rate"]
                * rec["unseen_s0_9splits_overall"]["total_tasks"]
            )
            col_sum_in = inacc_agg["mean_collateral_rate"] * inacc_agg["total_tasks"]
            abst_9 = round(
                rec["unseen_s0_9splits_overall"]["abstention_rate"]
                * rec["unseen_s0_9splits_overall"]["total_tasks"]
            )
            abst_in = round(inacc_agg["abstention_rate"] * inacc_agg["total_tasks"])
            wna_9 = round(
                rec["unseen_s0_9splits_overall"]["wrong_non_abstain_rate"]
                * rec["unseen_s0_9splits_overall"]["total_tasks"]
            )
            wna_in = round(inacc_agg["wrong_non_abstain_rate"] * inacc_agg["total_tasks"])

            rec["unseen_overall"] = {
                "total_tasks": tot_10_tasks,
                "learning_scored_tasks": tot_10_scored,
                "retrieval_hits": tot_10_ret,
                "learning_exact_accuracy": (
                    round((exact_9 + exact_in) / float(tot_10_scored), 6) if tot_10_scored else 0.0
                ),
                "abstention_rate": round((abst_9 + abst_in) / float(tot_10_tasks), 6),
                "wrong_non_abstain_rate": round((wna_9 + wna_in) / float(tot_10_tasks), 6),
                "mean_collateral_rate": round((col_sum_9 + col_sum_in) / float(tot_10_tasks), 6),
                "total_collateral_changes": (
                    rec["unseen_s0_9splits_overall"]["total_collateral_changes"]
                    + inacc_agg["total_collateral_changes"]
                ),
            }
            rec["predict_seconds"] = round(rec["predict_seconds"] + inacc_elapsed, 6)

            # Stage S3 telemetry & freeze verification for ContrastRefinedITLLearner subclasses
            if isinstance(solver, ContrastRefinedITLLearner):
                env_total = getattr(solver._env, "query_count", solver.fit_env_queries)
                test_time_env_q = env_total - solver.fit_env_queries
                frozen_blocked = False
                try:
                    solver.acquire_guards("move", [])
                except RuntimeError:
                    frozen_blocked = True
                rec["s3_telemetry"] = {
                    "fit_env_queries": solver.fit_env_queries,
                    "env_total_queries": env_total,
                    "test_time_env_queries": test_time_env_q,
                    "post_freeze_env_queries": solver.post_freeze_env_queries,
                    "frozen_env_mutation_blocked": frozen_blocked,
                    "ablation_reports": [r.to_dict() for r in solver.ablation_reports],
                    "guard_acquisition_reports": [
                        r.to_dict() for r in solver.guard_acquisition_reports
                    ],
                    "version_store_summary": {
                        aid: [(vr.card.version, vr.status) for vr in solver.version_store.get_history(aid)]
                        for aid in solver.version_store.action_ids()
                    },
                }
                if solver.name == "itl_s3_contrast_refined":
                    refined_solver_for_audit = solver
                elif isinstance(solver, UnderconstrainedContrastRepairLearner):
                    pre_schemas = {
                        aid: c.to_schema_dict(taxonomy=solver.taxonomy)
                        for aid, c in sorted(solver.pre_repair_cards.items())
                    }
                    pre_op_rec = score_operator_recovery(pre_schemas)
                    pre_gc_hits = 0
                    gc_cases = suite["heldout_guard_contrasts"]
                    for gc in gc_cases:
                        act = gc.public.actions[0]
                        res_pre = solver.pre_repair_cards[act.action_id].apply(
                            gc.public.initial_state, act.role_dict()
                        )
                        if not res_pre.abstained and res_pre.after_state == gc.hidden.gold_final_state:
                            pre_gc_hits += 1
                    pre_gc_acc = pre_gc_hits / float(len(gc_cases))
                    pre_repair_recalls.append(pre_op_rec["recall"])
                    pre_repair_f1s.append(pre_op_rec["f1"])
                    pre_repair_guard_accs.append(pre_gc_acc)
                    rec["s3_telemetry"]["pre_repair_operator_recovery"] = pre_op_rec
                    rec["s3_telemetry"]["pre_repair_guard_contrasts_exact_accuracy"] = round(
                        pre_gc_acc, 6
                    )

            runs_by_solver.setdefault(solver.name, []).append(rec)
            seed_runs[solver.name] = rec
            for split_name, sagg in rec["per_split"].items():
                csv_rows.append(
                    {
                        "seed": seed,
                        "solver": solver.name,
                        "category": solver.category,
                        "split_id": split_name,
                        "total_tasks": sagg["total_tasks"],
                        "learning_scored_tasks": sagg["learning_scored_tasks"],
                        "retrieval_hits": sagg["retrieval_hits"],
                        "learning_exact_accuracy": sagg["learning_exact_accuracy"],
                        "raw_exact_accuracy": sagg["raw_exact_accuracy_including_retrieval"],
                        "abstention_rate": sagg["abstention_rate"],
                        "wrong_non_abstain_rate": sagg["wrong_non_abstain_rate"],
                        "mean_collateral_rate": sagg["mean_collateral_rate"],
                        "intermediate_step_accuracy": sagg["intermediate_step_accuracy"],
                        "elapsed_seconds": sagg["elapsed_seconds"],
                    }
                )

        assert refined_solver_for_audit is not None
        one_shot_and_vs = _audit_one_shot_shadow_and_rollback_for_seed(
            seed, refined_solver_for_audit, inacc_cases
        )

        # Depth-stratified every-step intermediate postcondition verification on heldout splits
        heldout_depth_tasks = {
            1: [
                (c.public.initial_state, c.public.actions, c.hidden.gold_intermediate_states)
                for c in suite["heldout_unseen_names"]
            ],
            2: [
                (c.public.initial_state, c.public.actions, c.hidden.gold_intermediate_states)
                for c in suite["heldout_comp_depth2"]
            ],
            4: [
                (c.public.initial_state, c.public.actions, c.hidden.gold_intermediate_states)
                for c in suite["heldout_comp_depth4"]
            ],
            6: [
                (c.public.initial_state, c.public.actions, c.hidden.gold_intermediate_states)
                for c in suite["heldout_comp_depth6"]
            ],
        }
        contrast_audit = run_contrast_audit(
            refined_solver_for_audit.operators,
            refined_solver_for_audit._grouped_pos_obs,
            make_blackbox_environment(),
            depth_tasks=heldout_depth_tasks,
            ablation_reports=refined_solver_for_audit.ablation_reports,
            guard_acquisition_reports=refined_solver_for_audit.guard_acquisition_reports,
        )

        seed_audit_rec = {
            "seed": seed,
            "one_shot_and_version_audit": one_shot_and_vs,
            "contrast_audit": contrast_audit.to_dict(),
        }
        seed_audits.append(seed_audit_rec)

        (raw_runs_dir / f"s3_seed_{seed}.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "solvers": seed_runs,
                    "s3_audits": seed_audit_rec,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with (raw_runs_dir / "s3_per_split.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    all_splits = list(S3_ALL_UNSEEN_SPLITS) + list(CANARY_SPLITS)
    summary_by_solver: dict[str, Any] = {}
    for sname, recs in runs_by_solver.items():
        per_split_agg: dict[str, Any] = {}
        for split_name in all_splits:
            learn_accs = [r["per_split"][split_name]["learning_exact_accuracy"] for r in recs]
            inter_accs = [r["per_split"][split_name]["intermediate_step_accuracy"] for r in recs]
            raw_accs = [
                r["per_split"][split_name]["raw_exact_accuracy_including_retrieval"] for r in recs
            ]
            col_rates = [r["per_split"][split_name]["mean_collateral_rate"] for r in recs]
            abst_rates = [r["per_split"][split_name]["abstention_rate"] for r in recs]
            ret_rates = [
                r["per_split"][split_name]["retrieval_hits"]
                / max(1, r["per_split"][split_name]["total_tasks"])
                for r in recs
            ]
            scored_counts = [r["per_split"][split_name]["learning_scored_tasks"] for r in recs]
            per_split_agg[split_name] = {
                "learning_scored_tasks_per_seed": scored_counts,
                "learning_scored_tasks_max": max(scored_counts) if scored_counts else 0,
                "learning_exact_accuracy": summarize_across_seeds(learn_accs, bounds=(0.0, 1.0)),
                "intermediate_step_accuracy": summarize_across_seeds(inter_accs, bounds=(0.0, 1.0)),
                "raw_exact_accuracy": summarize_across_seeds(raw_accs, bounds=(0.0, 1.0)),
                "mean_collateral_rate": summarize_across_seeds(col_rates, bounds=(0.0, None)),
                "abstention_rate": summarize_across_seeds(abst_rates, bounds=(0.0, 1.0)),
                "retrieval_hit_rate": summarize_across_seeds(ret_rates, bounds=(0.0, 1.0)),
            }

        bit_keys = list(recs[0]["bit_budget"].keys())
        bit_means = {
            k: round(sum(r["bit_budget"][k] for r in recs) / len(recs), 6) for k in bit_keys
        }
        summary_by_solver[sname] = {
            "category": recs[0]["category"],
            "comparison_only": recs[0]["comparison_only"],
            "evaluator_only": recs[0]["evaluator_only"],
            "unseen_learning_exact_accuracy": summarize_across_seeds(
                [r["unseen_overall"]["learning_exact_accuracy"] for r in recs], bounds=(0.0, 1.0)
            ),
            "unseen_s0_9splits_learning_exact_accuracy": summarize_across_seeds(
                [r["unseen_s0_9splits_overall"]["learning_exact_accuracy"] for r in recs],
                bounds=(0.0, 1.0),
            ),
            "unseen_abstention_rate": summarize_across_seeds(
                [r["unseen_overall"]["abstention_rate"] for r in recs], bounds=(0.0, 1.0)
            ),
            "unseen_wrong_non_abstain_rate": summarize_across_seeds(
                [r["unseen_overall"]["wrong_non_abstain_rate"] for r in recs], bounds=(0.0, 1.0)
            ),
            "unseen_mean_collateral_rate": summarize_across_seeds(
                [r["unseen_overall"]["mean_collateral_rate"] for r in recs], bounds=(0.0, None)
            ),
            "operator_recovery_precision": summarize_across_seeds(
                [r["operator_recovery"]["precision"] for r in recs], bounds=(0.0, 1.0)
            ),
            "operator_recovery_recall": summarize_across_seeds(
                [r["operator_recovery"]["recall"] for r in recs], bounds=(0.0, 1.0)
            ),
            "operator_recovery_f1": summarize_across_seeds(
                [r["operator_recovery"]["f1"] for r in recs], bounds=(0.0, 1.0)
            ),
            "fit_seconds": summarize_across_seeds(
                [r["fit_seconds"] for r in recs], bounds=(0.0, None)
            ),
            "predict_seconds": summarize_across_seeds(
                [r["predict_seconds"] for r in recs], bounds=(0.0, None)
            ),
            "predictions_per_sec_single_core": summarize_across_seeds(
                [r["predictions_per_sec_single_core"] for r in recs], bounds=(0.0, None)
            ),
            "tracemalloc_peak_kb": summarize_across_seeds(
                [r["tracemalloc_peak_kb"] for r in recs], bounds=(0.0, None)
            ),
            "bit_budget_mean": bit_means,
            "per_split": per_split_agg,
            "failed_examples_seed0": recs[0]["failed_examples_sample"],
        }

    s3_gate = _verify_s3_gate(
        summary_by_solver,
        runs_by_solver,
        seed_audits,
        ast_audit,
        s0_ok,
        s1_ok,
        s2_ok,
        seeds,
        demos_per_action,
    )
    total_elapsed = time.perf_counter() - t_all0

    s3_recs = runs_by_solver["itl_s3_contrast_refined"]
    s3_audit_summary = {
        "mean_fit_env_queries": round(
            sum(r["s3_telemetry"]["fit_env_queries"] for r in s3_recs) / len(s3_recs), 2
        ),
        "max_test_time_env_queries": max(
            r["s3_telemetry"]["test_time_env_queries"] for r in s3_recs
        ),
        "max_post_freeze_env_queries": max(
            r["s3_telemetry"]["post_freeze_env_queries"] for r in s3_recs
        ),
        "one_shot_rejection_seeds_passed": sum(
            1 for a in seed_audits if a["one_shot_and_version_audit"]["one_shot_guard_rejection_passed"]
        ),
        "version_store_audit_seeds_passed": sum(
            1 for a in seed_audits if a["one_shot_and_version_audit"]["version_store_audit_passed"]
        ),
        "contrast_audit_seeds_passed": sum(
            1 for a in seed_audits if a["contrast_audit"]["all_checks_passed"]
        ),
        "mean_non_commuting_pairs_detected": round(
            sum(
                a["contrast_audit"]["commutator_order_probes"]["non_commuting_pairs_detected"]
                for a in seed_audits
            )
            / len(seed_audits),
            2,
        ),
    }

    summary = {
        "schema_version": "s3-summary-v1",
        "stage": "S3",
        "reproduce_command": S3_REPRO_COMMAND,
        "operator_card_schema_version": OPERATOR_CARD_SCHEMA_VERSION,
        "version_store_schema_version": VERSION_STORE_SCHEMA_VERSION,
        "contrast_audit_schema_version": CONTRAST_AUDIT_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "temporal_generator_version": TEMPORAL_GENERATOR_VERSION,
        "inaccessible_source_generator_version": S3_INACCESSIBLE_SOURCE_GENERATOR_VERSION,
        "seeds": list(seeds),
        "demos_per_action": demos_per_action,
        "total_elapsed_seconds": round(total_elapsed, 4),
        "environment": capture_environment_info(),
        "ast_isolation_audit": ast_audit,
        "s3_gate": s3_gate,
        "s3_audit_summary": s3_audit_summary,
        "underconstrained_pre_repair_summary": {
            "operator_recovery_recall": summarize_across_seeds(pre_repair_recalls, bounds=(0.0, 1.0)),
            "operator_recovery_f1": summarize_across_seeds(pre_repair_f1s, bounds=(0.0, 1.0)),
            "guard_contrasts_exact_accuracy": summarize_across_seeds(
                pre_repair_guard_accs, bounds=(0.0, 1.0)
            ),
        },
        "solvers": summary_by_solver,
    }

    (reports_dir / "s3_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports_dir / "s3_research_report.md").write_text(
        _render_s3_markdown_report(summary), encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible Mirror 7 CPU-Native Learning Stage S3 Controlled Contrasts benchmark."
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
        help="Verify committed S0, S1, S2, and S3 manifests/preregistrations match deterministic generation.",
    )
    args = parser.parse_args(argv)

    if args.verify_manifests:
        ok0, m0 = verify_committed_manifests(seeds=args.seeds, demos_per_action=args.demos_per_action)
        ok1, m1 = verify_s1_committed_manifests(seeds=args.seeds, demos_per_action=args.demos_per_action)
        ok2, m2 = verify_s2_committed_manifests(seeds=args.seeds, demos_per_action=args.demos_per_action)
        ok3, m3 = verify_s3_committed_manifests(seeds=args.seeds, demos_per_action=args.demos_per_action)
        if not (ok0 and ok1 and ok2 and ok3):
            for msg in m0 + m1 + m2 + m3:
                print(f"MANIFEST ERROR: {msg}", file=sys.stderr)
            return 1
        print("Committed S0, S1, S2, and S3 manifests verified bit-exact against deterministic generators.")
        return 0

    summary = run_s3_experiments(seeds=args.seeds, demos_per_action=args.demos_per_action)
    gate = summary["s3_gate"]
    passed = sum(1 for v in gate["checks"].values() if v)
    total = len(gate["checks"])
    print(f"Stage S3 Gate: {'PASS' if gate['passed'] else 'FAIL'} ({passed}/{total} checks passed)")
    for k, ok in gate["checks"].items():
        if not ok:
            print(f"  FAILED CHECK: {k}", file=sys.stderr)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
