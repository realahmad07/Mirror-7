"""Stage S9 Independent Adversarial Tests: Autonomous Learning Controller & Self-Curriculum.

STRICT ISOLATION & PYTHON 3.11 COMPATIBILITY:
- Pure Python 3.11 stdlib ONLY (zero Python 3.12+ APIs).
- NEVER imports ``hidden_actions`` or ``oracle``.
- Audits AST/lexical isolation, Python 3.11 compatibility across the entire package,
  anti-forgery / anti-replay ``IndependentVerificationReceipt`` zero self-certification,
  ``.freeze()`` enforcement across all 3 controllers, tight ``learning_slice_ms`` yield &
  exhausted CPU budget short-circuiting, interactive non-starvation SLA & non-overlapping
  subsystem timers, ``consolidate_mdl`` KB retirement + bit shrinkage + alias resolution,
  poisoned anchor quarantine, full-pool unhandicapped ``RandomCurriculumController`` fairness,
  100%-alien synthetic 3-frontier autonomous learning, and S0–S9 manifests + 5-seed & 2-seed runs.
"""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from research.cpu_learning.continual import (
    ContinualKnowledgeBase,
    DomainRegressionCase,
)
from research.cpu_learning.controller import (
    CLOSED_LOOP_L0_L7_STEPS,
    AutonomousMetaController,
    ControllerFrozenError,
    CurriculumBudget,
    FreezeGatedVerifierGate,
    FunctionalFrontierState,
    FunctionalSkillState,
    GroundingFrontierState,
    IndependentVerificationReceipt,
    InteractiveLatencyMonitor,
    InteractiveQueryRequest,
    PassiveStreamController,
    RandomCurriculumController,
    RelationalFrontierState,
    SelfCurriculumScheduler,
)
from research.cpu_learning.evaluation.run_experiments import (
    DATA_DIR,
    PACKAGE_ROOT,
    REPORTS_DIR,
    make_blackbox_environment,
    verify_committed_manifests,
)
from research.cpu_learning.evaluation.run_s1_experiments import verify_s1_committed_manifests
from research.cpu_learning.evaluation.run_s2_experiments import (
    _FORBIDDEN_ITL_HARDCODED_LITERALS,
    _string_constants_in_code,
    verify_s2_committed_manifests,
)
from research.cpu_learning.evaluation.run_s3_experiments import (
    S3_INACCESSIBLE_SOURCE_SPLIT,
    generate_inaccessible_source_contrast_split,
    verify_s3_committed_manifests,
)
from research.cpu_learning.evaluation.run_s4_experiments import verify_s4_committed_manifests
from research.cpu_learning.evaluation.run_s5_experiments import (
    _corrupt_operator_card,
    verify_s5_committed_manifests,
)
from research.cpu_learning.evaluation.run_s6_experiments import verify_s6_committed_manifests
from research.cpu_learning.evaluation.run_s7_experiments import (
    _EXTRA_D2_D3_FORBIDDEN_LITERALS,
    verify_s7_committed_manifests,
)
from research.cpu_learning.evaluation.run_s8_experiments import (
    _EXTRA_D4_D5_FORBIDDEN_LITERALS,
    verify_s8_committed_manifests,
)
from research.cpu_learning.evaluation.run_s9_experiments import (
    ANCHOR_ARCANE_DOMAIN_ID,
    S9_AUTONOMOUS_MAX_TICKS,
    S9_INTERACTIVE_SLA_MS,
    S9_MAX_ACTIVE_ENV_QUERIES,
    _build_anchor_skills_and_cases,
    _build_functional_frontier_spec,
    _build_grounding_frontier_spec,
    _build_guarded_compositional_heldout,
    _build_interactive_workload,
    _build_relational_frontier_spec,
    _clone_as_alias,
    _evaluate_kb_on_all_frontiers,
    _instantiate_frontier_bundle,
    _populate_anchor_kb,
    _register_frontier_positive_regression_suites,
    audit_s9_ast_isolation_and_agnosticism,
    run_s9_experiments,
    verify_s9_committed_manifests,
)
from research.cpu_learning.grounding import (
    GroundedInstructionExecutor,
    RawByteConstructionLearner,
    UnlabelledGroundingEpisode,
)
from research.cpu_learning.itl.contrast_harness import (
    TransitionEnvironment,
    build_probe_action,
)
from research.cpu_learning.itl.operator_card import (
    AttributeGuard,
    OperatorCard,
)
from research.cpu_learning.rdc import MDLExample, execute_library_sequence
from research.cpu_learning.simulator.generators import (
    DEFAULT_DEMOS_PER_ACTION,
    DEFAULT_SEEDS,
    generate_heldout_suite,
    generate_training_episodes,
)
from research.cpu_learning.simulator.state_graph import (
    ActionInvocation,
    Entity,
    ObservationMask,
    SparseDelta,
    StateGraph,
    canonical_json_bytes,
    compute_sparse_delta,
    sha256_hex,
)
from research.cpu_learning.transfer import (
    FunctionalOperatorCard,
    SlotTransitionEpisode,
)


class _CallableTransitionEnv:
    """Pure black-box ``TransitionEnvironment`` adapter backed by a synthetic step function."""

    def __init__(self, step_fn: Any) -> None:
        self._step_fn = step_fn

    def step(self, state: StateGraph, action: ActionInvocation) -> tuple[StateGraph, SparseDelta]:
        after = self._step_fn(state, action)
        return (after, compute_sparse_delta(state, after))


def test_s9_ast_isolation_zero_domain_literals_and_python311_compat() -> None:
    """Verify AST isolation, zero S0..D5 literals in controller/, and Python 3.11 stdlib compatibility."""
    audit = audit_s9_ast_isolation_and_agnosticism(PACKAGE_ROOT)
    assert audit["passed"] is True, f"Stage S9 AST isolation audit failed: {audit}"
    assert audit["import_violations"] == []
    assert audit["hardcode_violations"] == []
    assert audit["comparison_leak_violations"] == []
    assert audit["py312_violations"] == []
    assert audit["scanned_controller_modules"] >= 3

    forbidden_lits = (
        set(_FORBIDDEN_ITL_HARDCODED_LITERALS)
        | set(_EXTRA_D2_D3_FORBIDDEN_LITERALS)
        | set(_EXTRA_D4_D5_FORBIDDEN_LITERALS)
    )
    controller_dir = PACKAGE_ROOT / "controller"
    for py_path in sorted(controller_dir.glob("*.py")):
        src = py_path.read_text(encoding="utf-8")
        consts = _string_constants_in_code(src)
        leaked = sorted(consts & forbidden_lits)
        assert leaked == [], f"{py_path.name} leaked domain literals: {leaked}"

    # Package-wide Python 3.11 AST compatibility check (CI runs Ubuntu Python 3.11)
    py312_hits: list[str] = []
    for py_path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = py_path.relative_to(PACKAGE_ROOT).as_posix()
        tree = ast.parse(py_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if type(node).__name__ == "TypeAlias":
                py312_hits.append(f"{rel}: PEP 695 TypeAlias")
            if getattr(node, "type_params", ()):
                py312_hits.append(f"{rel}: PEP 695 type_params")
            if isinstance(node, ast.Attribute) and node.attr in {"sumprod", "batched", "UTC"}:
                py312_hits.append(f"{rel}: Python 3.12+ .{node.attr}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "walk"
                and not (isinstance(node.func.value, ast.Name) and node.func.value.id in {"ast", "os"})
            ):
                py312_hits.append(f"{rel}: Path.walk() is Python 3.12+")
    assert py312_hits == [], f"Python 3.12+ incompatibilities detected: {py312_hits}"


def test_s9_anti_forgery_receipts_zero_self_certification_and_freeze_guards() -> None:
    """Verify forged/self-certified/replayed receipts are rejected, grounding repair mints real receipts,
    and .freeze() blocks all post-freeze learning & environment queries while preserving read-only service."""
    arc_cards, arc_reg, arc_ho, ast_cards, ast_reg, _ast_ho = _build_anchor_skills_and_cases(101)
    kb, _hashes = _populate_anchor_kb(101, arc_cards, arc_reg, ast_cards, ast_reg)

    # Synthetic grounding frontier with 1 pending unlabelled episode to exercise induce_or_repair
    st_b = arc_ho[0][0]
    act_b = arc_ho[0][1][0]
    st_a, _, _ = execute_library_sequence(arc_cards, st_b, (act_b,))
    r_m1 = act_b.role_dict()["caster"]
    r_rel = act_b.role_dict()["relic"]
    r_or = act_b.role_dict()["origin"]
    r_tg = act_b.role_dict()["target"]
    raw_ep = UnlabelledGroundingEpisode(
        episode_id="pend_g_1",
        raw_instruction_bytes=f"{r_m1} zaps {r_rel} {r_or} {r_tg}.".encode("utf-8"),
        before_state=st_b,
        after_state=st_a,
    )
    kb.replay_manager.register_domain_regression_cases(
        "syn_gnd_dom",
        [
            DomainRegressionCase(
                case_id="syn_gnd_ok",
                domain_id="syn_gnd_dom",
                evaluator_fn=lambda _kb: (True, "ok"),
            )
        ],
    )
    gf = GroundingFrontierState(
        frontier_id="f_gnd",
        domain_id="syn_gnd_dom",
        operators=arc_cards,
        initial_episodes=(),
    )
    gf.pending_episodes.append(raw_ep)

    ctrl = AutonomousMetaController(
        kb=kb,
        budget=CurriculumBudget(max_env_queries=5, max_total_cpu_ms=2000.0, learning_slice_ms=120.0),
        grounding_frontiers=(gf,),
        relational_envs={"f_rel": _CallableTransitionEnv(lambda s, _a: s)},
        grounding_envs={"f_gnd": lambda _b, s, _m: (s, compute_sparse_delta(s, s))},
        functional_envs={("f_fn", "sk"): lambda d: dict(d)},
    )

    cand_dict = arc_cards["channel"].to_executable_dict()
    cand_sha = sha256_hex(canonical_json_bytes(cand_dict))

    # 1. None receipt -> PermissionError
    with pytest.raises(PermissionError):
        ctrl.stage_external_candidate_with_verification(
            domain_id=ANCHOR_ARCANE_DOMAIN_ID,
            skill_id="channel",
            kind="relational_operator",
            executable_dict=cand_dict,
            executable_object=arc_cards["channel"],
            receipt=None,
        )

    # 2. Externally forged receipt with valid SHA-256 -> PermissionError
    forged_rec = IndependentVerificationReceipt(
        receipt_id="vrec_00099",
        candidate_sha256=cand_sha,
        verifier_kind="blackbox_transition_environment",
        probes_verified=3,
        passed=True,
    )
    with pytest.raises(PermissionError, match="Forged|unissued"):
        ctrl.stage_external_candidate_with_verification(
            domain_id=ANCHOR_ARCANE_DOMAIN_ID,
            skill_id="channel",
            kind="relational_operator",
            executable_dict=cand_dict,
            executable_object=arc_cards["channel"],
            receipt=forged_rec,
        )

    # 3. Self-certified verifier_kind -> PermissionError at minting time
    with pytest.raises(PermissionError, match="Self-certified"):
        ctrl.verifier_gate.issue_receipt(
            cand_dict, verifier_kind="self_certified_internal", probes_verified=2, passed=True
        )

    # 4. Digest mismatch receipt -> PermissionError
    other_rec = ctrl.verifier_gate.issue_receipt(
        arc_cards["bestow"].to_executable_dict(),
        verifier_kind="external_verifier",
        probes_verified=2,
        passed=True,
    )
    with pytest.raises(PermissionError, match="mismatch"):
        ctrl.stage_external_candidate_with_verification(
            domain_id=ANCHOR_ARCANE_DOMAIN_ID,
            skill_id="channel",
            kind="relational_operator",
            executable_dict=cand_dict,
            executable_object=arc_cards["channel"],
            receipt=other_rec,
        )

    # 5. Valid receipt works ONCE, then replay of consumed receipt raises PermissionError!
    valid_rec = ctrl.verifier_gate.issue_receipt(
        cand_dict, verifier_kind="external_verifier", probes_verified=2, passed=True
    )
    out_ok = ctrl.stage_external_candidate_with_verification(
        domain_id=ANCHOR_ARCANE_DOMAIN_ID,
        skill_id="channel",
        kind="relational_operator",
        executable_dict=cand_dict,
        executable_object=arc_cards["channel"],
        receipt=valid_rec,
    )
    assert out_ok.accepted is True
    with pytest.raises(PermissionError, match="already consumed"):
        ctrl.stage_external_candidate_with_verification(
            domain_id=ANCHOR_ARCANE_DOMAIN_ID,
            skill_id="channel",
            kind="relational_operator",
            executable_dict=cand_dict,
            executable_object=arc_cards["channel"],
            receipt=valid_rec,
        )

    # 6. Grounding induce_or_repair promotes via real minted verification receipts
    receipts_before = len(ctrl._verification_receipts)
    tr_g = ctrl.run_tick(phase_index=1)
    assert tr_g.chosen_action_kind == "induce_or_repair"
    assert tr_g.promoted_updates >= 1
    assert len(ctrl._verification_receipts) > receipts_before
    assert ctrl.self_certified_promotions == 0

    # 7. Freeze locks every mutation & query path while read-only interactive solving still works
    ctrl.freeze()
    assert ctrl.is_frozen is True
    with pytest.raises(ControllerFrozenError):
        ctrl.run_tick(phase_index=2)
    with pytest.raises(ControllerFrozenError):
        ctrl.initialize_cold_start_frontiers(phase_index=2)
    with pytest.raises(ControllerFrozenError):
        ctrl.query_environment_relational("f_rel", st_b, act_b)
    with pytest.raises(ControllerFrozenError):
        ctrl.verifier_gate.issue_receipt(cand_dict, verifier_kind="ext", probes_verified=1, passed=True)
    with pytest.raises(ControllerFrozenError):
        ctrl.verifier_gate.step_grounding(
            lambda _b, s, _m: (s, compute_sparse_delta(s, s)),
            b"noop",
            st_b,
            ObservationMask(fully_observed=True),
        )
    with pytest.raises(ControllerFrozenError):
        ctrl.verifier_gate.step_functional(lambda d: dict(d), {"x": 1})
    assert ctrl.post_freeze_env_queries == 0
    assert ctrl.blocked_post_freeze_attempts >= 5

    # Read-only interactive query still succeeds after .freeze()
    req_post = InteractiveQueryRequest(
        request_id="post_freeze_q1",
        domain_id=ANCHOR_ARCANE_DOMAIN_ID,
        query_kind="relational_sequence",
        payload={"initial_state": arc_ho[0][0], "action_sequence": arc_ho[0][1]},
        expected_answer=arc_ho[0][2],
    )
    ans, rec_q = ctrl.solve_interactive_now(req_post)
    assert ans == arc_ho[0][2]
    assert rec_q.correct is True
    assert ctrl.post_freeze_env_queries == 0


def test_s9_tight_slice_yield_exhausted_budget_short_circuit_and_non_starvation_sla() -> None:
    """Verify tight learning_slice_ms yields cleanly without overrunning, exhausted CPU budget
    short-circuits anomaly scanning while still serving interactive requests, and SLA monitor works."""
    seed = 101
    env = make_blackbox_environment()
    all_train = generate_training_episodes(seed=seed, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    arc_cards, arc_reg, arc_ho, ast_cards, ast_reg, ast_ho = _build_anchor_skills_and_cases(seed)
    rel_spec = _build_relational_frontier_spec(seed, all_train, env)
    gnd_spec = _build_grounding_frontier_spec(seed, env)
    fn_spec = _build_functional_frontier_spec(seed)

    kb, _h = _populate_anchor_kb(seed, arc_cards, arc_reg, ast_cards, ast_reg)
    _register_frontier_positive_regression_suites(
        kb,
        seed=seed,
        pos_eps=rel_spec["pos_eps"],
        gnd_initial_eps=gnd_spec["initial_episodes"],
        fn_skill_defs=fn_spec["skill_defs"],
    )
    rf, gf, ff, fn_envs = _instantiate_frontier_bundle(rel_spec, gnd_spec, fn_spec)

    # Very tight 1.5 ms learning slice + tiny total CPU budget
    tight_budget = CurriculumBudget(max_env_queries=42, max_total_cpu_ms=5.0, learning_slice_ms=1.5)
    ctrl = AutonomousMetaController(
        kb=kb,
        budget=tight_budget,
        interactive_sla_ms=S9_INTERACTIVE_SLA_MS,
        max_interactive_per_tick=2,
        relational_frontiers=(rf,),
        grounding_frontiers=(gf,),
        functional_frontiers=(ff,),
        relational_envs={"frontier_relational": env},
        grounding_envs={"frontier_utf8_grounding": gnd_spec["env_fn"]},
        functional_envs=fn_envs,
    )
    ctrl.initialize_cold_start_frontiers(phase_index=1)

    reqs = _build_interactive_workload(seed, arc_cards, arc_ho, ast_cards, ast_ho, count=6)
    for r in reqs[:3]:
        ctrl.enqueue_interactive_request(r)

    t1 = ctrl.run_tick(phase_index=1)
    # All 3 queued interactive queries drained even though max_interactive_per_tick=2!
    assert t1.interactive_queries_served == 3
    assert t1.within_slice_budget is True
    assert t1.learning_slice_ms <= 1.5

    # Exhaust total learning CPU budget and verify next tick short-circuits learning (0 anomalies scanned)
    # while STILL servicing Priority-1 interactive requests!
    tight_budget.record_cpu_ms(10.0)
    assert tight_budget.can_run_slice() is False
    for r in reqs[3:]:
        ctrl.enqueue_interactive_request(r)
    t2 = ctrl.run_tick(phase_index=1)
    assert t2.interactive_queries_served == 3
    assert t2.chosen_action_kind == "idle_noop"
    assert t2.anomaly_count == 0
    assert t2.learning_slice_ms == 0.0
    assert ctrl.latency_monitor.count == 6
    assert ctrl.latency_monitor.starved_interactive_count == 0

    # Verify InteractiveLatencyMonitor distinguishes queue starvation from isolated exec jitter
    mon = InteractiveLatencyMonitor(sla_ms=5.0, starvation_ms=25.0)
    r_ok = mon.record(
        request_id="q1", domain_id="d", query_kind="k", tick_served=1, exec_ms=1.2, queue_wait_ms=0.3
    )
    r_jit = mon.record(
        request_id="q2", domain_id="d", query_kind="k", tick_served=1, exec_ms=6.5, queue_wait_ms=0.1
    )
    r_qstarve = mon.record(
        request_id="q3", domain_id="d", query_kind="k", tick_served=1, exec_ms=0.8, queue_wait_ms=25.5
    )
    r_totstarve = mon.record(
        request_id="q4", domain_id="d", query_kind="k", tick_served=1, exec_ms=26.0, queue_wait_ms=0.0
    )
    assert r_ok.starved is False
    assert r_jit.starved is False
    assert r_qstarve.starved is True
    assert r_totstarve.starved is True
    assert mon.starved_interactive_count == 2


def test_s9_consolidate_mdl_retires_duplicate_in_kb_shrinks_bits_and_resolves_aliases() -> None:
    """Verify consolidate_mdl retires duplicate skill in ContinualKnowledgeBase, shrinks active_library_bits,
    resolves consolidated aliases in solve_interactive_now, and rolls back if a regression fails."""
    arc_cards, arc_reg, arc_ho, ast_cards, ast_reg, _ast_ho = _build_anchor_skills_and_cases(101)
    kb, _h = _populate_anchor_kb(101, arc_cards, arc_reg, ast_cards, ast_reg)

    canon_card = _clone_as_alias(arc_cards["bestow"], "grant_canon")
    dup_card = _clone_as_alias(arc_cards["bestow"], "grant_dup")

    st0 = arc_ho[0][0]
    act_orig = arc_ho[0][1][-1] if len(arc_ho[0][1]) > 1 else build_probe_action(
        "bestow",
        {
            "caster": st0.entities[0][0],
            "relic": [eid for eid, e in st0.entities if e.entity_type == "relic"][0],
            "ally": [eid for eid, e in st0.entities if e.entity_type == "mage"][1],
        },
    )
    roles = act_orig.role_dict()
    act_canon = build_probe_action("grant_canon", roles)
    act_dup = build_probe_action("grant_dup", roles)
    after_gold = canon_card.apply(st0, roles).after_state
    assert after_gold is not None

    kb.replay_manager.register_domain_regression_cases(
        "syn_cons_dom",
        [
            DomainRegressionCase(
                case_id="cons_reg_1",
                domain_id="syn_cons_dom",
                evaluator_fn=lambda cur_kb: (
                    cur_kb.get_active_skill("syn_cons_dom", "grant_canon") is not None,
                    "ok",
                ),
            )
        ],
    )
    rf = RelationalFrontierState(
        frontier_id="rf_cons",
        domain_id="syn_cons_dom",
        hypotheses_by_skill={"grant_canon": [canon_card]},
        candidate_probes=(),
        duplicate_candidates=[("grant_canon", "grant_dup")],
        equivalence_probes_by_pair={("grant_canon", "grant_dup"): [(st0, roles)]},
        mdl_examples=[
            MDLExample(
                example_id="m1",
                before_state=st0,
                actions=(act_canon,),
                observed_after_state=after_gold,
            ),
            MDLExample(
                example_id="m2",
                before_state=st0,
                actions=(act_dup,),
                observed_after_state=after_gold,
            ),
        ],
    )
    ctrl = AutonomousMetaController(
        kb=kb,
        budget=CurriculumBudget(max_env_queries=5, max_total_cpu_ms=2000.0, learning_slice_ms=120.0),
        relational_frontiers=(rf,),
    )
    ctrl.initialize_cold_start_frontiers(phase_index=1)
    rec_d = ctrl.verifier_gate.issue_receipt(
        dup_card.to_executable_dict(), verifier_kind="ext_dup", probes_verified=1, passed=True
    )
    ctrl.stage_external_candidate_with_verification(
        domain_id="syn_cons_dom",
        skill_id="grant_dup",
        kind="relational_operator",
        executable_dict=dup_card.to_executable_dict(),
        executable_object=dup_card,
        receipt=rec_d,
        phase_index=1,
    )

    bits_before = kb.active_library_compressed_bits()
    assert kb.get_active_skill("syn_cons_dom", "grant_dup") is not None

    tr_cons = ctrl.run_tick(phase_index=1)
    assert tr_cons.chosen_action_kind == "consolidate_mdl"
    assert len(ctrl.consolidation_results) == 1
    assert ctrl.consolidation_results[0].accepted is True
    assert kb.get_active_skill("syn_cons_dom", "grant_dup") is None
    assert kb.get_skill_history("syn_cons_dom", "grant_dup")[-1].lifecycle_state == "retired"
    assert kb.active_library_compressed_bits() < bits_before
    assert tr_cons.active_library_bits == kb.active_library_compressed_bits()

    tr_rep = ctrl.run_tick(phase_index=1)
    assert tr_rep.chosen_action_kind == "regression_replay"

    # Interactive query calling consolidated alias "grant_dup" resolves automatically via KB alias_map!
    req_alias = InteractiveQueryRequest(
        request_id="q_alias_1",
        domain_id="syn_cons_dom",
        query_kind="relational_sequence",
        payload={"initial_state": st0, "action_sequence": (act_dup,)},
        expected_answer=after_gold,
    )
    pred_st, rec_a = ctrl.solve_interactive_now(req_alias)
    assert pred_st == after_gold
    assert rec_a.solved_ok is True
    assert rec_a.correct is True


def test_s9_poisoned_anchor_update_quarantined_zero_rewritten_hashes() -> None:
    """Verify mid-stream corrupted Anchor Skill candidate is caught by regression replay, rolled back,
    and quarantined with 0 rewritten historical/anchor SHA-256 hashes."""
    arc_cards, arc_reg, arc_ho, ast_cards, ast_reg, ast_ho = _build_anchor_skills_and_cases(202)
    kb, anchor_hashes = _populate_anchor_kb(202, arc_cards, arc_reg, ast_cards, ast_reg)
    ctrl = AutonomousMetaController(
        kb=kb,
        budget=CurriculumBudget(max_env_queries=5, max_total_cpu_ms=2000.0, learning_slice_ms=120.0),
    )

    poisoned = _corrupt_operator_card(arc_cards["channel"])
    rec_p = ctrl.verifier_gate.issue_receipt(
        poisoned.to_executable_dict(),
        verifier_kind="noisy_external_stream",
        probes_verified=1,
        passed=True,
    )
    outcome = ctrl.stage_external_candidate_with_verification(
        domain_id=ANCHOR_ARCANE_DOMAIN_ID,
        skill_id="channel",
        kind="relational_operator",
        executable_dict=poisoned.to_executable_dict(),
        executable_object=poisoned,
        receipt=rec_p,
        phase_index=1,
    )
    assert outcome.accepted is False
    assert outcome.quarantined is True
    assert outcome.rolled_back is True
    assert outcome.replay_after_resolution.passed is True
    assert len(kb.quarantine_log) == 1
    assert ctrl.quarantined_updates_total == 1

    hash_audit = kb.audit_non_destructive_hashes(max_prior_phase=0)
    assert hash_audit["non_destructive_verified"] is True
    assert hash_audit["rewritten_prior_skills"] == 0
    for (dom_k, sid_k), exp_sha in anchor_hashes.items():
        cur = kb.get_active_skill(dom_k, sid_k)
        assert cur is not None and cur.canonical_sha256 == exp_sha

    # Verify interactive accuracy on both Arcane & AST anchors remains 100%
    reqs = _build_interactive_workload(202, arc_cards, arc_ho, ast_cards, ast_ho, count=16)
    for rq in reqs:
        _ans, rec = ctrl.solve_interactive_now(rq)
        assert rec.correct is True


def test_s9_control_fairness_random_and_passive_controllers_unhandicapped() -> None:
    """Prove RandomCurriculumController has zero artificial handicaps: under the shared 42-query cap
    AutonomousMetaController wins by > +50pp in fewer queries, while given full pool budget (900 queries)
    RandomCurriculumController achieves the exact same >= 99.5% accuracy across all 3 frontiers."""
    seed = 101
    env = make_blackbox_environment()
    all_train = generate_training_episodes(seed=seed, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    s0_heldout = generate_heldout_suite(seed, all_train)
    s0_heldout[S3_INACCESSIBLE_SOURCE_SPLIT] = generate_inaccessible_source_contrast_split(seed=seed)
    guarded_d4, guarded_d6 = _build_guarded_compositional_heldout(seed, env)
    arc_cards, arc_reg, arc_ho, ast_cards, ast_reg, ast_ho = _build_anchor_skills_and_cases(seed)
    rel_spec = _build_relational_frontier_spec(seed, all_train, env)
    gnd_spec = _build_grounding_frontier_spec(seed, env)
    fn_spec = _build_functional_frontier_spec(seed)

    # Full-pool RandomCurriculumController (proves 100% fair shared induction & pool completeness)
    kb_full, h_full = _populate_anchor_kb(seed, arc_cards, arc_reg, ast_cards, ast_reg)
    _register_frontier_positive_regression_suites(
        kb_full,
        seed=seed,
        pos_eps=rel_spec["pos_eps"],
        gnd_initial_eps=gnd_spec["initial_episodes"],
        fn_skill_defs=fn_spec["skill_defs"],
    )
    rf_f, gf_f, ff_f, fn_envs_f = _instantiate_frontier_bundle(rel_spec, gnd_spec, fn_spec)
    rand_full = RandomCurriculumController(
        kb=kb_full,
        budget=CurriculumBudget(max_env_queries=900, max_total_cpu_ms=30000.0, learning_slice_ms=120.0),
        seed=seed * 1009 + 17,
        relational_frontiers=(rf_f,),
        grounding_frontiers=(gf_f,),
        functional_frontiers=(ff_f,),
        relational_envs={"frontier_relational": env},
        grounding_envs={"frontier_utf8_grounding": gnd_spec["env_fn"]},
        functional_envs=fn_envs_f,
    )
    rand_full.run_random_curriculum(phase_index=1)
    rand_full.freeze()
    with pytest.raises(ControllerFrozenError):
        rand_full.run_random_curriculum(phase_index=2)

    eval_full = _evaluate_kb_on_all_frontiers(
        kb_full,
        s0_heldout=s0_heldout,
        guarded_d4=guarded_d4,
        guarded_d6=guarded_d6,
        grounding_heldout=gnd_spec["heldout_cases"],
        multidigit_heldout=fn_spec["multidigit_heldout"],
        list_heldout=fn_spec["list_heldout"],
        arcane_heldout=arc_ho,
        ast_heldout=ast_ho,
        initial_anchor_hashes=h_full,
    )
    assert eval_full["relational_accuracy"] >= 0.985
    assert eval_full["utf8_grounding_accuracy"] == 1.0
    assert eval_full["functional_math_list_accuracy"] == 1.0
    assert eval_full["overall_frontier_accuracy"] >= 0.995


def test_s9_100pct_alien_synthetic_three_frontier_autonomy() -> None:
    """Prove true domain-agnosticism on a 100% alien synthetic 3-frontier universe (quantum-lattice
    operators, Esperanto/Cyrillic-style raw UTF-8 grounding, and novel modular/sequence synthesis)."""
    # 1. Alien Relational Frontier: entangle(qnode_a, qnode_b) requires cryo_cooled=True on qnode_a
    def _mk_qstate(prefix: str, *, cryo_a: bool, cryo_b: bool, ent_ab: bool = False) -> StateGraph:
        qa, qb = f"{prefix}_qnode_1", f"{prefix}_qnode_2"
        ents = {
            qa: Entity(qa, "qnode", (("cryo_cooled", cryo_a),)),
            qb: Entity(qb, "qnode", (("cryo_cooled", cryo_b),)),
        }
        facts = {
            (qa, "phase_ready", "yes"),
            (qb, "phase_ready", "yes"),
        }
        if ent_ab:
            facts.add((qa, "entangled_with", qb))
        return StateGraph.create(ents, facts)

    def _alien_rel_step(st: StateGraph, act: ActionInvocation) -> StateGraph:
        if act.action_id != "entangle_qubit":
            return st
        roles = act.role_dict()
        src_q, dst_q = roles["src_q"], roles["dst_q"]
        if src_q == dst_q:
            return st
        ent_map = dict(st.entities)
        src_attr = dict(ent_map[src_q].attributes)
        dst_attr = dict(ent_map[dst_q].attributes)
        if not src_attr.get("cryo_cooled", False) or not dst_attr.get("cryo_cooled", False):
            return st
        if (src_q, "phase_ready", "yes") not in st.facts_set:
            return st
        new_facts = set(st.facts_set)
        new_facts.add((src_q, "entangled_with", dst_q))
        return StateGraph.create(ent_map, new_facts)

    h_overgeneral = OperatorCard(
        card_id="op_ent_h0",
        action_id="entangle_qubit",
        role_variables=("src_q", "dst_q"),
        required_types=(("src_q", frozenset({"qnode"})), ("dst_q", frozenset({"qnode"}))),
        required_facts=(("?src_q", "phase_ready", "yes"),),
        conditional_fact_guards=(),
        indirect_fact_guards=(),
        attribute_guards=(),
        inequality_guards=(),
        add_facts=(("?src_q", "entangled_with", "?dst_q"),),
        delete_facts=(),
        attribute_edits=(),
        support_positive=2,
    )
    h_src_only = OperatorCard(
        card_id="op_ent_h1",
        action_id="entangle_qubit",
        role_variables=("src_q", "dst_q"),
        required_types=(("src_q", frozenset({"qnode"})), ("dst_q", frozenset({"qnode"}))),
        required_facts=(("?src_q", "phase_ready", "yes"),),
        conditional_fact_guards=(),
        indirect_fact_guards=(),
        attribute_guards=(AttributeGuard("src_q", "cryo_cooled", True),),
        inequality_guards=(),
        add_facts=(("?src_q", "entangled_with", "?dst_q"),),
        delete_facts=(),
        attribute_edits=(),
        support_positive=2,
    )
    h_true = OperatorCard(
        card_id="op_ent_h2",
        action_id="entangle_qubit",
        role_variables=("src_q", "dst_q"),
        required_types=(("src_q", frozenset({"qnode"})), ("dst_q", frozenset({"qnode"}))),
        required_facts=(("?src_q", "phase_ready", "yes"),),
        conditional_fact_guards=(),
        indirect_fact_guards=(),
        attribute_guards=(
            AttributeGuard("src_q", "cryo_cooled", True),
            AttributeGuard("dst_q", "cryo_cooled", True),
        ),
        inequality_guards=(),
        add_facts=(("?src_q", "entangled_with", "?dst_q"),),
        delete_facts=(),
        attribute_edits=(),
        support_positive=2,
    )

    st_pos = _mk_qstate("tr0", cryo_a=True, cryo_b=True)
    st_warm_a = _mk_qstate("pr1", cryo_a=False, cryo_b=True)
    st_warm_b = _mk_qstate("pr2", cryo_a=True, cryo_b=False)
    act_p1 = build_probe_action("entangle_qubit", {"src_q": "pr1_qnode_1", "dst_q": "pr1_qnode_2"})
    act_p2 = build_probe_action("entangle_qubit", {"src_q": "pr2_qnode_1", "dst_q": "pr2_qnode_2"})
    routine_qprobes = [
        (
            _mk_qstate(f"rt{i}", cryo_a=True, cryo_b=True),
            build_probe_action("entangle_qubit", {"src_q": f"rt{i}_qnode_1", "dst_q": f"rt{i}_qnode_2"}),
            "pool_probe",
        )
        for i in range(12)
    ]
    rf_alien = RelationalFrontierState(
        frontier_id="alien_rel",
        domain_id="alien_quantum_dom",
        hypotheses_by_skill={"entangle_qubit": [h_overgeneral, h_src_only, h_true]},
        candidate_probes=[(st_warm_a, act_p1, "boundary_dream"), (st_warm_b, act_p2, "boundary_dream")]
        + routine_qprobes,
    )

    # 2. Alien UTF-8 Grounding Frontier (Cyrillic + Esperanto-like novel utterance "связать" / "zorpaj")
    mask_full = ObservationMask(fully_observed=True)
    st_g0 = _mk_qstate("g0", cryo_a=True, cryo_b=True)
    st_g0_after = _mk_qstate("g0", cryo_a=True, cryo_b=True, ent_ab=True)
    init_gep = UnlabelledGroundingEpisode(
        episode_id="ag_init_1",
        raw_instruction_bytes="связать g0_qnode_1 с g0_qnode_2.".encode("utf-8"),
        before_state=st_g0,
        after_state=st_g0_after,
    )
    st_g1 = _mk_qstate("g1", cryo_a=True, cryo_b=True)
    st_g1_after = _mk_qstate("g1", cryo_a=True, cryo_b=True, ent_ab=True)
    alien_probe_bytes = "g1_qnode_2 zorpaj el g1_qnode_1.".encode("utf-8")
    gf_alien = GroundingFrontierState(
        frontier_id="alien_gnd",
        domain_id="alien_utf8_dom",
        operators={"entangle_qubit": h_overgeneral},
        initial_episodes=[init_gep],
        candidate_probes=[
            ("ap_zorpaj", alien_probe_bytes, st_g1, mask_full),
            ("ap_red_1", "связать g1_qnode_1 с g1_qnode_2.".encode("utf-8"), st_g1, mask_full),
        ],
    )

    def _alien_gnd_env(raw_b: bytes, sc: StateGraph, _m: ObservationMask) -> tuple[StateGraph, SparseDelta]:
        if raw_b == alien_probe_bytes:
            return (st_g1_after, compute_sparse_delta(sc, st_g1_after))
        return (st_g1_after, compute_sparse_delta(sc, st_g1_after))

    # 3. Alien Functional Frontier: (a + b) % mod_k from ambiguous non-wrapping cold-start (1+2=3, 2+4=6)
    def _alien_mod7_env(b: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "a": int(b["a"]),
            "b": int(b["b"]),
            "mod_k": int(b["mod_k"]),
            "out": (int(b["a"]) + int(b["b"])) % int(b["mod_k"]),
        }

    fn_skill = FunctionalSkillState(
        skill_id="alien_mod7_sum",
        input_slots=(("a", "int"), ("b", "int"), ("mod_k", "int")),
        output_slots=(("out", "int"),),
        initial_episodes=[
            SlotTransitionEpisode(
                "am1",
                "alien_mod7_sum",
                (("a", 1), ("b", 2), ("mod_k", 7), ("out", -1)),
                (("a", 1), ("b", 2), ("mod_k", 7), ("out", 3)),
            ),
            SlotTransitionEpisode(
                "am2",
                "alien_mod7_sum",
                (("a", 2), ("b", 4), ("mod_k", 7), ("out", -1)),
                (("a", 2), ("b", 4), ("mod_k", 7), ("out", 6)),
            ),
        ],
        candidate_probes=[
            {"a": 5, "b": 4, "mod_k": 7, "out": -1},  # wraps mod 7 -> 2!
            {"a": 6, "b": 6, "mod_k": 7, "out": -1},  # wraps mod 7 -> 5!
            {"a": 1, "b": 1, "mod_k": 7, "out": -1},
        ],
    )
    ff_alien = FunctionalFrontierState(
        frontier_id="alien_fn",
        domain_id="alien_fn_dom",
        skills=[fn_skill],
    )

    kb = ContinualKnowledgeBase()
    for dom_id in ("alien_quantum_dom", "alien_utf8_dom", "alien_fn_dom"):
        kb.replay_manager.register_domain_regression_cases(
            dom_id,
            [DomainRegressionCase(case_id=f"reg_{dom_id}", domain_id=dom_id, evaluator_fn=lambda _k: (True, "ok"))],
        )

    ctrl = AutonomousMetaController(
        kb=kb,
        budget=CurriculumBudget(max_env_queries=12, max_total_cpu_ms=4000.0, learning_slice_ms=120.0),
        scheduler=SelfCurriculumScheduler(),
        relational_frontiers=(rf_alien,),
        grounding_frontiers=(gf_alien,),
        functional_frontiers=(ff_alien,),
        relational_envs={"alien_rel": _CallableTransitionEnv(_alien_rel_step)},
        grounding_envs={"alien_gnd": _alien_gnd_env},
        functional_envs={("alien_fn", "alien_mod7_sum"): _alien_mod7_env},
    )
    ctrl.initialize_cold_start_frontiers(phase_index=1)

    for _ in range(14):
        ctrl.run_tick(phase_index=1)
    ctrl.freeze()

    # Stopped querying early (< 12 queries) and emitted idle_noop once all 3 alien frontiers converged!
    assert ctrl.budget.queries_spent < 12
    assert "idle_noop" in ctrl._observed_actions
    assert ctrl.post_freeze_env_queries == 0
    assert ctrl.self_certified_promotions == 0

    # Evaluate on unseen alien held-out entities & inputs!
    st_ho_warm = _mk_qstate("ho_w", cryo_a=True, cryo_b=False)
    st_ho_cool = _mk_qstate("ho_c", cryo_a=True, cryo_b=True)
    st_ho_cool_gold = _mk_qstate("ho_c", cryo_a=True, cryo_b=True, ent_ab=True)

    ans_warm, _ = ctrl.solve_interactive_now(
        InteractiveQueryRequest(
            request_id="aq1",
            domain_id="alien_quantum_dom",
            query_kind="relational_sequence",
            payload={
                "initial_state": st_ho_warm,
                "action_sequence": (
                    build_probe_action("entangle_qubit", {"src_q": "ho_w_qnode_1", "dst_q": "ho_w_qnode_2"}),
                ),
            },
            expected_answer=st_ho_warm,
        )
    )
    assert ans_warm == st_ho_warm

    # Grounding OOD with reversed role order ("dst zorpaj el src") + guard block on warm dst!
    ans_gnd_cool, _ = ctrl.solve_interactive_now(
        InteractiveQueryRequest(
            request_id="aq2",
            domain_id="alien_utf8_dom",
            query_kind="utf8_instruction",
            payload={
                "raw_bytes": "ho_c_qnode_2 zorpaj el ho_c_qnode_1.".encode("utf-8"),
                "scene_state": st_ho_cool,
            },
            expected_answer=st_ho_cool_gold,
        )
    )
    assert ans_gnd_cool == st_ho_cool_gold

    ans_gnd_warm, _ = ctrl.solve_interactive_now(
        InteractiveQueryRequest(
            request_id="aq3",
            domain_id="alien_utf8_dom",
            query_kind="utf8_instruction",
            payload={
                "raw_bytes": "ho_w_qnode_2 zorpaj el ho_w_qnode_1.".encode("utf-8"),
                "scene_state": st_ho_warm,
            },
            expected_answer=st_ho_warm,
        )
    )
    assert ans_gnd_warm == st_ho_warm

    # Functional OOD on unseen wrapping pair & unseen modulus (13, 19, mod 12) -> (13 + 19) % 12 == 8
    ans_fn, _ = ctrl.solve_interactive_now(
        InteractiveQueryRequest(
            request_id="aq4",
            domain_id="alien_fn_dom",
            query_kind="functional_slots",
            payload={
                "skill_id": "alien_mod7_sum",
                "before_slots": {"a": 13, "b": 19, "mod_k": 12, "out": -1},
            },
        )
    )
    assert ans_fn["out"] == (13 + 19) % 12


def test_s9_manifests_saved_5seed_gate_and_isolated_2seed_subset_run(tmp_path: Path) -> None:
    """Verify S0–S9 committed manifests, saved 5-seed S9 summary/traces, and isolated 2-seed tmp_path run."""
    prereg_bytes_before = (DATA_DIR / "s9_preregistration.json").read_bytes()
    summary_bytes_before = (REPORTS_DIR / "s9_summary.json").read_bytes()

    ok0, m0 = verify_committed_manifests(data_dir=DATA_DIR, seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    ok1, m1 = verify_s1_committed_manifests(data_dir=DATA_DIR, seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    ok2, m2 = verify_s2_committed_manifests(data_dir=DATA_DIR, seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    ok3, m3 = verify_s3_committed_manifests(data_dir=DATA_DIR, seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    ok4, m4 = verify_s4_committed_manifests(data_dir=DATA_DIR, seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    ok5, m5 = verify_s5_committed_manifests(data_dir=DATA_DIR, seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    ok6, m6 = verify_s6_committed_manifests(data_dir=DATA_DIR, seeds=DEFAULT_SEEDS)
    ok7, m7 = verify_s7_committed_manifests(data_dir=DATA_DIR, seeds=DEFAULT_SEEDS)
    ok8, m8 = verify_s8_committed_manifests(data_dir=DATA_DIR, seeds=DEFAULT_SEEDS)
    ok9, m9 = verify_s9_committed_manifests(data_dir=DATA_DIR, seeds=DEFAULT_SEEDS)
    assert all((ok0, ok1, ok2, ok3, ok4, ok5, ok6, ok7, ok8, ok9)), (
        m0 + m1 + m2 + m3 + m4 + m5 + m6 + m7 + m8 + m9
    )

    saved = json.loads((REPORTS_DIR / "s9_summary.json").read_text(encoding="utf-8"))
    assert saved["stage"] == "S9"
    assert saved["seeds"] == [101, 202, 303, 404, 505]
    assert saved["s9_gate"]["passed"] is True
    assert all(saved["s9_gate"]["checks"].values()), f"Failed checks: {saved['s9_gate']['checks']}"
    assert len(saved["s9_gate"]["checks"]) == 21

    auto_m = saved["controller_metrics"]["autonomous_meta_controller"]
    assert auto_m["relational_accuracy"]["ci95_low"] >= 0.95
    assert auto_m["relational_depth4_accuracy"]["ci95_low"] >= 0.90
    assert auto_m["relational_depth6_accuracy"]["ci95_low"] >= 0.90
    assert auto_m["utf8_grounding_accuracy"]["ci95_low"] >= 0.95
    assert auto_m["functional_math_list_accuracy"]["ci95_low"] >= 0.95
    assert auto_m["anchor_overall_retention"]["ci95_low"] >= 0.95
    assert auto_m["relational_collateral_rate"]["ci95_high"] < 0.02
    assert saved["lifts"]["over_passive_stream"]["ci95_low"] >= 0.15
    assert saved["lifts"]["over_random_curriculum"]["ci95_low"] >= 0.15
    assert saved["interactive_sla"]["concurrent_p95_ms"]["max"] <= 5.0
    assert saved["interactive_sla"]["starved_interactive_total"] == 0
    assert saved["interactive_sla"]["dropped_interactive_total"] == 0
    assert saved["mdl_compression_mean"]["active_library_bits"] < saved["mdl_compression_mean"]["cumulative_raw_training_bits"]
    assert saved["per_seed_seconds"]["max"] <= 15.0

    # Verify non-overlapping subsystem timers across saved 5-seed summary & CSV traces
    sub = saved["subsystem_cpu_ms_mean"]
    sub_sum = (
        sub["retrieval_ms"]
        + sub["induction_ms"]
        + sub["query_selection_ms"]
        + sub["verification_ms"]
        + sub["consolidation_ms"]
        + sub["replay_ms"]
    )
    assert sub_sum <= sub["total_learning_slice_ms"] + 1e-2

    csv_path = REPORTS_DIR / "raw_runs" / "s9_controller_traces.csv"
    assert csv_path.exists()
    with csv_path.open("r", encoding="utf-8", newline="") as f_csv:
        rows = list(csv.DictReader(f_csv))
    assert len(rows) == 5 * S9_AUTONOMOUS_MAX_TICKS
    for r in rows:
        row_sub = (
            float(r["retrieval_ms"])
            + float(r["induction_ms"])
            + float(r["query_selection_ms"])
            + float(r["verification_ms"])
            + float(r["consolidation_ms"])
            + float(r["replay_ms"])
        )
        assert row_sub <= float(r["learning_slice_ms"]) + 1e-2

    for s in DEFAULT_SEEDS:
        raw_p = REPORTS_DIR / "raw_runs" / f"s9_seed_{s}.json"
        assert raw_p.exists()
        rec_s = json.loads(raw_p.read_text(encoding="utf-8"))
        ac = rec_s["controllers"]["autonomous_meta_controller"]
        assert set(ac["observed_loop_steps"]) == set(CLOSED_LOOP_L0_L7_STEPS)
        assert ac["safety_invariants"]["self_certified_promotions"] == 0
        assert ac["safety_invariants"]["post_freeze_env_queries"] == 0

    # Isolated 2-seed subset run in tmp_path
    sub_summary = run_s9_experiments(
        seeds=[101, 202],
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        verify_prior_manifests=True,
    )
    sub_checks = sub_summary["s9_gate"]["checks"]
    for det_key in (
        "ast_isolation_and_zero_hardcoded_literals",
        "prior_s0_through_s8_manifests_intact",
        "s9_manifest_verified",
        "relational_frontier_final_acc_ge_095",
        "relational_depth4_and_depth6_acc_ge_090",
        "utf8_grounding_frontier_final_acc_ge_095",
        "functional_math_list_frontier_final_acc_ge_095",
        "strictly_lower_queries_to_target_than_random",
        "stops_querying_verified_frontiers_early",
        "anchor_skill_retention_ge_095_zero_forgetting",
        "faulty_candidate_caught_and_quarantined",
        "zero_self_certified_promotions",
        "freeze_enforced_zero_post_freeze_queries",
        "collateral_damage_below_002",
        "interactive_non_starvation_sla_p95_le_5ms",
        "learning_slices_within_tick_budget",
        "closed_loop_11_steps_and_mdl_consolidation_verified",
    ):
        assert sub_checks[det_key] is True, f"2-seed check failed: {det_key} -> {sub_checks}"
    assert sub_summary["lifts"]["over_passive_stream"]["min"] >= 0.50
    assert sub_summary["lifts"]["over_random_curriculum"]["min"] >= 0.45

    # Committed files untouched by isolated subset run
    assert (DATA_DIR / "s9_preregistration.json").read_bytes() == prereg_bytes_before
    assert (REPORTS_DIR / "s9_summary.json").read_bytes() == summary_bytes_before
