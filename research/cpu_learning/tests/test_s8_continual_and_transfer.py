"""Stage S8 Independent Adversarial Tests: Continual Learning, Regression Replay & Cross-Domain Transfer.

STRICT ISOLATION & PYTHON 3.11 COMPATIBILITY:
- Pure Python 3.11 stdlib ONLY (zero Python 3.12+ APIs).
- NEVER imports ``hidden_actions`` or ``oracle``.
- Audits AST/lexical isolation, Python 3.11 compatibility across the entire package,
  ``SmallCPUNeuralContinualRef`` isolation & analytical backprop gradients, fail-closed
  mandatory regression replay, cross-domain downstream regression catching, quarantined
  rollback refusal, genuine immediate post-phase learning before forgetting on both
  baselines, 100%-alien synthetic cross-domain positive & scoped negative transfer,
  and S0–S8 manifests + saved 5-seed & isolated 2-seed subset runs.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from research.cpu_learning.continual import (
    ContinualKnowledgeBase,
    ContinualLearnerFrozenError,
    CrossDomainAbstractionRegistry,
    CrossDomainTransferEvaluator,
    DomainRegressionCase,
    NaiveOverwriteNoReplayBaseline,
    SmallCPUNeuralContinualRef,
    encode_hashed_feature_vector,
)
from research.cpu_learning.evaluation.run_experiments import (
    DATA_DIR,
    PACKAGE_ROOT,
    REPORTS_DIR,
    verify_committed_manifests,
)
from research.cpu_learning.evaluation.run_s1_experiments import verify_s1_committed_manifests
from research.cpu_learning.evaluation.run_s2_experiments import (
    _FORBIDDEN_ITL_HARDCODED_LITERALS,
    _string_constants_in_code,
    verify_s2_committed_manifests,
)
from research.cpu_learning.evaluation.run_s3_experiments import verify_s3_committed_manifests
from research.cpu_learning.evaluation.run_s4_experiments import verify_s4_committed_manifests
from research.cpu_learning.evaluation.run_s5_experiments import (
    build_domain2_arcane_library,
    verify_s5_committed_manifests,
)
from research.cpu_learning.evaluation.run_s6_experiments import verify_s6_committed_manifests
from research.cpu_learning.evaluation.run_s7_experiments import (
    _EXTRA_D2_D3_FORBIDDEN_LITERALS,
    verify_s7_committed_manifests,
)
from research.cpu_learning.evaluation.run_s8_experiments import (
    _EXTRA_D4_D5_FORBIDDEN_LITERALS,
    PHASE_DOMAIN_IDS,
    _run_single_s8_seed,
    _verify_neural_mlp_analytical_gradients,
    audit_s8_ast_isolation_and_agnosticism,
    generate_domain4_biolab_few_shot_and_ood,
    generate_domain5_incompatible_reactor_suite,
    run_s8_experiments,
    verify_s8_committed_manifests,
)
from research.cpu_learning.grounding import (
    GroundedInstructionExecutor,
    RawByteConstructionLearner,
    UnlabelledGroundingEpisode,
)
from research.cpu_learning.itl.contrast_harness import build_probe_action
from research.cpu_learning.itl.operator_card import (
    AttributeGuard,
    ConditionalRoleFactGuard,
    OperatorCard,
    RoleInequalityGuard,
)
from research.cpu_learning.simulator.state_graph import (
    ActionInvocation,
    Entity,
    EvaluationTaskPublic,
    ObservationMask,
    StateGraph,
    TrainingEpisode,
    compute_sparse_delta,
)


def test_s8_ast_isolation_zero_domain_literals_neural_isolation_and_python311_compat() -> None:
    """Verify AST isolation, zero S0..D5 literals in continual/, neural isolation, and Python 3.11 stdlib compatibility."""
    audit = audit_s8_ast_isolation_and_agnosticism(PACKAGE_ROOT)
    assert audit["passed"] is True, f"S8 AST audit failed: {audit}"
    assert audit["import_violations"] == []
    assert audit["hardcode_violations"] == []
    assert audit["neural_leak_violations"] == []
    assert audit["scanned_continual_modules"] >= 3

    assert SmallCPUNeuralContinualRef.comparison_only is True
    assert SmallCPUNeuralContinualRef.uses_backprop is True
    assert NaiveOverwriteNoReplayBaseline.comparison_only is False
    assert NaiveOverwriteNoReplayBaseline.uses_backprop is False

    forbidden_all = (
        set(_FORBIDDEN_ITL_HARDCODED_LITERALS)
        | set(_EXTRA_D2_D3_FORBIDDEN_LITERALS)
        | set(_EXTRA_D4_D5_FORBIDDEN_LITERALS)
    )
    for fpath in sorted((PACKAGE_ROOT / "continual").rglob("*.py")):
        consts = _string_constants_in_code(fpath.read_text(encoding="utf-8"))
        assert (consts & forbidden_all) == set(), f"Hardcoded literal in {fpath.name}"

    # Python 3.11 compatibility scan across ALL .py files in research/cpu_learning
    py312_violations: list[str] = []
    for fpath in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = fpath.relative_to(PACKAGE_ROOT).as_posix()
        tree = ast.parse(fpath.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if hasattr(ast, "TypeAlias") and isinstance(node, ast.TypeAlias):
                py312_violations.append(f"{rel}: PEP 695 TypeAlias")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and getattr(
                node, "type_params", ()
            ):
                py312_violations.append(f"{rel}: PEP 695 type_params on {node.name}")
            if isinstance(node, ast.Attribute) and node.attr in {"sumprod", "batched"}:
                py312_violations.append(f"{rel}: Python 3.12+ .{node.attr}")
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "UTC"
                and isinstance(node.value, ast.Name)
                and node.value.id == "datetime"
            ):
                py312_violations.append(f"{rel}: Python 3.11+ datetime.UTC alias")
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "walk"
                and not (isinstance(node.value, ast.Name) and node.value.id in {"ast", "os"})
            ):
                py312_violations.append(f"{rel}: potential Python 3.12+ Path.walk")
    assert py312_violations == [], f"Python 3.12+ compatibility violations: {py312_violations}"


def test_s8_kb_freeze_fail_closed_replay_and_quarantined_rollback_guard() -> None:
    """Verify fail-closed replay, non-destructive history, Defect #1 quarantined rollback refusal, and freeze guards."""
    kb = ContinualKnowledgeBase()
    v1_entry = kb.register_initial_skill(
        domain_id="dom_alpha",
        skill_id="calc_fn",
        kind="fn",
        executable_dict={"op": "double", "k": 2},
        executable_object=lambda x: x * 2,
        phase_index=1,
    )
    assert v1_entry.version == 1

    # Fail-closed check: updating before registering regression cases for dom_alpha MUST fail & quarantine!
    unprotected_out = kb.apply_candidate_update_with_replay(
        domain_id="dom_alpha",
        skill_id="calc_fn",
        kind="fn",
        executable_dict={"op": "double_alt", "k": 2},
        executable_object=lambda x: x + x,
        phase_index=1,
    )
    assert unprotected_out.accepted is False
    assert unprotected_out.quarantined is True
    assert unprotected_out.rolled_back is True
    assert kb.get_active_skill("dom_alpha", "calc_fn") == v1_entry

    # Register regression suite for dom_alpha
    kb.replay_manager.register_domain_regression_cases(
        "dom_alpha",
        [
            DomainRegressionCase(
                case_id="alpha_c1",
                domain_id="dom_alpha",
                evaluator_fn=lambda cur: (
                    cur.get_active_domain_objects("dom_alpha")["calc_fn"](7) == 14,
                    "check_7_to_14",
                ),
            )
        ],
    )
    with pytest.raises(ValueError, match="Duplicate regression case_id"):
        kb.replay_manager.register_domain_regression_cases(
            "dom_alpha",
            [DomainRegressionCase("alpha_c1", "dom_alpha", lambda _c: (True, "dup"))],
        )

    # Valid replay-passing v3 update promotes cleanly
    good_out = kb.apply_candidate_update_with_replay(
        domain_id="dom_alpha",
        skill_id="calc_fn",
        kind="fn",
        executable_dict={"op": "shift_left", "shift": 1},
        executable_object=lambda x: x << 1,
        phase_index=2,
    )
    assert good_out.accepted is True and good_out.candidate_version == 3
    assert kb.get_active_skill("dom_alpha", "calc_fn").version == 3

    # Poisoned v4 update fails replay, rolls back to v3, and logs quarantine
    bad_out = kb.apply_candidate_update_with_replay(
        domain_id="dom_alpha",
        skill_id="calc_fn",
        kind="fn",
        executable_dict={"op": "corrupt_zero"},
        executable_object=lambda _x: 0,
        phase_index=3,
    )
    assert bad_out.accepted is False and bad_out.quarantined is True and bad_out.rolled_back is True
    assert bad_out.active_version_after == 3
    assert "alpha_c1" in bad_out.quarantine_record.offending_case_ids

    # Defect #1 regression: rollback_skill MUST refuse quarantined v2 and v4!
    for q_ver in (2, 4):
        with pytest.raises(ValueError, match="quarantined"):
            kb.rollback_skill("dom_alpha", "calc_fn", target_version=q_ver)

    # Explicit rollback to verified parent v1 succeeds and preserves all 4 archived versions
    restored_v1 = kb.rollback_skill("dom_alpha", "calc_fn", target_version=1)
    assert restored_v1.version == 1 and restored_v1.lifecycle_state == "promoted"
    audit = kb.audit_non_destructive_hashes()
    assert audit["rewritten_versions"] == 0
    assert audit["deleted_versions"] == 0
    assert audit["total_versions_archived"] == 4

    # Freeze blocks ALL mutating methods
    kb.freeze()
    assert kb.is_frozen is True and kb.replay_manager.is_frozen is True
    with pytest.raises(ContinualLearnerFrozenError):
        kb.register_initial_skill(
            domain_id="dom_beta",
            skill_id="s",
            kind="fn",
            executable_dict={"a": 1},
            executable_object=lambda: 1,
            phase_index=4,
        )
    with pytest.raises(ContinualLearnerFrozenError):
        kb.apply_candidate_update_with_replay(
            domain_id="dom_alpha",
            skill_id="calc_fn",
            kind="fn",
            executable_dict={"a": 2},
            executable_object=lambda x: x * 2,
            phase_index=4,
        )
    with pytest.raises(ContinualLearnerFrozenError):
        kb.rollback_skill("dom_alpha", "calc_fn", target_version=3)
    with pytest.raises(ContinualLearnerFrozenError):
        kb.set_domain_aux("dom_alpha", "k", "v")
    with pytest.raises(ContinualLearnerFrozenError):
        kb.replay_manager.register_domain_regression_cases("dom_alpha", [])


def test_s8_cross_domain_regression_replay_catches_downstream_utf8_breakage() -> None:
    """Verify modifying a Phase-1 relational operator that breaks a downstream UTF-8 grounding case is auto-rolled back."""
    ents = {
        "sage_1": Entity("sage_1", "sage", ()),
        "pylon_1": Entity("pylon_1", "pylon", ()),
        "pylon_warded": Entity("pylon_warded", "pylon", ()),
    }
    w0 = StateGraph.create(
        ents,
        {
            ("pylon_1", "lit", "false"),
            ("pylon_1", "warded", "false"),
            ("pylon_warded", "lit", "false"),
            ("pylon_warded", "warded", "true"),
        },
    )
    good_ignite = OperatorCard(
        card_id="ignite_v1",
        action_id="ignite",
        role_variables=("actor", "target"),
        required_types=(("actor", frozenset({"sage"})), ("target", frozenset({"pylon"}))),
        required_facts=(("?target", "lit", "false"), ("?target", "warded", "false")),
        conditional_fact_guards=(),
        indirect_fact_guards=(),
        attribute_guards=(),
        inequality_guards=(),
        add_facts=(("?target", "lit", "true"),),
        delete_facts=(("?target", "lit", "false"),),
        attribute_edits=(),
        support_positive=3,
    )
    af_good = good_ignite.apply(w0, {"actor": "sage_1", "target": "pylon_1"}).after_state
    assert af_good is not None

    kb = ContinualKnowledgeBase()
    kb.register_initial_skill(
        domain_id="rel_dom",
        skill_id="ignite",
        kind="relational_operator",
        executable_dict=good_ignite.to_executable_dict(),
        executable_object=good_ignite,
        phase_index=1,
    )
    # Domain 1 local regression only tests positive ignition on unwarded pylon_1
    kb.replay_manager.register_domain_regression_cases(
        "rel_dom",
        [
            DomainRegressionCase(
                case_id="rel_pos_only",
                domain_id="rel_dom",
                evaluator_fn=lambda cur: (
                    cur.get_active_domain_objects("rel_dom")["ignite"]
                    .apply(w0, {"actor": "sage_1", "target": "pylon_1"})
                    .after_state
                    == af_good,
                    "pos_ignite_ok",
                ),
            )
        ],
    )

    # Domain 2: UTF-8 grounding learned on top of rel_dom, with regression case testing warded pylon stays blocked!
    g_learner = RawByteConstructionLearner({"ignite": good_ignite})
    g_learner.fit_episodes(
        [UnlabelledGroundingEpisode("u1", b"sage_1 sparks pylon_1", w0, af_good)]
    )
    cx = g_learner.promoted_constructions[0]
    kb.register_initial_skill(
        domain_id="utf8_dom",
        skill_id=cx.construction_id,
        kind="utf8_construction",
        executable_dict=cx.to_executable_dict(),
        executable_object=cx,
        phase_index=2,
    )

    def _utf8_ward_replay(cur: ContinualKnowledgeBase) -> tuple[bool, str]:
        ops = cur.get_active_domain_objects("rel_dom", kind="relational_operator")
        cxs = cur.get_active_domain_objects("utf8_dom", kind="utf8_construction")
        gl = RawByteConstructionLearner(ops)
        for c_obj in cxs.values():
            gl._constructions_by_pattern.setdefault(c_obj.pattern_tokens, []).append(c_obj)
        gl.freeze()
        res = GroundedInstructionExecutor(gl).execute_raw_utf8(b"sage_1 sparks pylon_warded", w0)
        ok = (not res.abstained) and (res.final_state == w0)
        return (ok, "warded_pylon_must_remain_unlit")

    kb.replay_manager.register_domain_regression_cases(
        "utf8_dom",
        [DomainRegressionCase("utf8_ward_guard", "utf8_dom", _utf8_ward_replay)],
    )

    # Candidate update to rel_dom drops `warded == false`: passes rel_dom's local check, FAILS utf8_dom!
    overgen_ignite = OperatorCard.from_dict(
        {**good_ignite.to_dict(), "required_facts": [["?target", "lit", "false"]]}
    )
    out = kb.apply_candidate_update_with_replay(
        domain_id="rel_dom",
        skill_id="ignite",
        kind="relational_operator",
        executable_dict=overgen_ignite.to_executable_dict(),
        executable_object=overgen_ignite,
        phase_index=3,
    )
    assert out.accepted is False and out.rolled_back is True and out.quarantined is True
    assert out.replay_on_candidate.failed_domains == ("utf8_dom",)
    assert out.replay_after_resolution.passed is True
    assert (
        kb.get_active_domain_objects("rel_dom")["ignite"].to_executable_dict()
        == good_ignite.to_executable_dict()
    )


def test_s8_baselines_learn_each_phase_first_and_analytical_mlp_gradients() -> None:
    """Verify Defect #2 regression: both baselines genuinely learn each phase first before forgetting, plus MLP grad check."""
    assert _verify_neural_mlp_analytical_gradients(101) < 1e-5

    mlp = SmallCPUNeuralContinualRef(seed=101, input_dim=16, hidden_dim=8, num_classes=4, lr=0.15)
    p1_data = [
        (encode_hashed_feature_vector((f"feat_{i % 2}", "shared_ctx"), dim=16), i % 2)
        for i in range(20)
    ]
    p2_data = [
        (encode_hashed_feature_vector((f"feat_{i % 2}", "shared_ctx", "phase2"), dim=16), 2 + (i % 2))
        for i in range(24)
    ]
    mlp.fit_phase_sequential(p1_data, epochs=18, phase_seed_offset=1)
    acc_p1_after_p1 = mlp.evaluate_accuracy(p1_data)
    assert acc_p1_after_p1 >= 0.90

    mlp.fit_phase_sequential(p2_data, epochs=25, phase_seed_offset=2)
    acc_p2_after_p2 = mlp.evaluate_accuracy(p2_data)
    acc_p1_after_p2 = mlp.evaluate_accuracy(p1_data)
    assert acc_p2_after_p2 >= 0.90
    assert acc_p1_after_p2 < acc_p1_after_p1  # genuine catastrophic interference without replay

    mlp.freeze()
    with pytest.raises(ContinualLearnerFrozenError):
        mlp.fit_phase_sequential(p1_data)

    # Verify saved seed 101 diagonal immediate post-phase learning on BOTH baselines across Phases 1..4
    s101 = json.loads((REPORTS_DIR / "raw_runs" / "s8_seed_101.json").read_text(encoding="utf-8"))
    rm = s101["retention_matrix"]
    for idx_p, dom_id in enumerate(PHASE_DOMAIN_IDS, start=1):
        p_key = f"phase_{idx_p}"
        assert rm[p_key]["naive_overwrite_no_replay"][dom_id] >= 0.95, f"Naive failed to learn {dom_id}"
        assert rm[p_key]["small_cpu_neural_continual_ref"][dom_id] >= 0.75, f"MLP failed to learn {dom_id}"
    assert s101["final_mean_retention"]["naive_overwrite_no_replay"] < 0.40
    assert s101["final_mean_retention"]["small_cpu_neural_continual_ref"] < 0.40


def test_s8_synthetic_alien_domains_positive_transfer_and_mechanistic_scoped_rejection() -> None:
    """Verify CrossDomainAbstractionRegistry on a 100%-alien domain pair + alien 3rd target + alien incompatible domain."""
    # Alien Source Domain A (Astro Navigation) & Alien Source Domain B (Deep Submersible)
    # Share isomorphic guarded relocation schema + 1 Domain-A-only arity-2 beacon schema!
    astro_warp = OperatorCard(
        card_id="astro_warp_v1",
        action_id="warp_pod",
        role_variables=("pilot", "pod", "orig_sec", "dest_sec"),
        required_types=(
            ("pilot", frozenset({"navigator"})),
            ("pod", frozenset({"craft"})),
            ("orig_sec", frozenset({"orbit_pad", "hangar_ring"})),
            ("dest_sec", frozenset({"orbit_pad", "hangar_ring"})),
        ),
        required_facts=(
            ("?pod", "piloted_by", "?pilot"),
            ("?pod", "moored_at", "?orig_sec"),
            ("?dest_sec", "nav_clear", "true"),
        ),
        conditional_fact_guards=(
            ConditionalRoleFactGuard("orig_sec", "hangar_ring", "bay_shield_open", "true"),
            ConditionalRoleFactGuard("dest_sec", "hangar_ring", "bay_shield_open", "true"),
        ),
        indirect_fact_guards=(),
        attribute_guards=(AttributeGuard("pod", "fuel_ok", True),),
        inequality_guards=(RoleInequalityGuard("orig_sec", "dest_sec"),),
        add_facts=(("?pod", "moored_at", "?dest_sec"),),
        delete_facts=(("?pod", "moored_at", "?orig_sec"),),
        attribute_edits=(),
        support_positive=6,
    )
    astro_ping = OperatorCard(
        card_id="astro_ping_v1",
        action_id="arm_transponder",
        role_variables=("pilot", "beacon"),
        required_types=(("pilot", frozenset({"navigator"})), ("beacon", frozenset({"star_beacon"}))),
        required_facts=(("?beacon", "armed", "false"), ("?beacon", "jammed_rf", "false")),
        conditional_fact_guards=(),
        indirect_fact_guards=(),
        attribute_guards=(),
        inequality_guards=(),
        add_facts=(("?beacon", "armed", "true"),),
        delete_facts=(("?beacon", "armed", "false"),),
        attribute_edits=(),
        support_positive=6,
    )

    sub_dive = OperatorCard(
        card_id="sub_dive_v1",
        action_id="shift_bathysphere",
        role_variables=("captain", "vessel", "from_trench", "to_trench"),
        required_types=(
            ("captain", frozenset({"diver"})),
            ("vessel", frozenset({"sub_hull"})),
            ("from_trench", frozenset({"reef_shelf", "airlock_dome"})),
            ("to_trench", frozenset({"reef_shelf", "airlock_dome"})),
        ),
        required_facts=(
            ("?vessel", "helmed_by", "?captain"),
            ("?vessel", "anchored_in", "?from_trench"),
            ("?to_trench", "sonar_safe", "true"),
        ),
        conditional_fact_guards=(
            ConditionalRoleFactGuard("from_trench", "airlock_dome", "floodgate_open", "true"),
            ConditionalRoleFactGuard("to_trench", "airlock_dome", "floodgate_open", "true"),
        ),
        indirect_fact_guards=(),
        attribute_guards=(AttributeGuard("vessel", "hull_intact", True),),
        inequality_guards=(RoleInequalityGuard("from_trench", "to_trench"),),
        add_facts=(("?vessel", "anchored_in", "?to_trench"),),
        delete_facts=(("?vessel", "anchored_in", "?from_trench"),),
        attribute_edits=(),
        support_positive=6,
    )

    reg = CrossDomainAbstractionRegistry()
    rep = reg.register_source_domain_pair(
        "alien_astro",
        {"warp_pod": astro_warp, "arm_transponder": astro_ping},
        "alien_sub",
        {"shift_bathysphere": sub_dive},
    )
    assert rep.reversibility_verified is True
    assert len(reg.universal_entries()) == 1
    assert len(reg.domain_local_entries()) == 1
    assert reg.domain_local_entries()[0].scope == "domain_local"

    evaluator = CrossDomainTransferEvaluator(reg)

    # Alien Target Domain C (Mars Rover — 2 positive demos, 0 negatives -> tests symmetric garage guard transfer!)
    def _mars_world(
        tag: str, *, r_loc: str, pad2_safe: bool = True, gar_open: bool = True, battery: bool = True
    ) -> tuple[StateGraph, dict[str, str]]:
        ids = {"c": f"{tag}_cmdr", "rv": f"{tag}_rover", "p1": f"{tag}_crater1", "p2": f"{tag}_crater2", "g1": f"{tag}_hab_gar"}
        ents = {
            ids["c"]: Entity(ids["c"], "commander", ()),
            ids["rv"]: Entity(ids["rv"], "rover_unit", (("charged_cell", battery), ("paint", tag))),
            ids["p1"]: Entity(ids["p1"], "regolith_pad", ()),
            ids["p2"]: Entity(ids["p2"], "regolith_pad", ()),
            ids["g1"]: Entity(ids["g1"], "hab_garage", ()),
        }
        facts = {
            (ids["rv"], "assigned_cmdr", ids["c"]),
            (ids["rv"], "parked_on", ids[r_loc]),
            (ids["p1"], "terrain_passable", "true"),
            (ids["p2"], "terrain_passable", "true" if pad2_safe else "false"),
            (ids["g1"], "terrain_passable", "true"),
            (ids["g1"], "airseal_open", "true" if gar_open else "false"),
        }
        return StateGraph.create(ents, facts), ids

    mask = ObservationMask(fully_observed=True)
    w_m1, id_m1 = _mars_world("m_tr1", r_loc="p1")
    act_m1 = build_probe_action("drive_rover", {"cmdr": id_m1["c"], "rover": id_m1["rv"], "src": id_m1["p1"], "dst": id_m1["p2"]})
    af_m1 = w_m1.with_changes(delete_facts=((id_m1["rv"], "parked_on", id_m1["p1"]),), add_facts=((id_m1["rv"], "parked_on", id_m1["p2"]),))

    w_m2, id_m2 = _mars_world("m_tr2", r_loc="p1", gar_open=True)
    act_m2 = build_probe_action("drive_rover", {"cmdr": id_m2["c"], "rover": id_m2["rv"], "src": id_m2["p1"], "dst": id_m2["g1"]})
    af_m2 = w_m2.with_changes(delete_facts=((id_m2["rv"], "parked_on", id_m2["p1"]),), add_facts=((id_m2["rv"], "parked_on", id_m2["g1"]),))

    mars_train = [
        TrainingEpisode("mt1", w_m1, act_m1, mask, af_m1, compute_sparse_delta(w_m1, af_m1), (), "sim", "tr", "gen", "v1", 101, "2026-01-01T00:00:00Z"),
        TrainingEpisode("mt2", w_m2, act_m2, mask, af_m2, compute_sparse_delta(w_m2, af_m2), (), "sim", "tr", "gen", "v1", 101, "2026-01-01T00:00:01Z"),
    ]
    # Unseen OOD tests: valid garage->pad, blocked closed-garage origin, blocked closed-garage dest, blocked impassable pad2
    w_o1, io1 = _mars_world("mo1", r_loc="g1", gar_open=True)
    a_o1 = build_probe_action("drive_rover", {"cmdr": io1["c"], "rover": io1["rv"], "src": io1["g1"], "dst": io1["p2"]})
    g_o1 = w_o1.with_changes(delete_facts=((io1["rv"], "parked_on", io1["g1"]),), add_facts=((io1["rv"], "parked_on", io1["p2"]),))

    w_o2, io2 = _mars_world("mo2", r_loc="g1", gar_open=False)
    a_o2 = build_probe_action("drive_rover", {"cmdr": io2["c"], "rover": io2["rv"], "src": io2["g1"], "dst": io2["p2"]})

    w_o3, io3 = _mars_world("mo3", r_loc="p1", gar_open=False)
    a_o3 = build_probe_action("drive_rover", {"cmdr": io3["c"], "rover": io3["rv"], "src": io3["p1"], "dst": io3["g1"]})

    w_o4, io4 = _mars_world("mo4", r_loc="p1", pad2_safe=False)
    a_o4 = build_probe_action("drive_rover", {"cmdr": io4["c"], "rover": io4["rv"], "src": io4["p1"], "dst": io4["p2"]})

    mars_eval = [
        {"initial_state": w_o1, "actions": (a_o1,), "gold_final_state": g_o1},
        {"initial_state": w_o2, "actions": (a_o2,), "gold_final_state": w_o2},
        {"initial_state": w_o3, "actions": (a_o3,), "gold_final_state": w_o3},
        {"initial_state": w_o4, "actions": (a_o4,), "gold_final_state": w_o4},
    ]
    mars_rep, _, _ = evaluator.evaluate_target_domain(
        seed=101, target_domain_id="alien_mars", train_episodes=mars_train, eval_cases=mars_eval
    )
    assert mars_rep.mirror_transfer_exact_acc == 1.0
    assert mars_rep.accuracy_lift_over_cold_start > 0.20
    assert mars_rep.negative_transfer_rate == 0.0

    # Alien Incompatible Domain D (Quantum Qubit Attribute Calibration — arity 2 attribute edit, incompatible with astro_ping!)
    def _qubit_world(tag: str, *, tuned: bool, cooled: bool) -> tuple[StateGraph, dict[str, str]]:
        ids = {"ph": f"{tag}_phys", "qb": f"{tag}_qubit"}
        ents = {
            ids["ph"]: Entity(ids["ph"], "physicist", ()),
            ids["qb"]: Entity(ids["qb"], "qubit_chip", (("phase_tuned", tuned),)),
        }
        facts = {(ids["qb"], "cryo_cooled", "true" if cooled else "false")}
        return StateGraph.create(ents, facts), ids

    q_train: list[TrainingEpisode] = []
    for idx_q, (tn, cl) in enumerate(((False, True), (False, True), (False, False), (True, True))):
        wq, iq = _qubit_world(f"qt_{idx_q}", tuned=tn, cooled=cl)
        aq = build_probe_action("tune_qubit", {"physicist": iq["ph"], "chip": iq["qb"]})
        afq = wq.with_changes(attribute_edits=((iq["qb"], "phase_tuned", True),)) if (not tn and cl) else wq.copy()
        q_train.append(
            TrainingEpisode(f"qe_{idx_q}", wq, aq, mask, afq, compute_sparse_delta(wq, afq), (), "sim", "tr", "gen", "v1", 101, "2026-01-01T00:00:00Z")
        )
    wq_eval1, iqe1 = _qubit_world("qe_pos", tuned=False, cooled=True)
    aq_eval1 = build_probe_action("tune_qubit", {"physicist": iqe1["ph"], "chip": iqe1["qb"]})
    gq_eval1 = wq_eval1.with_changes(attribute_edits=((iqe1["qb"], "phase_tuned", True),))
    wq_eval2, iqe2 = _qubit_world("qe_neg", tuned=False, cooled=False)
    aq_eval2 = build_probe_action("tune_qubit", {"physicist": iqe2["ph"], "chip": iqe2["qb"]})

    q_rep, q_cards, _ = evaluator.evaluate_target_domain(
        seed=101,
        target_domain_id="alien_qubit",
        train_episodes=q_train,
        eval_cases=[
            {"initial_state": wq_eval1, "actions": (aq_eval1,), "gold_final_state": gq_eval1},
            {"initial_state": wq_eval2, "actions": (aq_eval2,), "gold_final_state": wq_eval2},
        ],
    )
    assert q_rep.mirror_transfer_exact_acc == 1.0
    assert q_rep.negative_transfer_rate == 0.0
    assert q_rep.ungated_negative_transfer_rate > 0.0  # mechanistic mismatch from astro_ping!

    cand_q = reg.evaluate_candidate_for_universal_promotion(
        "alien_qubit", q_cards["tune_qubit"], {"warp_pod": astro_warp}
    )
    assert cand_q.promoted_universal is False and cand_q.scope == "domain_local"

    reg.freeze()
    with pytest.raises(ContinualLearnerFrozenError):
        reg.evaluate_candidate_for_universal_promotion("alien_qubit", q_cards["tune_qubit"], {})


def test_s8_domain4_biolab_and_domain5_reactor_suites_on_seed101() -> None:
    """Verify Domain 4 (Biolab 2-shot) and Domain 5 (Reactor scoped rejection) suites on seed 101."""
    d4_tr, d4_ood = generate_domain4_biolab_few_shot_and_ood(101)
    d5_tr, d5_ood = generate_domain5_incompatible_reactor_suite(101)
    assert len(d4_tr) == 4 and len(d4_ood) == 44
    assert len(d5_tr) == 10 and len(d5_ood) == 24

    s101 = json.loads((REPORTS_DIR / "raw_runs" / "s8_seed_101.json").read_text(encoding="utf-8"))
    xd = s101["cross_domain_transfer"]
    assert xd["domain3_logistics"]["mirror_transfer_exact_acc"] == 1.0
    assert xd["domain4_biolab"]["mirror_transfer_exact_acc"] == 1.0
    assert xd["domain4_biolab"]["accuracy_lift_over_cold_start"] >= 0.30
    assert xd["domain5_incompatible_reactor"]["mirror_transfer_exact_acc"] == 1.0
    assert xd["domain5_incompatible_reactor"]["negative_transfer_rate"] == 0.0
    assert xd["domain5_incompatible_reactor"]["ungated_negative_transfer_rate"] >= 0.40


def test_s8_single_seed_5phase_e2e_poison_rollbacks_and_sublinear_growth() -> None:
    """Run _run_single_s8_seed(101) live and verify 0 forgetting, 3/3 poison rollbacks, and sub-linear library growth."""
    rec, rows = _run_single_s8_seed(101)
    assert len(rows) >= 30
    assert rec["max_forgetting_drop"] == 0.0
    assert rec["hash_immutability_audit"]["rewritten_prior_skills"] == 0
    assert rec["freeze_guard_verified"] is True
    assert rec["poison_rollback_tests"]["caught_and_rolled_back_rate"] == 1.0
    assert rec["poison_rollback_tests"]["quarantine_count"] == 3

    fg = rec["final_growth"]
    assert fg["active_library_bits"] < fg["cumulative_compressed_training_bits"] < fg["cumulative_raw_training_bits"]
    assert fg["library_to_raw_ratio"] < 0.05


def test_s8_committed_manifests_saved_5seed_gate_and_fresh_isolated_subset_run(tmp_path: Path) -> None:
    """Verify S0..S8 committed manifests, saved 5-seed S8 summary gate, and isolated 2-seed subset run."""
    ok0, e0 = verify_committed_manifests(data_dir=DATA_DIR)
    ok1, e1 = verify_s1_committed_manifests(data_dir=DATA_DIR)
    ok2, e2 = verify_s2_committed_manifests(data_dir=DATA_DIR)
    ok3, e3 = verify_s3_committed_manifests(data_dir=DATA_DIR)
    ok4, e4 = verify_s4_committed_manifests(data_dir=DATA_DIR)
    ok5, e5 = verify_s5_committed_manifests(data_dir=DATA_DIR)
    ok6, e6 = verify_s6_committed_manifests(data_dir=DATA_DIR)
    ok7, e7 = verify_s7_committed_manifests(data_dir=DATA_DIR)
    ok8, e8 = verify_s8_committed_manifests(data_dir=DATA_DIR)
    assert ok0 and ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7 and ok8, (
        f"Manifest mismatch: {e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7 + e8}"
    )

    saved_summary_path = REPORTS_DIR / "s8_summary.json"
    saved_bytes_before = saved_summary_path.read_bytes()
    saved_summary = json.loads(saved_bytes_before.decode("utf-8"))

    assert saved_summary["stage"] == "S8"
    assert saved_summary["seeds"] == [101, 202, 303, 404, 505]
    assert saved_summary["s8_gate"]["passed"] is True
    assert all(saved_summary["s8_gate"]["checks"].values())
    assert len(saved_summary["s8_gate"]["checks"]) == 19

    tmp_data = tmp_path / "data"
    tmp_reports = tmp_path / "reports"
    fresh = run_s8_experiments(
        seeds=[101, 202],
        data_dir=tmp_data,
        reports_dir=tmp_reports,
        verify_prior_manifests=True,
    )
    assert fresh["s8_gate"]["checks"]["prior_s0_through_s7_manifests_intact"] is True
    assert fresh["s8_gate"]["checks"]["s8_manifest_verified"] is True
    for k, ok in fresh["s8_gate"]["checks"].items():
        # Student-t df=1 (t=12.706) widens random MLP baseline CI on n=2 seeds; check min/max directly below
        if k in {
            "cpu_budget_within_15s_per_seed",
            "mirror_strictly_beats_naive_overwrite_ci95_by_25pp",
            "mirror_strictly_beats_neural_continual_ref_ci95_by_25pp",
        }:
            continue
        assert ok is True, f"Fresh isolated 2-seed S8 gate failed check: {k}"

    mr = fresh["final_mean_retention"]
    assert mr["mirror_continual_kb"]["min"] > mr["naive_overwrite_no_replay"]["max"] + 0.50
    assert mr["mirror_continual_kb"]["min"] > mr["small_cpu_neural_continual_ref"]["max"] + 0.50

    assert (tmp_data / "s8_preregistration.json").exists()
    assert (tmp_reports / "s8_summary.json").exists()
    assert (tmp_reports / "s8_research_report.md").exists()
    assert (tmp_reports / "raw_runs" / "s8_continual_matrix.csv").exists()
    assert (tmp_reports / "raw_runs" / "s8_seed_101.json").exists()
    assert (tmp_reports / "raw_runs" / "s8_seed_202.json").exists()
    assert saved_summary_path.read_bytes() == saved_bytes_before
