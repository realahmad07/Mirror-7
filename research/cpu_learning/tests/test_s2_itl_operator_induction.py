"""Automated Stage S2 tests: ITL Operator Induction, Domain-Agnosticism, Partial Masks,
Zero Hardcoding / AST Isolation, Ablations, S1+S2 End-to-End Raw UTF-8, and 5-Seed Gates.

STRICT ISOLATION: Never imports ``hidden_actions`` or ``oracle``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.cpu_learning.evaluation.run_experiments import (
    REPORTS_DIR,
    evaluate_solver_on_seed,
)
from research.cpu_learning.evaluation.run_s2_experiments import (
    audit_s2_ast_isolation_and_agnosticism,
    run_s2_experiments,
    verify_s2_committed_manifests,
)
from research.cpu_learning.itl.operator_card import OperatorCard, TypeTaxonomy
from research.cpu_learning.itl.operator_inducer import ITLOperatorLearner
from research.cpu_learning.ledger.episode_ledger import EpisodeLedger
from research.cpu_learning.simulator.generators import (
    DEFAULT_DEMOS_PER_ACTION,
    DEFAULT_SEEDS,
    generate_heldout_suite,
    generate_training_episodes,
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


def _make_synthetic_ep(
    ep_id: str,
    before: StateGraph,
    after: StateGraph,
    action_id: str,
    roles: dict[str, str],
) -> TrainingEpisode:
    raw_text = f"{action_id} " + " ".join(f"{k}={v}" for k, v in sorted(roles.items()))
    raw_bytes = raw_text.encode("utf-8")
    act = ActionInvocation(
        action_id=action_id,
        roles=tuple(sorted(roles.items())),
        raw_instruction_bytes=raw_bytes,
        byte_spans=(("action", 0, len(action_id.encode("utf-8"))),),
    )
    mask = ObservationMask(fully_observed=True)
    return TrainingEpisode(
        episode_id=ep_id,
        before_state=before,
        action=act,
        observation_mask=mask,
        after_state=after,
        sparse_delta=compute_sparse_delta(before, after, mask=mask),
        context=(("domain", "synthetic_arcane_v1"),),
        observation_source="synthetic_unit_test",
        split_id="synthetic_train",
        generator_id="synth_v1",
        generator_version="v1",
        seed=999,
        timestamp="2026-11-22T00:00:00Z",
    )


def test_s2_ast_isolation_and_zero_hardcoded_domain_literals() -> None:
    audit = audit_s2_ast_isolation_and_agnosticism()
    assert audit["passed"] is True, f"AST audit failed: {audit}"
    assert audit["import_violations"] == []
    assert audit["hardcode_violations"] == []
    assert audit["scanned_itl_files"] >= 4
    assert audit["scanned_learner_files"] >= 9


def test_s2_domain_agnostic_novel_synthetic_world_and_actions() -> None:
    """Prove ITL induces exact preconditions, guards, and effects on a 100% novel domain
    with unseen action names, entity types, attributes, and predicates."""
    taxonomy = TypeTaxonomy.from_mapping({"shrine": frozenset({"altar", "sanctum"})})

    def make_world(
        suffix: str,
        *,
        charged0: bool = True,
        charged_dist: bool = False,
        sanctum_gate: str = "true",
        sanctum_ward: str = "false",
        altar1_open_path: str = "true",
        rune0_at_sanctum: bool = False,
        badge_color: str = "gold",
        heir_aura: str = "silver",
    ) -> tuple[StateGraph, dict[str, str]]:
        w0, w1 = f"mageA_{suffix}", f"mageB_{suffix}"
        r0, r1 = f"sigil0_{suffix}", f"sigil1_{suffix}"
        al0, al1 = f"altar0_{suffix}", f"altar1_{suffix}"
        sc0 = f"sanctum0_{suffix}"
        ents = [
            Entity(w0, "wizard", (("aura", badge_color), ("rank", "adept"))),
            Entity(w1, "wizard", (("aura", heir_aura), ("rank", "adept"))),
            Entity(r0, "rune", (("glyph", badge_color), ("charged", charged0))),
            Entity(r1, "rune", (("glyph", "indigo"), ("charged", charged_dist))),
            Entity(al0, "altar", (("stone", badge_color),)),
            Entity(al1, "altar", (("stone", "obsidian"),)),
            Entity(
                sc0,
                "sanctum",
                (("stone", badge_color), ("seal_label", "sealed" if sanctum_ward == "true" else "clear")),
            ),
        ]
        loc0 = sc0 if rune0_at_sanctum else al0
        facts = [
            (r0, "attuned_to", w0),
            (r0, "anchored_at", loc0),
            (r1, "attuned_to", w1),
            (r1, "anchored_at", al1),
            (al0, "reachable", "true"),
            (al1, "reachable", altar1_open_path),
            (sc0, "reachable", "true"),
            (sc0, "gate_open", sanctum_gate),
            (sc0, "ward_active", sanctum_ward),
        ]
        return StateGraph.create(ents, facts), {
            "w0": w0,
            "w1": w1,
            "r0": r0,
            "r1": r1,
            "al0": al0,
            "al1": al1,
            "sc0": sc0,
        }

    train_eps: list[TrainingEpisode] = []

    # Action 1: "channel" (relocates rune from origin -> dest across altars & sanctums)
    colors = ("gold", "ruby", "jade", "cobalt")
    for idx in range(4):
        use_sc_dst = idx == 2
        use_sc_src = idx == 3
        w, ids = make_world(
            f"ch{idx}",
            charged_dist=(idx % 2 == 0),
            sanctum_gate="true",
            sanctum_ward="false",
            rune0_at_sanctum=use_sc_src,
            badge_color=colors[idx],
        )
        src = ids["sc0"] if use_sc_src else ids["al0"]
        dst = ids["sc0"] if use_sc_dst else ids["al1"]
        after = w.with_changes(
            delete_facts=((ids["r0"], "anchored_at", src),),
            add_facts=((ids["r0"], "anchored_at", dst),),
        )
        train_eps.append(
            _make_synthetic_ep(
                f"ep_ch_pos_{idx}",
                w,
                after,
                "channel",
                {"caster": ids["w0"], "relic": ids["r0"], "origin": src, "dest": dst},
            )
        )
    # 2 negative demos for "channel": unreachable dest + closed sanctum dest
    w_n0, ids_n0 = make_world("chn0", altar1_open_path="false")
    train_eps.append(
        _make_synthetic_ep(
            "ep_ch_neg_0",
            w_n0,
            w_n0.copy(),
            "channel",
            {"caster": ids_n0["w0"], "relic": ids_n0["r0"], "origin": ids_n0["al0"], "dest": ids_n0["al1"]},
        )
    )
    w_n1, ids_n1 = make_world("chn1", sanctum_gate="false")
    train_eps.append(
        _make_synthetic_ep(
            "ep_ch_neg_1",
            w_n1,
            w_n1.copy(),
            "channel",
            {"caster": ids_n1["w0"], "relic": ids_n1["r0"], "origin": ids_n1["al0"], "dest": ids_n1["sc0"]},
        )
    )

    # Action 2: "bestow" (transfers attuned_to from caster -> heir; blocked if anchored_at closed sanctum)
    for idx in range(4):
        in_open_sc = idx == 3
        w, ids = make_world(
            f"bs{idx}",
            charged_dist=(idx % 2 == 0),
            sanctum_gate="true",
            sanctum_ward="false",
            rune0_at_sanctum=in_open_sc,
            badge_color=colors[idx],
        )
        after = w.with_changes(
            delete_facts=((ids["r0"], "attuned_to", ids["w0"]),),
            add_facts=((ids["r0"], "attuned_to", ids["w1"]),),
        )
        train_eps.append(
            _make_synthetic_ep(
                f"ep_bs_pos_{idx}",
                w,
                after,
                "bestow",
                {"caster": ids["w0"], "relic": ids["r0"], "heir": ids["w1"]},
            )
        )
    w_bn0, ids_bn0 = make_world("bsn0")
    train_eps.append(
        _make_synthetic_ep(
            "ep_bs_neg_0",
            w_bn0,
            w_bn0.copy(),
            "bestow",
            {"caster": ids_bn0["w1"], "relic": ids_bn0["r0"], "heir": ids_bn0["w0"]},
        )
    )
    w_bn1, ids_bn1 = make_world("bsn1", sanctum_gate="false", sanctum_ward="false", rune0_at_sanctum=True)
    train_eps.append(
        _make_synthetic_ep(
            "ep_bs_neg_1",
            w_bn1,
            w_bn1.copy(),
            "bestow",
            {"caster": ids_bn1["w0"], "relic": ids_bn1["r0"], "heir": ids_bn1["w1"]},
        )
    )

    # Action 3: "unward" (disables ward_active on a sanctum & edits seal_label="clear"; works whether gate is open or closed)
    for idx in range(4):
        w, ids = make_world(
            f"uw{idx}",
            sanctum_gate="true" if idx % 2 == 0 else "false",
            sanctum_ward="true",
            badge_color=colors[idx],
        )
        after = w.with_changes(
            delete_facts=((ids["sc0"], "ward_active", "true"),),
            add_facts=((ids["sc0"], "ward_active", "false"),),
            attribute_edits=((ids["sc0"], "seal_label", "clear"),),
        )
        train_eps.append(
            _make_synthetic_ep(
                f"ep_uw_pos_{idx}",
                w,
                after,
                "unward",
                {"caster": ids["w0"], "shrine_target": ids["sc0"]},
            )
        )
    w_un0, ids_un0 = make_world("uwn0", sanctum_gate="false", sanctum_ward="false")
    train_eps.append(
        _make_synthetic_ep(
            "ep_uw_neg_0",
            w_un0,
            w_un0.copy(),
            "unward",
            {"caster": ids_un0["w0"], "shrine_target": ids_un0["sc0"]},
        )
    )

    learner = ITLOperatorLearner(seed=7, taxonomy=taxonomy)
    learner.fit(train_eps)
    learner.freeze()

    schemas = learner.recovered_operator_schemas()
    assert set(schemas.keys()) == {"bestow", "channel", "unward"}
    assert ["?relic", "charged", "true"] in [list(x) for x in schemas["channel"]["required_attr_guards"]]
    assert ["?origin", "neq", "?dest"] in [list(x) for x in schemas["channel"]["required_attr_guards"]]
    assert ["?relic_sanctum_if_any", "gate_open", "true"] in [
        list(x) for x in schemas["bestow"]["required_facts"]
    ]
    assert ["?shrine_target", "seal_label", "clear"] in [
        list(x) for x in schemas["unward"]["attribute_edits"]
    ]

    # Verify decorative-coincidence immunity: heir.aura was 'silver' in all 4 bestow training demos
    # yet MUST NOT be induced because wizard.aura varies freely across other positive wizard roles!
    assert all(g[1] != "aura" for g in schemas["bestow"]["required_attr_guards"])

    # Evaluate on completely unseen entity names + unseen heir_aura + multi-step sequence + blocked contrast
    w_test, t_ids = make_world(
        "UNSEEN99", sanctum_gate="true", sanctum_ward="true", badge_color="emerald", heir_aura="violet"
    )
    a1 = ActionInvocation("unward", (("caster", t_ids["w0"]), ("shrine_target", t_ids["sc0"])), b"unward", (("action", 0, 6),))
    a2 = ActionInvocation(
        "channel",
        (("caster", t_ids["w0"]), ("dest", t_ids["sc0"]), ("origin", t_ids["al0"]), ("relic", t_ids["r0"])),
        b"channel",
        (("action", 0, 7),),
    )
    a3 = ActionInvocation(
        "bestow",
        (("caster", t_ids["w0"]), ("heir", t_ids["w1"]), ("relic", t_ids["r0"])),
        b"bestow",
        (("action", 0, 6),),
    )
    task = EvaluationTaskPublic(
        task_id="synth_comp_3",
        split_id="synthetic_heldout",
        generator_id="synth_test",
        generator_version="v1",
        seed=999,
        depth=3,
        initial_state=w_test,
        actions=(a1, a2, a3),
        observation_mask=ObservationMask(fully_observed=True),
    )
    pred = learner.predict(task)
    assert pred.abstained is False
    assert pred.predicted_final_state is not None
    expected = w_test.with_changes(
        delete_facts=(
            (t_ids["sc0"], "ward_active", "true"),
            (t_ids["r0"], "anchored_at", t_ids["al0"]),
            (t_ids["r0"], "attuned_to", t_ids["w0"]),
        ),
        add_facts=(
            (t_ids["sc0"], "ward_active", "false"),
            (t_ids["r0"], "anchored_at", t_ids["sc0"]),
            (t_ids["r0"], "attuned_to", t_ids["w1"]),
        ),
        attribute_edits=((t_ids["sc0"], "seal_label", "clear"),),
    )
    assert pred.predicted_final_state == expected

    # Blocked guard contrast: bestow inside closed sanctum MUST predict unchanged state
    w_blk, b_ids = make_world("BLK99", sanctum_gate="false", sanctum_ward="false", rune0_at_sanctum=True)
    act_blk = ActionInvocation(
        "bestow",
        (("caster", b_ids["w0"]), ("heir", b_ids["w1"]), ("relic", b_ids["r0"])),
        b"bestow",
        (("action", 0, 6),),
    )
    t_blk = EvaluationTaskPublic(
        task_id="synth_blk",
        split_id="synthetic_heldout",
        generator_id="synth_test",
        generator_version="v1",
        seed=999,
        depth=1,
        initial_state=w_blk,
        actions=(act_blk,),
        observation_mask=ObservationMask(fully_observed=True),
    )
    pred_blk = learner.predict(t_blk)
    assert pred_blk.abstained is False
    assert pred_blk.predicted_final_state == w_blk


def test_s2_operator_card_serialization_partial_mask_and_unknown_action() -> None:
    train_eps = generate_training_episodes(101, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    learner = ITLOperatorLearner(seed=101)
    learner.fit(train_eps)
    learner.freeze()

    assert len(learner.operators) == 4
    for aid, card in learner.operators.items():
        reloaded = OperatorCard.from_dict(card.to_dict())
        assert reloaded == card
        assert reloaded.fingerprint() == card.fingerprint()
        assert reloaded.to_schema_dict(learner.taxonomy) == card.to_schema_dict(learner.taxonomy)
        assert card.confidence == 1.0
        assert card.support_positive == 4
        assert card.support_negative == 2

    # Partial ObservationMask hiding a required fact MUST abstain honestly rather than hallucinate
    suite = generate_heldout_suite(101, train_eps)
    open_case = next(
        c for c in suite["heldout_unseen_names"] if c.public.actions[0].action_id == "open"
    )
    partial_mask = ObservationMask(
        observed_entities=frozenset(eid for eid, _ in open_case.public.initial_state.entities),
        observed_predicates=frozenset({"is_open", "accessible"}),  # Hides is_locked!
        fully_observed=False,
    )
    masked_task = EvaluationTaskPublic(
        task_id="masked_open_probe",
        split_id="partial_mask_probe",
        generator_id="test",
        generator_version="v1",
        seed=101,
        depth=1,
        initial_state=open_case.public.initial_state,
        actions=open_case.public.actions,
        observation_mask=partial_mask,
    )
    masked_pred = learner.predict(masked_task)
    assert masked_pred.abstained is True
    assert masked_pred.predicted_final_state is None

    # Unknown unseen action MUST abstain honestly
    unknown_act = ActionInvocation(
        "levitate", (("actor", "zara_1"), ("object", "chalice_1")), b"zara_1 levitate chalice_1", (("action", 7, 15),)
    )
    unk_task = EvaluationTaskPublic(
        task_id="unknown_action_probe",
        split_id="ood_probe",
        generator_id="test",
        generator_version="v1",
        seed=101,
        depth=1,
        initial_state=open_case.public.initial_state,
        actions=(unknown_act,),
        observation_mask=ObservationMask(fully_observed=True),
    )
    unk_pred = learner.predict(unk_task)
    assert unk_pred.abstained is True
    assert unk_pred.predicted_final_state is None


def test_s2_freeze_zero_test_time_retrieval_disk_ledger_and_edge_cases(tmp_path: Path) -> None:
    train_eps = generate_training_episodes(202, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    suite = generate_heldout_suite(202, train_eps)

    # Train directly from an on-disk S1 EpisodeLedger
    db_path = tmp_path / "s2_train_ledger.sqlite3"
    with EpisodeLedger(db_path) as ledger:
        for ep in train_eps:
            ledger.ingest_training_episode(ep, use_typed_action=True)
        ledger.freeze()

        learner = ITLOperatorLearner(seed=202)
        learner.fit_from_ledger(ledger, split_id="train")
        learner.freeze()
        ledger.reset_audit_log()

        with pytest.raises(RuntimeError):
            learner.fit(train_eps)
        with pytest.raises(RuntimeError):
            learner.fit_from_ledger(ledger)

        for case in suite["heldout_comp_depth6"][:5]:
            pred = learner.predict(case.public)
            assert pred.abstained is False
            assert pred.retrieval_hits == 0
            assert pred.memory_accesses == 0
            assert pred.predicted_final_state == case.hidden.gold_final_state

        # Zero reads against the frozen on-disk ledger during inference
        assert ledger.audit_summary()["total_accesses"] == 0

    # Empty fit & low-demo (4 demos/action) robustness
    empty_learner = ITLOperatorLearner(seed=202)
    empty_learner.fit([])
    assert empty_learner.bit_budget(0, 0).compressed_model_mdl_bits == 0
    assert empty_learner.predict(suite["heldout_unseen_names"][0].public).abstained is True

    low_demo_eps = generate_training_episodes(202, demos_per_action=4)
    low_learner = ITLOperatorLearner(seed=202)
    low_learner.fit(low_demo_eps)
    assert len(low_learner.operators) == 4


def test_s2_ablation_and_s1_s2_raw_utf8_e2e_on_seed_101() -> None:
    full_res = evaluate_solver_on_seed(ITLOperatorLearner(seed=101), seed=101)
    nc_res = evaluate_solver_on_seed(
        ITLOperatorLearner(seed=101, use_negative_contrasts=False), seed=101
    )
    e2e_res = evaluate_solver_on_seed(
        ITLOperatorLearner(seed=101, raw_utf8_parse_mode=True), seed=101
    )

    assert full_res["unseen_overall"]["learning_exact_accuracy"] >= 0.99
    assert full_res["unseen_overall"]["mean_collateral_rate"] == 0.0
    assert full_res["operator_recovery"]["precision"] >= 0.95
    assert full_res["operator_recovery"]["recall"] == 1.0
    assert full_res["operator_recovery"]["recall"] > nc_res["operator_recovery"]["recall"]
    assert (
        full_res["per_split"]["heldout_guard_contrasts"]["learning_exact_accuracy"]
        > nc_res["per_split"]["heldout_guard_contrasts"]["learning_exact_accuracy"]
    )
    assert e2e_res["unseen_overall"]["learning_exact_accuracy"] >= 0.99
    assert e2e_res["per_split"]["heldout_comp_depth6"]["learning_exact_accuracy"] >= 0.95
    assert e2e_res["unseen_overall"]["mean_collateral_rate"] == 0.0


def test_s2_committed_manifests_and_saved_5seed_gate_pass(tmp_path: Path) -> None:
    ok, mismatches = verify_s2_committed_manifests(
        seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION
    )
    assert ok is True, f"S2 preregistration drift: {mismatches}"

    summary_path = REPORTS_DIR / "s2_summary.json"
    assert summary_path.is_file(), "Missing committed s2_summary.json"
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["s2_gate"]["passed"] is True
    assert len(saved["s2_gate"]["checks"]) >= 19

    # Fresh isolated 2-seed end-to-end execution test
    fresh = run_s2_experiments(
        seeds=(101, 202),
        demos_per_action=DEFAULT_DEMOS_PER_ACTION,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        verify_prior_manifests=False,
    )
    for check_name, ok_val in fresh["s2_gate"]["checks"].items():
        if check_name != "five_distinct_fixed_seeds_present":
            assert ok_val is True, f"Fresh S2 check failed: {check_name}"
