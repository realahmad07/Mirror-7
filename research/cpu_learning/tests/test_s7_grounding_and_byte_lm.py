"""Stage S7 Independent Adversarial Tests: Raw UTF-8 Grounding, Plan Realization & Byte-LM.

STRICT ISOLATION:
- NEVER imports ``hidden_actions`` or ``oracle``.
- Audits AST/lexical isolation, ``comparison_only=True`` transformer isolation, multi-byte
  UTF-8 byte-span exactness, 100% alien non-English construction/negation/conditional/connector
  induction, conditional+negation symmetry, partial ``ObservationMask`` epistemic abstention,
  minimal-contrast ambiguity resolution, separated semantic vs surface realization metrics
  (including multi-byte UTF-8 & occluded-entity hallucination checks), Byte-LM 256-way
  probability normalization & construction-skeleton ablation, and S0–S7 manifests + saved 5-seed
  & isolated 2-seed subset runs.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

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
from research.cpu_learning.evaluation.run_s3_experiments import verify_s3_committed_manifests
from research.cpu_learning.evaluation.run_s4_experiments import verify_s4_committed_manifests
from research.cpu_learning.evaluation.run_s5_experiments import verify_s5_committed_manifests
from research.cpu_learning.evaluation.run_s6_experiments import (
    _identifiers_in_code,
    verify_s6_committed_manifests,
)
from research.cpu_learning.evaluation.run_s7_experiments import (
    _EXTRA_D2_D3_FORBIDDEN_LITERALS,
    _run_single_s7_seed,
    _verify_transformer_analytical_gradients,
    audit_s7_ast_isolation_and_agnosticism,
    build_s7_grounding_training_episodes,
    run_s7_experiments,
    verify_s7_committed_manifests,
)
from research.cpu_learning.frames.state_frame import TypedStateFrame
from research.cpu_learning.grounding import (
    Byte3GramKneserNeyLM,
    ByteLMFrozenError,
    ConstructionLearnerFrozenError,
    GroundedInstructionExecutor,
    GroundedPlanRealizer,
    MirrorCTWConstructionByteLM,
    PlanRealizerFrozenError,
    PPMCOrder5ByteLM,
    RawByteConstructionLearner,
    SmallCPUByteTransformerRef,
    UniformByteFloorLM,
    UnlabelledGroundingEpisode,
    decode_and_tokenize_utf8,
    evaluate_byte_lm_bpb,
    generate_disjoint_byte_lm_corpus,
)
from research.cpu_learning.itl.contrast_harness import (
    ContrastRefinedITLLearner,
    build_probe_action,
)
from research.cpu_learning.itl.operator_card import OperatorCard
from research.cpu_learning.simulator.generators import (
    HELDOUT_POOL,
    TRAIN_POOL,
    _make_base_world,
    generate_training_episodes,
)
from research.cpu_learning.simulator.state_graph import (
    Entity,
    ObservationMask,
    StateGraph,
)


def _induce_s3_operators(seed: int = 101) -> tuple[dict[str, OperatorCard], object]:
    env = make_blackbox_environment()
    s0_eps = generate_training_episodes(seed=seed, demos_per_action=6)
    s3 = ContrastRefinedITLLearner(seed=seed, env=env)
    s3.fit(s0_eps)
    s3.freeze()
    ops = {aid: s3.version_store.get_active(aid).card for aid in ("move", "give", "open", "lock")}
    return ops, env


def test_s7_ast_isolation_zero_domain_literals_and_transformer_isolation() -> None:
    """Verify strict AST import isolation, zero S0/D2/D3 domain literals, and comparison-only transformer isolation."""
    audit = audit_s7_ast_isolation_and_agnosticism(PACKAGE_ROOT)
    assert audit["passed"] is True, f"S7 AST audit failed: {audit}"
    assert audit["import_violations"] == []
    assert audit["hardcode_violations"] == []
    assert audit["transformer_leak_violations"] == []
    assert audit["scanned_grounding_modules"] >= 5

    assert SmallCPUByteTransformerRef.comparison_only is True
    assert SmallCPUByteTransformerRef.uses_backprop is True
    assert MirrorCTWConstructionByteLM.comparison_only is False
    assert MirrorCTWConstructionByteLM.uses_backprop is False

    forbidden_lits = set(_FORBIDDEN_ITL_HARDCODED_LITERALS) | set(_EXTRA_D2_D3_FORBIDDEN_LITERALS)
    for fname in ("byte_aligner.py", "construction_learner.py", "plan_realizer.py"):
        src = (PACKAGE_ROOT / "grounding" / fname).read_text(encoding="utf-8")
        consts = _string_constants_in_code(src)
        idents = _identifiers_in_code(src)
        assert (consts & forbidden_lits) == set(), f"Domain literals in {fname}: {consts & forbidden_lits}"
        assert "SmallCPUByteTransformerRef" not in idents
        assert "small_cpu_byte_transformer_ref" not in consts


def test_s7_multibyte_utf8_byte_spans_and_alien_non_english_syntax_induction() -> None:
    """Verify multi-byte UTF-8 byte offsets and 100% alien non-English syntax induction on a synthetic world."""
    ok_bad, toks_bad = decode_and_tokenize_utf8(b"\xff\xfeinvalid")
    assert ok_bad is False and toks_bad == ()

    raw_sample = "  élan_1 → ключ_2 ; маг_3! ".encode("utf-8")
    ok_u8, toks_u8 = decode_and_tokenize_utf8(raw_sample)
    assert ok_u8 is True
    for tok in toks_u8:
        assert raw_sample[tok.byte_start : tok.byte_end].decode("utf-8") == tok.text

    # Pure synthetic non-S0 world with multi-byte Cyrillic/accented entity IDs & alien operators
    ents = {
        "маг_1": Entity("маг_1", "sage", ()),
        "кристалл_2": Entity("кристалл_2", "shard_item", ()),
        "алтарь_3": Entity("алтарь_3", "altar_node", ()),
        "свод_4": Entity("свод_4", "altar_node", ()),
        "маяк_5": Entity("маяк_5", "pylon", ()),
        "маяк_6": Entity("маяк_6", "pylon", ()),
        "маяк_7": Entity("маяк_7", "pylon", ()),
    }
    w_base = StateGraph.create(
        ents,
        {
            ("кристалл_2", "docked_at", "алтарь_3"),
            ("маяк_5", "lit_state", "false"),
            ("маяк_5", "sealed_ward", "false"),
            ("маяк_6", "lit_state", "true"),
            ("маяк_6", "sealed_ward", "false"),
            ("маяк_7", "lit_state", "false"),
            ("маяк_7", "sealed_ward", "true"),
        },
    )
    w_op_sealed = w_base.with_changes(
        delete_facts=(("маяк_6", "sealed_ward", "false"),),
        add_facts=(("маяк_6", "sealed_ward", "true"),),
    )

    from research.cpu_learning.itl.operator_card import RoleInequalityGuard

    ignite_card = OperatorCard(
        card_id="syn_ignite_v1",
        action_id="ignite_pylon",
        role_variables=("actor", "pylon"),
        required_types=(("actor", frozenset({"sage"})), ("pylon", frozenset({"pylon"}))),
        required_facts=(("?pylon", "lit_state", "false"), ("?pylon", "sealed_ward", "false")),
        conditional_fact_guards=(),
        indirect_fact_guards=(),
        attribute_guards=(),
        inequality_guards=(),
        add_facts=(("?pylon", "lit_state", "true"),),
        delete_facts=(("?pylon", "lit_state", "false"),),
        attribute_edits=(),
        support_positive=4,
    )
    ferry_card = OperatorCard(
        card_id="syn_ferry_v1",
        action_id="ferry_shard",
        role_variables=("actor", "item", "from_node", "to_node"),
        required_types=(
            ("actor", frozenset({"sage"})),
            ("item", frozenset({"shard_item"})),
            ("from_node", frozenset({"altar_node"})),
            ("to_node", frozenset({"altar_node"})),
        ),
        required_facts=(("?item", "docked_at", "?from_node"),),
        conditional_fact_guards=(),
        indirect_fact_guards=(),
        attribute_guards=(),
        inequality_guards=(RoleInequalityGuard("from_node", "to_node"),),
        add_facts=(("?item", "docked_at", "?to_node"),),
        delete_facts=(("?item", "docked_at", "?from_node"),),
        attribute_edits=(),
        support_positive=4,
    )
    syn_ops = {"ignite_pylon": ignite_card, "ferry_shard": ferry_card}

    af_ignite = ignite_card.apply(w_base, {"actor": "маг_1", "pylon": "маяк_5"}).after_state
    af_ferry = ferry_card.apply(
        w_base,
        {"actor": "маг_1", "item": "кристалл_2", "from_node": "алтарь_3", "to_node": "свод_4"},
    ).after_state
    af_ferry_s = ferry_card.apply(
        w_op_sealed,
        {"actor": "маг_1", "item": "кристалл_2", "from_node": "алтарь_3", "to_node": "свод_4"},
    ).after_state
    assert af_ignite is not None and af_ferry is not None and af_ferry_s is not None

    learner = RawByteConstructionLearner(syn_ops)
    alien_eps = [
        # Alien OSV / postpositional constructions
        UnlabelledGroundingEpisode(
            "al_1", "маяк_5 маг_1 lumos".encode("utf-8"), w_base, af_ignite
        ),
        UnlabelledGroundingEpisode(
            "al_2", "кристалл_2 алтарь_3 » свод_4 маг_1 xylar".encode("utf-8"), w_base, af_ferry
        ),
        # Alien negation prefix "zot!"
        UnlabelledGroundingEpisode(
            "al_3", "zot! маяк_5 маг_1 lumos".encode("utf-8"), w_base, w_base
        ),
        UnlabelledGroundingEpisode(
            "al_4", "zot! кристалл_2 алтарь_3 » свод_4 маг_1 xylar".encode("utf-8"), w_base, w_base
        ),
        # Alien conditional guard "krel <ent> vax," -> lit_state == true
        UnlabelledGroundingEpisode(
            "al_5", "krel маяк_6 vax, кристалл_2 алтарь_3 » свод_4 маг_1 xylar".encode("utf-8"), w_base, af_ferry
        ),
        UnlabelledGroundingEpisode(
            "al_6", "krel маяк_6 vax, кристалл_2 алтарь_3 » свод_4 маг_1 xylar".encode("utf-8"), w_op_sealed, af_ferry_s
        ),
        UnlabelledGroundingEpisode(
            "al_7", "krel маяк_5 vax, кристалл_2 алтарь_3 » свод_4 маг_1 xylar".encode("utf-8"), w_base, w_base
        ),
        # Regression for Defect #3: 2-clause connector "koda" where clause 2 is blocked (sealed_ward=true)!
        UnlabelledGroundingEpisode(
            "al_8",
            "маяк_5 маг_1 lumos koda маяк_7 маг_1 lumos".encode("utf-8"),
            w_base,
            af_ignite,
        ),
    ]
    stats = learner.fit_episodes(alien_eps)
    learner.freeze()
    assert stats["promoted_constructions"] == 2
    assert any(r.prefix_tokens == ("zot",) for r in learner.negation_rules)
    assert any(
        r.intro_tokens == ("krel",) and r.post_tokens == ("vax",) and r.predicate == "lit_state"
        for r in learner.condition_rules
    )
    assert ("koda",) in learner.clause_connectors

    # Execute chained alien instruction on unseen multi-byte entities!
    unseen_ents = {
        "élan_9": Entity("élan_9", "sage", ()),
        "рубин_8": Entity("рубин_8", "shard_item", ()),
        "база_1": Entity("база_1", "altar_node", ()),
        "база_2": Entity("база_2", "altar_node", ()),
        "пилон_3": Entity("пилон_3", "pylon", ()),
    }
    w_unseen = StateGraph.create(
        unseen_ents,
        {
            ("рубин_8", "docked_at", "база_1"),
            ("пилон_3", "lit_state", "false"),
            ("пилон_3", "sealed_ward", "false"),
        },
    )
    # Step 1 lights пилон_3; Step 2 checks `krel пилон_3 vax` (now true!) and ferries рубин_8; Step 3 is negated!
    chained_cmd = (
        "пилон_3 élan_9 lumos koda "
        "krel пилон_3 vax, рубин_8 база_1 » база_2 élan_9 xylar koda "
        "zot! рубин_8 база_2 » база_1 élan_9 xylar."
    ).encode("utf-8")
    executor = GroundedInstructionExecutor(learner)
    res = executor.execute_raw_utf8(chained_cmd, w_unseen)
    assert res.abstained is False
    assert [t.outcome for t in res.clause_traces] == ["applied", "applied", "negated_noop"]
    assert res.final_state is not None
    assert ("пилон_3", "lit_state", "true") in res.final_state.facts
    assert ("рубин_8", "docked_at", "база_2") in res.final_state.facts
    for tr in res.clause_traces:
        for rname, (bs, be) in tr.byte_spans.items():
            assert chained_cmd[bs:be].decode("utf-8") == tr.roles[rname]


def test_s7_conditional_negation_symmetry_epistemic_mask_and_ood_abstention() -> None:
    """Verify Defect #1 regression (conditional+negation symmetry), epistemic mask abstention, and freeze guard."""
    ops, env = _induce_s3_operators(101)
    g_eps = build_s7_grounding_training_episodes(101, env)
    learner = RawByteConstructionLearner(ops)
    learner.fit_episodes(g_eps)
    learner.freeze()

    with pytest.raises(ConstructionLearnerFrozenError):
        learner.fit_episodes(g_eps[:1])
    with pytest.raises(ConstructionLearnerFrozenError):
        learner.observe_utterance_in_scene(b"agent_1 opens box_1", g_eps[0].before_state)

    rng = random.Random(101)
    w, ids = _make_base_world(
        HELDOUT_POOL,
        rng,
        container_states=((False, False, True), (True, False, True), (False, True, True)),
    )
    ag1 = ids["agents"][0]
    ob1 = ids["objects"][0]
    ct_cl, ct_op = ids["containers"][0], ids["containers"][1]
    executor = GroundedInstructionExecutor(learner)

    # Defect #1 regression: BOTH true-condition and false-condition negated bodies must succeed cleanly!
    cmd_true_neg = f"if {ct_op} is open, do not {ag1} opens {ct_cl}.".encode("utf-8")
    cmd_false_neg = f"if {ct_cl} is open, do not {ag1} opens {ct_cl}.".encode("utf-8")
    r_true = executor.execute_raw_utf8(cmd_true_neg, w)
    r_false = executor.execute_raw_utf8(cmd_false_neg, w)

    assert not r_true.abstained and r_true.final_state == w
    assert [t.outcome for t in r_true.clause_traces] == ["negated_noop"]
    assert not r_false.abstained and r_false.final_state == w
    assert [t.outcome for t in r_false.clause_traces] == ["condition_false_noop"]
    for raw_b, r_item in ((cmd_true_neg, r_true), (cmd_false_neg, r_false)):
        for rname, (bs, be) in r_item.clause_traces[0].byte_spans.items():
            assert raw_b[bs:be].decode("utf-8") == r_item.clause_traces[0].roles[rname]

    # OOD body inside a false condition must STILL abstain honestly!
    r_ood_false = executor.execute_raw_utf8(
        f"if {ct_cl} is open, {ag1} vaporizes {ob1}.".encode("utf-8"), w
    )
    assert r_ood_false.abstained is True and r_ood_false.final_state is None

    # Partial ObservationMask occluding `is_open` -> epistemic UNOBSERVED abstention!
    mask_hide_open = ObservationMask(
        observed_entities=frozenset(eid for eid, _ in w.entities),
        observed_predicates=frozenset({"located_on", "owned_by", "is_locked", "accessible"}),
        fully_observed=False,
    )
    r_unobs = executor.execute_raw_utf8(cmd_true_neg, w, mask=mask_hide_open)
    assert r_unobs.abstained is True and r_unobs.abstention_reason == "unobserved_condition_fact_under_mask"


def test_s7_ambiguity_preservation_and_minimal_contrast_disambiguation() -> None:
    """Verify uncontrasted utterances preserve competing shadow parses and 2 minimal contrasts resolve them."""
    ops, env = _induce_s3_operators(101)
    learner = RawByteConstructionLearner(ops)
    executor = GroundedInstructionExecutor(learner)

    rng = random.Random(202)
    w_tr1, ids_tr1 = _make_base_world(TRAIN_POOL, rng)
    w_tr2, ids_tr2 = _make_base_world(TRAIN_POOL, rng)
    w_ho, ids_ho = _make_base_world(HELDOUT_POOL, rng)

    # Preposition-less reversed double-location: "<actor> blargs <dst> <src> <obj>"
    u_tr1 = f"{ids_tr1['agents'][0]} blargs {ids_tr1['surfaces'][1]} {ids_tr1['surfaces'][0]} {ids_tr1['objects'][0]}".encode("utf-8")
    shadows = learner.observe_utterance_in_scene(u_tr1, w_tr1)
    assert len(shadows) >= 2 and all(c.card_state == "shadow" for c in shadows)

    u_ho = f"{ids_ho['agents'][0]} blargs {ids_ho['surfaces'][1]} {ids_ho['surfaces'][0]} {ids_ho['objects'][0]}".encode("utf-8")
    parse_pre = learner.parse_atomic_clause(u_ho, TypedStateFrame.from_state_graph(w_ho))
    exec_pre = executor.execute_raw_utf8(u_ho, w_ho)
    assert parse_pre.is_ambiguous is True and len(parse_pre.candidate_frames) >= 2
    assert exec_pre.abstained is True and exec_pre.is_ambiguous is True

    # Supply 2 minimal contrast episodes demonstrating dst-first move semantics
    after_tr1, _ = env.step(
        w_tr1,
        build_probe_action(
            "move",
            {
                "actor": ids_tr1["agents"][0],
                "object": ids_tr1["objects"][0],
                "source": ids_tr1["surfaces"][0],
                "destination": ids_tr1["surfaces"][1],
            },
        ),
    )
    u_tr2 = f"{ids_tr2['agents'][0]} blargs {ids_tr2['surfaces'][1]} {ids_tr2['surfaces'][0]} {ids_tr2['objects'][0]}".encode("utf-8")
    after_tr2, _ = env.step(
        w_tr2,
        build_probe_action(
            "move",
            {
                "actor": ids_tr2["agents"][0],
                "object": ids_tr2["objects"][0],
                "source": ids_tr2["surfaces"][0],
                "destination": ids_tr2["surfaces"][1],
            },
        ),
    )
    learner.fit_episodes(
        [
            UnlabelledGroundingEpisode("c1", u_tr1, w_tr1, after_tr1),
            UnlabelledGroundingEpisode("c2", u_tr2, w_tr2, after_tr2),
        ]
    )
    learner.freeze()

    assert len(learner.promoted_constructions) == 1
    assert learner.promoted_constructions[0].version == 2
    assert any(c.card_state == "retired" for c in learner._history)

    gold_ho, _ = env.step(
        w_ho,
        build_probe_action(
            "move",
            {
                "actor": ids_ho["agents"][0],
                "object": ids_ho["objects"][0],
                "source": ids_ho["surfaces"][0],
                "destination": ids_ho["surfaces"][1],
            },
        ),
    )
    exec_post = executor.execute_raw_utf8(u_ho, w_ho)
    assert exec_post.abstained is False and exec_post.is_ambiguous is False
    assert exec_post.final_state == gold_ho


def test_s7_plan_realizer_dual_metrics_utf8_hallucination_mask_and_freeze() -> None:
    """Verify Defect #2 regressions (multi-byte UTF-8 & occluded-entity hallucination, template refresh, freeze) and dual-metric independence."""
    ops, env = _induce_s3_operators(101)
    learner = RawByteConstructionLearner(ops)
    # Instantiate realizer BEFORE learner.fit_episodes to test automatic template refresh!
    realizer = GroundedPlanRealizer(learner)

    g_eps = build_s7_grounding_training_episodes(101, env)
    learner.fit_episodes(g_eps)
    learner.freeze()

    realizer.fit_prose_corpus(
        [
            "In this step, the action succeeded and updated 2 verified world relations.",
            "In this step, the instruction was intentionally skipped due to negation, preserving all existing relations.",
            "In this step, the attempt was blocked because a required guard failed, leaving the world unchanged.",
        ]
    )
    realizer.freeze()
    assert realizer.is_frozen is True
    with pytest.raises(PlanRealizerFrozenError):
        realizer.fit_prose_corpus(["Extra prose."])

    rng = random.Random(303)
    w, ids = _make_base_world(
        HELDOUT_POOL,
        rng,
        container_states=((False, False, True), (True, False, True), (False, True, True)),
    )
    ag1 = ids["agents"][0]
    ct_lck = ids["containers"][2]
    executor = GroundedInstructionExecutor(learner)

    # Blocked execution trace (attempting to open a locked container)
    raw_blk = f"{ag1} opens {ct_lck}.".encode("utf-8")
    ex_blk = executor.execute_raw_utf8(raw_blk, w)
    assert not ex_blk.abstained and ex_blk.clause_traces[0].outcome == "blocked"

    ev_mirror = realizer.evaluate_realization("mirror", realizer.realize(ex_blk), ex_blk, w)
    ev_dump = realizer.evaluate_realization(
        "dump", realizer.realize_telegraphic_tuple_dump(ex_blk), ex_blk, w
    )
    ev_assume = realizer.evaluate_realization(
        "assumer", realizer.realize_ungated_fluent_assumer(raw_blk, w), ex_blk, w
    )

    assert ev_mirror.semantic_correctness == 1.0 and ev_mirror.hallucination_rate == 0.0
    assert ev_dump.semantic_correctness == 1.0 and ev_dump.hallucination_rate == 0.0
    assert ev_mirror.surface_fluency_score > ev_dump.surface_fluency_score + 0.40
    assert ev_assume.surface_fluency_score > 0.75
    assert ev_assume.semantic_correctness == 0.0 and ev_assume.hallucination_rate > 0.0

    # Defect #2a regression: hallucinated multi-byte UTF-8 entity IDs (élan_99, ключ_42) MUST be caught!
    ev_utf8_halluc = realizer.evaluate_realization(
        "bad_utf8",
        f"In this step, the attempt by {ag1} and élan_99 on {ct_lck} and ключ_42 was blocked.",
        ex_blk,
        w,
    )
    assert ev_utf8_halluc.semantic_correctness == 0.0
    assert ev_utf8_halluc.hallucination_rate > 0.0

    # Defect #2b regression: mentioning an occluded entity under partial ObservationMask MUST be caught!
    occluded_id = ids["objects"][0]
    mask_occ = ObservationMask(
        observed_entities=frozenset(eid for eid, _ in w.entities if eid != occluded_id),
        observed_predicates=frozenset({"located_on", "owned_by", "is_open", "is_locked", "accessible"}),
        fully_observed=False,
    )
    ev_occ = realizer.evaluate_realization(
        "occ_leak",
        f"In this step, the attempt by {ag1} on {ct_lck} and {occluded_id} was blocked.",
        ex_blk,
        w,
        mask=mask_occ,
    )
    assert ev_occ.semantic_correctness == 0.0 and ev_occ.hallucination_rate > 0.0


def test_s7_byte_lm_zero_leakage_normalization_0_to_255_grad_check_and_skeleton_gain() -> None:
    """Verify corpus disjointness, 256-way normalization across 0x00..0xFF, transformer gradients, and skeleton gain."""
    corpus = generate_disjoint_byte_lm_corpus(101)
    assert corpus.disjoint_entities_verified is True
    assert (set(corpus.train_entity_ids) & set(corpus.heldout_entity_ids)) == set()

    # Freeze guards
    m_unfrozen = UniformByteFloorLM()
    m_unfrozen.fit(b"abc")
    with pytest.raises(ByteLMFrozenError):
        evaluate_byte_lm_bpb(m_unfrozen, b"abc")
    m_unfrozen.freeze()
    with pytest.raises(ByteLMFrozenError):
        m_unfrozen.fit(b"abc")

    # Adversarial binary stream containing all 256 byte values 0x00..0xFF
    train_mini = "élan_10 relocates token_12 from rack_11 → tray_14.\n".encode("utf-8") * 16
    eval_all256 = bytes(range(256)) + "noël_77 repositions wafer_88.\n".encode("utf-8")
    for model in (
        UniformByteFloorLM(),
        Byte3GramKneserNeyLM(),
        PPMCOrder5ByteLM(),
        MirrorCTWConstructionByteLM(),
        SmallCPUByteTransformerRef(seed=101, d_model=4, ctx_len=6, train_stride=16, epochs=1),
    ):
        model.fit(train_mini)
        model.freeze()
        ev = evaluate_byte_lm_bpb(model, eval_all256, norm_check_positions=290)
        assert ev["normalization_verified"] is True
        assert ev["max_sum_prob_error"] <= 1e-6
        assert ev["max_consistency_error"] <= 1e-9

    # Analytical vs finite-difference gradient check on SmallCPUByteTransformerRef
    assert _verify_transformer_analytical_gradients(101) < 1e-5

    # Verify construction-skeleton context tree genuinely lowers held-out bpb vs raw order-10 KT alone!
    m_full = MirrorCTWConstructionByteLM()
    m_full.fit(corpus.train_bytes)
    m_full.freeze()
    bpb_full = evaluate_byte_lm_bpb(m_full, corpus.heldout_bytes)["bits_per_byte"]

    m_raw_only = MirrorCTWConstructionByteLM()
    m_raw_only.fit(corpus.train_bytes)
    m_raw_only.w_raw_deeper = 1.0
    m_raw_only.w_tie = 1.0
    m_raw_only.w_skel_deeper = 1.0
    m_raw_only.freeze()
    bpb_raw_only = evaluate_byte_lm_bpb(m_raw_only, corpus.heldout_bytes)["bits_per_byte"]

    m_ppm5 = PPMCOrder5ByteLM()
    m_ppm5.fit(corpus.train_bytes)
    m_ppm5.freeze()
    bpb_ppm5 = evaluate_byte_lm_bpb(m_ppm5, corpus.heldout_bytes)["bits_per_byte"]

    assert bpb_full < bpb_raw_only - 0.15
    assert bpb_full < bpb_ppm5


def test_s7_single_seed_end_to_end_depth6_and_all_splits() -> None:
    """Verify full single-seed Stage S7 pipeline on seed 101 across all 8 splits including Depth-6 chains."""
    rec = _run_single_s7_seed(101)
    assert rec["pool_disjoint_verified"] is True
    assert rec["overall_grounding_exact_acc"] == 1.0
    assert rec["overall_collateral_rate"] == 0.0
    for sp_name, sm in rec["grounding_splits"].items():
        assert sm["mirror_exact_acc"] == 1.0, f"Failed split {sp_name}: {sm}"
        assert sm["mirror_intermediate_acc"] == 1.0
        assert sm["mirror_byte_span_exact_rate"] == 1.0
        assert sm["mirror_collateral_rate"] == 0.0
    assert rec["overall_grounding_exact_acc"] > rec["overall_lexical_baseline_acc"] + 0.50


def test_s7_committed_manifests_saved_5seed_gate_and_fresh_isolated_subset_run(tmp_path: Path) -> None:
    """Verify S0..S7 committed manifests, saved 5-seed S7 summary gate, and Defect #4 isolated 2-seed subset run."""
    ok0, e0 = verify_committed_manifests(data_dir=DATA_DIR)
    ok1, e1 = verify_s1_committed_manifests(data_dir=DATA_DIR)
    ok2, e2 = verify_s2_committed_manifests(data_dir=DATA_DIR)
    ok3, e3 = verify_s3_committed_manifests(data_dir=DATA_DIR)
    ok4, e4 = verify_s4_committed_manifests(data_dir=DATA_DIR)
    ok5, e5 = verify_s5_committed_manifests(data_dir=DATA_DIR)
    ok6, e6 = verify_s6_committed_manifests(data_dir=DATA_DIR)
    ok7, e7 = verify_s7_committed_manifests(data_dir=DATA_DIR)
    assert ok0 and ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7, (
        f"Manifest mismatch: {e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7}"
    )

    saved_summary_path = REPORTS_DIR / "s7_summary.json"
    saved_bytes_before = saved_summary_path.read_bytes()
    saved_summary = json.loads(saved_bytes_before.decode("utf-8"))

    assert saved_summary["stage"] == "S7"
    assert saved_summary["seeds"] == [101, 202, 303, 404, 505]
    assert saved_summary["s7_gate"]["passed"] is True
    assert all(saved_summary["s7_gate"]["checks"].values())
    assert len(saved_summary["s7_gate"]["checks"]) == 24

    bb = saved_summary["bit_budget_mean"]
    assert bb["compressed_induced_archive_bits"] < bb["compressed_training_corpus_bits"] < bb["raw_training_bits"]

    # Defect #4 regression: isolated 2-seed subset run in tmp_path with verify_prior_manifests=True
    tmp_data = tmp_path / "data"
    tmp_reports = tmp_path / "reports"
    fresh = run_s7_experiments(
        seeds=[101, 202],
        data_dir=tmp_data,
        reports_dir=tmp_reports,
        verify_prior_manifests=True,
    )
    assert fresh["s7_gate"]["checks"]["prior_s0_through_s6_manifests_intact"] is True
    for k, ok in fresh["s7_gate"]["checks"].items():
        # Student-t df=1 (t=12.706) widens transformer/3gram CIs on n=2 seeds; check max/min directly below
        if k in {
            "cpu_budget_within_30s_per_seed",
            "byte_lm_mirror_strictly_beats_3gram_kneser_ney_ci95",
            "byte_lm_mirror_strictly_beats_small_cpu_transformer_ci95",
        }:
            continue
        assert ok is True, f"Fresh isolated 2-seed S7 gate failed check: {k}"

    blm_f = fresh["byte_lm_benchmark"]["models"]
    assert (
        blm_f["mirror_ctw_construction_byte_lm"]["bits_per_byte"]["max"]
        < blm_f["byte_3gram_kneser_ney"]["bits_per_byte"]["min"]
    )
    assert (
        blm_f["mirror_ctw_construction_byte_lm"]["bits_per_byte"]["max"]
        < blm_f["small_cpu_byte_transformer_ref"]["bits_per_byte"]["min"]
    )

    assert (tmp_data / "s7_preregistration.json").exists()
    assert (tmp_reports / "s7_summary.json").exists()
    assert (tmp_reports / "s7_research_report.md").exists()
    assert (tmp_reports / "raw_runs" / "s7_splits.csv").exists()
    assert (tmp_reports / "raw_runs" / "s7_seed_101.json").exists()
    assert (tmp_reports / "raw_runs" / "s7_seed_202.json").exists()
    assert saved_summary_path.read_bytes() == saved_bytes_before
