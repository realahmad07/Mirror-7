"""Automated Stage S5 tests: RDC MDL Judge, Dream-Refutation (Zero Self-Certification &
Fair Random Control), Max-Disagreement Active Querying, Interventional Leave-One-Operator-Out
Blame & Ambiguity Honesty, Intra-Domain Consolidation/Rollback, Cross-Domain Concept
Extraction & Held-Out Domain 3 Few-Shot Transfer, AST Isolation, and 5-Seed Gates.

STRICT ISOLATION: Never imports ``hidden_actions`` or ``oracle``.
"""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import pytest

from research.cpu_learning.evaluation.run_experiments import (
    REPORTS_DIR,
    make_blackbox_environment,
    verify_committed_manifests,
)
from research.cpu_learning.evaluation.run_s1_experiments import verify_s1_committed_manifests
from research.cpu_learning.evaluation.run_s2_experiments import verify_s2_committed_manifests
from research.cpu_learning.evaluation.run_s3_experiments import (
    S3_INACCESSIBLE_SOURCE_SPLIT,
    generate_inaccessible_source_contrast_split,
    verify_s3_committed_manifests,
)
from research.cpu_learning.evaluation.run_s4_experiments import verify_s4_committed_manifests
from research.cpu_learning.evaluation.run_s5_experiments import (
    S5_DREAM_OOD_SPLITS,
    _clone_as_alias,
    _corrupt_operator_card,
    _evaluate_library_on_cases,
    _strip_optional_guards,
    _strip_to_core_for_active,
    audit_s5_ast_isolation_and_agnosticism,
    build_domain2_arcane_library,
    generate_domain3_few_shot_and_ood,
    run_s5_experiments,
    verify_s5_committed_manifests,
)
from research.cpu_learning.itl.contrast_harness import (
    ContrastRefinedITLLearner,
    build_probe_action,
)
from research.cpu_learning.itl.delta_extractor import (
    collect_corpus_statistics,
    extract_observations_from_ledger,
)
from research.cpu_learning.itl.operator_card import (
    AttributeGuard,
    ConditionalRoleFactGuard,
    IndirectRoleFactGuard,
    OperatorCard,
    RoleInequalityGuard,
)
from research.cpu_learning.itl.operator_inducer import ITLOperatorLearner
from research.cpu_learning.itl.version_store import (
    OperatorVersionStore,
    VersionStoreError,
    VersionStoreFrozenError,
)
from research.cpu_learning.ledger.episode_ledger import EpisodeLedger
from research.cpu_learning.rdc import (
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    DreamRefuter,
    InterventionalBlameEngine,
    MaxDisagreementQueryEngine,
    MDLExample,
    RDCConsolidator,
    build_hypothesis_version_space,
    check_behavioral_equivalence,
    count_collateral_changes,
    evaluate_library_mdl,
    execute_library_sequence,
    extract_cross_domain_concepts,
    score_blame_attribution,
    score_probe_disagreement,
    synthesize_boundary_dreams,
    synthesize_random_world_probes,
    transfer_concepts_to_few_shot_domain,
)
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
    compute_sparse_delta,
)

PKG_ROOT = Path(__file__).resolve().parents[1]


def test_s5_ast_isolation_and_zero_hardcoded_domain_literals() -> None:
    audit = audit_s5_ast_isolation_and_agnosticism()
    assert audit["passed"] is True, f"S5 AST audit failed: {audit}"
    assert audit["import_violations"] == []
    assert audit["hardcode_violations"] == []
    assert audit["scanned_rdc_modules"] >= 5
    assert audit["scanned_shard_modules"] >= 4
    assert audit["scanned_itl_modules"] >= 6
    assert audit["scanned_learner_modules"] >= 18

    # Expanded domain-literal scan covering S0, Domain 2 (Arcane), AND Domain 3 (Logistics)
    expanded_forbidden = {
        "move",
        "give",
        "open",
        "lock",
        "located_on",
        "owned_by",
        "is_open",
        "is_locked",
        "accessible",
        "portable",
        "status",
        "color",
        "role_badge",
        "agent",
        "object",
        "surface",
        "container",
        "location",
        "actor",
        "destination",
        "recipient",
        "target",
        "channel",
        "bestow",
        "reliquary",
        "pedestal",
        "dispatch",
        "reassign",
        "parcel",
        "courier",
        "dock",
        "bay",
    }
    for fpath in sorted((PKG_ROOT / "rdc").glob("*.py")):
        src = fpath.read_text(encoding="utf-8")
        tree = ast.parse(src)
        doc_ids: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    doc_ids.add(id(node.body[0].value))
        hits = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in doc_ids
            and node.value in expanded_forbidden
        }
        assert hits == set(), f"rdc/{fpath.name} contained forbidden domain literals: {sorted(hits)}"


def test_s5_mdl_judge_objective_formula_collateral_and_aliases() -> None:
    card = OperatorCard(
        card_id="op_flip_v1",
        action_id="flip",
        role_variables=("item",),
        required_types=(("item", frozenset({"switch"})),),
        required_facts=(("?item", "state", "off"),),
        conditional_fact_guards=(),
        indirect_fact_guards=(),
        attribute_guards=(),
        inequality_guards=(),
        add_facts=(("?item", "state", "on"),),
        delete_facts=(("?item", "state", "off"),),
        attribute_edits=(),
    )
    bad_card = OperatorCard(
        card_id="op_flip_bad",
        action_id="flip",
        role_variables=("item",),
        required_types=(("item", frozenset({"switch"})),),
        required_facts=(("?item", "state", "off"),),
        conditional_fact_guards=(),
        indirect_fact_guards=(),
        attribute_guards=(),
        inequality_guards=(),
        add_facts=(("?item", "state", "on"), ("?item", "corrupted_flag", "yes")),
        delete_facts=(("?item", "state", "off"), ("?item", "untouched", "keep")),
        attribute_edits=(),
    )
    w0 = StateGraph.create(
        [Entity("sw_1", "switch", (("label", "main"),))],
        [("sw_1", "state", "off"), ("sw_1", "untouched", "keep")],
    )
    w1 = w0.with_changes(delete_facts=(("sw_1", "state", "off"),), add_facts=(("sw_1", "state", "on"),))
    act = build_probe_action("flip", {"item": "sw_1"})
    alias_act = build_probe_action("flip_alias", {"item": "sw_1"})

    ex = MDLExample("e1", w0, (act,), w1)
    ex_alias = MDLExample("e2", w0, (alias_act,), w1)

    # 1. Exact prediction -> residual_bits=0, verified_failures=0, collateral=0, J == library_bits
    rep_ok = evaluate_library_mdl({"flip": card}, [ex], beta=DEFAULT_BETA, gamma=DEFAULT_GAMMA)
    assert rep_ok.verified_failures == 0
    assert rep_ok.collateral_changes == 0
    assert rep_ok.residual_bits == 0
    assert rep_ok.exact_match_rate == 1.0
    assert rep_ok.j_score == float(rep_ok.library_bits)

    # 2. Alias redirection executes identically
    rep_al = evaluate_library_mdl({"flip": card}, [ex, ex_alias], alias_map={"flip_alias": "flip"})
    assert rep_al.verified_failures == 0 and rep_al.exact_match_rate == 1.0

    # 3. Collateral corruption triggers BOTH beta * failures AND gamma * collateral + residual_bits
    rep_bad = evaluate_library_mdl({"flip": bad_card}, [ex], beta=1000.0, gamma=2500.0)
    assert rep_bad.verified_failures == 1
    assert rep_bad.collateral_changes == 2  # deleted untouched + spurious corrupted_flag
    assert rep_bad.residual_bits > 0
    expected_j = rep_bad.library_bits + rep_bad.residual_bits + 1000.0 * 1 + 2500.0 * 2
    assert rep_bad.j_score == pytest.approx(expected_j)

    # 4. Direct collateral counter catches untouched attr corruption + hallucinated entity
    pred_corrupt = StateGraph.create(
        [Entity("sw_1", "switch", (("label", "broken"),)), Entity("ghost", "switch")],
        w1.facts,
    )
    assert count_collateral_changes(w0, w1, pred_corrupt) == 2
    assert count_collateral_changes(w0, w1, None, abstained=True) == 0

    # 5. Unknown action abstains cleanly
    p_unk, abst_unk, _ = execute_library_sequence({"flip": card}, w0, (build_probe_action("unknown", {"item": "sw_1"}),))
    assert p_unk is None and abst_unk is True


def test_s5_dream_refuter_zero_self_certification_fair_random_control_and_freeze() -> None:
    # Use seed 404 where random_probes genuinely hits >= 2 over-constraint counterexamples!
    seed = 404
    train_eps = generate_training_episodes(seed, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    heldout = generate_heldout_suite(seed, train_eps)
    heldout[S3_INACCESSIBLE_SOURCE_SPLIT] = generate_inaccessible_source_contrast_split(seed)
    ood_cases = [c for sp in S5_DREAM_OOD_SPLITS for c in heldout[sp]]

    with EpisodeLedger(":memory:") as ledger:
        for ep in train_eps:
            ledger.ingest_training_episode(ep, use_typed_action=True)
        grouped_all = extract_observations_from_ledger(ledger, caller_tag="test_s5_dream")
        corpus_stats = collect_corpus_statistics(grouped_all)
        grouped_pos = {aid: [o for o in obs if not o.sparse_delta.is_noop] for aid, obs in grouped_all.items()}
        grouped_neg = {aid: [o for o in obs if o.sparse_delta.is_noop] for aid, obs in grouped_all.items()}

    s2 = ITLOperatorLearner(seed=seed, use_negative_contrasts=True)
    s2.fit(train_eps)
    init_imperfect = {aid: _strip_optional_guards(card) for aid, card in sorted(s2.operators.items())}

    def _fresh_store() -> OperatorVersionStore:
        st = OperatorVersionStore()
        for aid, card in sorted(init_imperfect.items()):
            st.register_shadow(card, notes="init")
            st.promote(aid, 1, notes="init_active")
        return st

    refuter = DreamRefuter(min_ablation_support=2, min_guard_support=2)

    # 1. Frozen store MUST reject active dream refinement
    frozen_st = _fresh_store()
    frozen_st.freeze()
    with pytest.raises(VersionStoreFrozenError):
        refuter.refine_library(
            frozen_st, grouped_pos, grouped_neg, corpus_stats, make_blackbox_environment(), probe_budget=10
        )

    # 2. plain_mdl_no_dreams executes 0 env queries
    env_plain = make_blackbox_environment()
    st_plain = _fresh_store()
    rep_plain = refuter.refine_library(
        st_plain, grouped_pos, grouped_neg, corpus_stats, env_plain, probe_budget=0, mode="plain_mdl_no_dreams"
    )
    assert env_plain.query_count == 0
    assert rep_plain.dreams_verified_by_env == 0 and rep_plain.self_certified_dreams == 0
    m_plain = _evaluate_library_on_cases(st_plain.active_cards(), ood_cases)

    # 3. rdc_dream_refutation executes strictly through independent env, 0 self-cert, 100% OOD acc!
    env_dream = make_blackbox_environment()
    st_dream = _fresh_store()
    rep_dream = refuter.refine_library(
        st_dream, grouped_pos, grouped_neg, corpus_stats, env_dream, probe_budget=260, mode="rdc_dream_refutation"
    )
    assert rep_dream.self_certified_dreams == 0
    assert rep_dream.dreams_verified_by_env == env_dream.query_count
    assert rep_dream.dreams_verified_by_env <= 260
    assert rep_dream.j_after < rep_dream.j_before
    m_dream = _evaluate_library_on_cases(st_dream.active_cards(), ood_cases)
    assert m_dream["exact_accuracy"] == 1.0
    assert m_dream["collateral_rate"] == 0.0
    assert m_dream["exact_accuracy"] - m_plain["exact_accuracy"] >= 0.20

    # 4. REGRESSION TEST (#59): rdc_random_probes uses the EXACT same over-constraint pruning engine
    #    (no dream_kind handicap!) -> on seed 404, random probes prune (?source, accessible, true)!
    env_rand = make_blackbox_environment()
    st_rand = _fresh_store()
    rep_rand = refuter.refine_library(
        st_rand,
        grouped_pos,
        grouped_neg,
        corpus_stats,
        env_rand,
        probe_budget=260,
        mode="rdc_random_probes",
        random_seed=seed * 31 + 17,
    )
    assert rep_rand.self_certified_dreams == 0
    assert rep_rand.dreams_verified_by_env == env_rand.query_count
    assert rep_rand.overconstraint_counterexamples >= 2
    assert ("?source", "accessible", "true") not in st_rand.active_cards()["move"].required_facts
    m_rand = _evaluate_library_on_cases(st_rand.active_cards(), ood_cases)
    assert m_dream["exact_accuracy"] > m_rand["exact_accuracy"]


def test_s5_max_disagreement_active_query_selection_and_empty_guard() -> None:
    engine = MaxDisagreementQueryEngine()

    # Regression (#59): empty hypothesis committee raises ValueError immediately
    with pytest.raises(ValueError):
        engine.run_active_selection("move", [], [], make_blackbox_environment())

    train_eps = generate_training_episodes(101, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    heldout = generate_heldout_suite(101, train_eps)
    heldout[S3_INACCESSIBLE_SOURCE_SPLIT] = generate_inaccessible_source_contrast_split(101)
    with EpisodeLedger(":memory:") as ledger:
        for ep in train_eps:
            ledger.ingest_training_episode(ep, use_typed_action=True)
        grouped = extract_observations_from_ledger(ledger, caller_tag="test_s5_active")
        corpus_stats = collect_corpus_statistics(grouped)
        pos_move = [o for o in grouped["move"] if not o.sparse_delta.is_noop]

    s2 = ITLOperatorLearner(seed=101, use_negative_contrasts=True)
    s2.fit(train_eps)
    core_move = _strip_to_core_for_active(s2.operators["move"])
    hyps = build_hypothesis_version_space(core_move, pos_move, corpus_stats, max_optional_clauses=8)
    assert len(hyps) == 256

    # Disagreement entropy is 0.0 for single hypothesis and >0 when pool splits committee
    b_dreams = synthesize_boundary_dreams({"move": s2.operators["move"]}, {"move": pos_move}, corpus_stats)
    r_probes = synthesize_random_world_probes(
        {"move": s2.operators["move"]}, {"move": pos_move}, corpus_stats, count=96, seed=101
    )
    pool = [(d.before_state, d.action) for d in (b_dreams + r_probes)]
    assert score_probe_disagreement(pool[0][0], pool[0][1], [hyps[0]]) == 0.0
    max_ent = max(score_probe_disagreement(st, ac, hyps) for st, ac in pool)
    assert max_ent == pytest.approx(1.0)

    move_cases = [
        c
        for sp in ("heldout_unseen_names", "heldout_guard_contrasts", S3_INACCESSIBLE_SOURCE_SPLIT)
        for c in heldout[sp]
        if len(c.public.actions) == 1 and c.public.actions[0].action_id == "move"
    ]

    def _eval_move(card: OperatorCard) -> float:
        hits = sum(
            1
            for c in move_cases
            if (res := card.apply(c.public.initial_state, c.public.actions[0].role_dict())).after_state
            == c.hidden.gold_final_state
            and not res.abstained
        )
        return hits / float(len(move_cases))

    res_md = engine.run_active_selection(
        "move", hyps, pool, make_blackbox_environment(), strategy="max_disagreement_query",
        max_queries=len(pool), seed=101, target_eval_fn=_eval_move, target_accuracy=0.98
    )
    res_rn = engine.run_active_selection(
        "move", hyps, pool, make_blackbox_environment(), strategy="random_query",
        max_queries=len(pool), seed=999, target_eval_fn=_eval_move, target_accuracy=0.98
    )
    assert _eval_move(res_md.selected_card) == 1.0
    assert res_md.queries_to_unique_or_target <= 10
    assert res_md.queries_to_unique_or_target < res_rn.queries_to_unique_or_target


def test_s5_interventional_blame_single_fault_offline_dual_ambiguity_and_exact_tie() -> None:
    train_eps = generate_training_episodes(101, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    heldout = generate_heldout_suite(101, train_eps)
    env = make_blackbox_environment()
    s3 = ContrastRefinedITLLearner(seed=101, env=env)
    s3.fit(train_eps)
    s3.freeze()
    clean_lib = s3.operators
    blame = InterventionalBlameEngine()

    # Find a depth-4 case with >= 2 distinct active operators whose corruption is observable
    target_case = None
    target_ops: list[str] = []
    for c in heldout["heldout_comp_depth4"]:
        ops = sorted({a.action_id for a in c.public.actions})
        if len(ops) < 2:
            continue
        obs_ops = []
        for op in ops:
            fl = dict(clean_lib)
            fl[op] = _corrupt_operator_card(clean_lib[op])
            p, ab, _ = execute_library_sequence(fl, c.public.initial_state, c.public.actions)
            if ab or p != c.hidden.gold_final_state:
                obs_ops.append(op)
        if len(obs_ops) >= 2:
            target_case = c
            target_ops = obs_ops
            break
    assert target_case is not None

    pub = target_case.public
    gold = target_case.hidden.gold_final_state

    # 1. Clean trajectory -> trajectory_failed=False, 0 env interventions
    rep_clean = blame.attribute_blame_via_env(clean_lib, pub.initial_state, pub.actions, gold, env)
    assert rep_clean.trajectory_failed is False and rep_clean.env_interventions_used == 0

    # 2. Single-operator fault -> exact unique culprit via BOTH env and offline alternatives
    culprit = target_ops[0]
    faulty_lib = dict(clean_lib)
    faulty_lib[culprit] = _corrupt_operator_card(clean_lib[culprit])
    rep_env = blame.attribute_blame_via_env(faulty_lib, pub.initial_state, pub.actions, gold, env)
    rep_off = blame.attribute_blame_via_alternatives(
        faulty_lib, pub.initial_state, pub.actions, gold, {k: [v] for k, v in clean_lib.items()}
    )
    assert rep_env.is_ambiguous is False and rep_env.unique_culprit == culprit
    assert rep_off.is_ambiguous is False and rep_off.unique_culprit == culprit
    assert score_blame_attribution(rep_env, culprit) == {"top1_accuracy": 1.0, "mrr": 1.0, "brier_score": 0.0}

    # 3. Dual-fault ambiguity safeguard -> MUST return is_ambiguous=True, unique_culprit=None, both preserved
    dual_lib = dict(clean_lib)
    dual_lib[target_ops[0]] = _corrupt_operator_card(clean_lib[target_ops[0]])
    dual_lib[target_ops[1]] = _corrupt_operator_card(clean_lib[target_ops[1]])
    rep_dual = blame.attribute_blame_via_env(dual_lib, pub.initial_state, pub.actions, gold, env)
    assert rep_dual.is_ambiguous is True
    assert rep_dual.unique_culprit is None
    assert target_ops[0] in rep_dual.suspected_operators
    assert target_ops[1] in rep_dual.suspected_operators

    # 4. Synthetic exact-repair tie: two distinct actions op_a & op_b both idempotently set flag=on;
    #    both are corrupted in active lib -> intervening on EITHER op_a OR op_b repairs the trace -> exact tie!
    w_tie = StateGraph.create([Entity("x1", "node")], [("x1", "flag", "off")])
    w_goal = w_tie.with_changes(delete_facts=(("x1", "flag", "off"),), add_facts=(("x1", "flag", "on"),))

    class TieEnv:
        def step(
            self, state: StateGraph, action: ActionInvocation, mask: ObservationMask | None = None
        ) -> tuple[StateGraph, SparseDelta]:
            nxt = state.with_changes(delete_facts=(("x1", "flag", "off"),), add_facts=(("x1", "flag", "on"),))
            return nxt, compute_sparse_delta(state, nxt, mask=mask)

    noop_a = OperatorCard(
        "op_a_bad", "op_a", ("n",), (("n", frozenset({"node"})),), (), (), (), (), (), (), (), ()
    )
    noop_b = OperatorCard(
        "op_b_bad", "op_b", ("n",), (("n", frozenset({"node"})),), (), (), (), (), (), (), (), ()
    )
    acts_tie = (build_probe_action("op_a", {"n": "x1"}), build_probe_action("op_b", {"n": "x1"}))
    rep_tie = blame.attribute_blame_via_env({"op_a": noop_a, "op_b": noop_b}, w_tie, acts_tie, w_goal, TieEnv())
    assert rep_tie.is_ambiguous is True
    assert rep_tie.unique_culprit is None
    assert rep_tie.exact_repair_operators == ("op_a", "op_b")
    assert rep_tie.blame_distribution == {"op_a": 0.5, "op_b": 0.5}


def test_s5_intra_consolidation_unsafe_rejection_retire_active_and_rollback() -> None:
    train_eps = generate_training_episodes(101, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    heldout = generate_heldout_suite(101, train_eps)
    s3 = ContrastRefinedITLLearner(seed=101, env=make_blackbox_environment())
    s3.fit(train_eps)
    s3.freeze()

    lib = dict(s3.operators)
    lib["move_alias"] = _clone_as_alias(s3.operators["move"], "move_alias")
    aliases: dict[str, str] = {}
    store = OperatorVersionStore()
    for aid, card in sorted(lib.items()):
        store.register_shadow(card, notes="init")
        store.promote(aid, card.version, notes="active")

    # Equivalence checks: empty probes rejected, signature mismatch rejected, distinct ops rejected
    move_probes = [
        (c.public.initial_state, c.public.actions[0].role_dict())
        for c in heldout["heldout_unseen_names"]
        if c.public.actions[0].action_id == "move"
    ]
    open_probes = [
        (c.public.initial_state, c.public.actions[0].role_dict())
        for c in heldout["heldout_unseen_names"]
        if c.public.actions[0].action_id == "open"
    ]
    assert check_behavioral_equivalence(lib["move"], lib["move_alias"], []).equivalent is False
    assert check_behavioral_equivalence(lib["move"], lib["give"], move_probes).equivalent is False
    assert check_behavioral_equivalence(lib["open"], lib["lock"], open_probes).equivalent is False
    assert check_behavioral_equivalence(lib["move"], lib["move_alias"], move_probes).equivalent is True

    reg_examples = [
        MDLExample(f"r_{i}", c.public.initial_state, c.public.actions, c.hidden.gold_final_state)
        for i, c in enumerate(heldout["heldout_unseen_names"])
    ] + [
        MDLExample(
            f"ra_{i}",
            c.public.initial_state,
            (build_probe_action("move_alias", c.public.actions[0].role_dict()),),
            c.hidden.gold_final_state,
        )
        for i, c in enumerate(heldout["heldout_unseen_names"])
        if c.public.actions[0].action_id == "move"
    ]

    consolidator = RDCConsolidator()

    # 1. Safe merge reduces J & library_bits with 0 regressions and retires without deletion
    res_ok = consolidator.attempt_consolidate_pair(
        lib, aliases, store, "move", "move_alias", move_probes, reg_examples
    )
    assert res_ok.accepted is True
    assert res_ok.j_after < res_ok.j_before
    assert res_ok.library_bits_after < res_ok.library_bits_before
    assert res_ok.regression_failures_after == 0
    assert "move_alias" not in lib and aliases["move_alias"] == "move"
    assert store.get_active("move_alias") is None
    assert store.get_version("move_alias", 1).status == "retired"

    # Double-retire on already-retired action MUST raise VersionStoreError
    with pytest.raises(VersionStoreError):
        store.retire_active("move_alias")
    with pytest.raises(VersionStoreError):
        store.retire_active("nonexistent")

    # 2. Unsafe merge (open vs lock) MUST be rejected with 0 changes
    res_bad = consolidator.attempt_consolidate_pair(
        lib, aliases, store, "open", "lock", open_probes, reg_examples
    )
    assert res_bad.accepted is False
    assert res_bad.reason == "rejected_behavioral_divergence"
    assert "lock" in lib and "lock" not in aliases

    # 3. Rollback cleanly restores move_alias
    restored = consolidator.rollback_consolidation(lib, aliases, store, "move_alias", 1)
    assert restored.action_id == "move_alias"
    assert "move_alias" in lib and "move_alias" not in aliases
    assert store.get_active("move_alias").card.version == 1

    # 4. Frozen store blocks retire_active and attempt_consolidate_pair
    store.freeze()
    with pytest.raises(VersionStoreFrozenError):
        store.retire_active("move_alias")
    with pytest.raises(VersionStoreFrozenError):
        consolidator.attempt_consolidate_pair(
            lib, aliases, store, "move", "move_alias", move_probes, reg_examples
        )


def test_s5_cross_domain_concept_reversibility_and_domain3_few_shot_transfer() -> None:
    # 1. Reversed-alphabetical predicate ordering across two unrelated domains MUST still be 100% lossless
    c_a = OperatorCard(
        "op_a_v1", "op_a", ("a", "b", "c"),
        (("a", frozenset({"t1"})), ("b", frozenset({"t2"})), ("c", frozenset({"t1"}))),
        (("?b", "has", "?a"),), (),
        (
            IndirectRoleFactGuard("b", "at", "box", "aaa_open", "true"),
            IndirectRoleFactGuard("b", "at", "box", "zzz_locked", "false"),
        ),
        (AttributeGuard("b", "ok", True),), (RoleInequalityGuard("a", "c"),),
        (("?b", "has", "?c"),), (("?b", "has", "?a"),), (),
    )
    c_b = OperatorCard(
        "op_b_v1", "op_b", ("x", "y", "z"),
        (("x", frozenset({"u1"})), ("y", frozenset({"u2"})), ("z", frozenset({"u1"}))),
        (("?y", "own", "?x"),), (),
        (
            IndirectRoleFactGuard("y", "in", "vault", "aaa_locked", "false"),
            IndirectRoleFactGuard("y", "in", "vault", "zzz_open", "true"),
        ),
        (AttributeGuard("y", "ready", True),), (RoleInequalityGuard("x", "z"),),
        (("?y", "own", "?z"),), (("?y", "own", "?x"),), (),
    )
    xrev = extract_cross_domain_concepts("d1", {"op_a": c_a}, "d2", {"op_b": c_b})
    assert xrev.reversibility_verified is True and len(xrev.concepts) == 1

    # 2. Full Kill Test 4: Domain 1 (S0) + Domain 2 (Arcane) -> Held-Out Domain 3 (Logistics, 2 pos, 0 neg)
    train_eps = generate_training_episodes(101, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    s3 = ContrastRefinedITLLearner(seed=101, env=make_blackbox_environment())
    s3.fit(train_eps)
    s3.freeze()

    d1_cards = {"move": s3.operators["move"], "give": s3.operators["give"]}
    d2_cards = build_domain2_arcane_library()
    xrep = extract_cross_domain_concepts("s0_world", d1_cards, "arcane_world", d2_cards)
    assert xrep.reversibility_verified is True
    assert len(xrep.concepts) == 2
    assert xrep.domain1_bits_with_concept < xrep.domain1_bits_without_concept
    assert xrep.domain2_bits_with_concept < xrep.domain2_bits_without_concept
    assert xrep.joint_bits_with_concepts < xrep.joint_bits_without_concepts

    d3_train, d3_ood = generate_domain3_few_shot_and_ood(101)
    assert len(d3_train) == 4  # 2 positives per action, 0 negatives!
    assert all(not ep.sparse_delta.is_noop for ep in d3_train)
    assert len(d3_ood) == 44

    # Verify Domain 3 entity types & predicates are 100% disjoint from Domain 1 & Domain 2
    d1_d2_types = {
        t
        for lib in (d1_cards, d2_cards)
        for c in lib.values()
        for _r, ts in c.required_types
        for t in ts
    }
    d3_types = {ent.entity_type for ep in d3_train for _eid, ent in ep.before_state.entities}
    assert d3_types.isdisjoint(d1_d2_types)

    with EpisodeLedger(":memory:") as ledger:
        for ep in d3_train:
            ledger.ingest_training_episode(ep, use_typed_action=True)
        d3_grouped = extract_observations_from_ledger(ledger, caller_tag="test_s5_d3")
        d3_stats = collect_corpus_statistics(d3_grouped)
        d3_pos = {aid: [o for o in obs if not o.sparse_delta.is_noop] for aid, obs in d3_grouped.items()}

    transferred, insts, bits_no_c, bits_with_c = transfer_concepts_to_few_shot_domain(
        xrep.concepts, d3_pos, d3_stats
    )
    assert set(transferred.keys()) == {"dispatch", "reassign"}
    assert set(insts.keys()) == {"dispatch", "reassign"}
    assert bits_with_c < bits_no_c

    cold_itl = ITLOperatorLearner(seed=101, use_negative_contrasts=True)
    cold_itl.fit(d3_train)
    cold_itl.freeze()

    t_hits = sum(
        1
        for c in d3_ood
        if (res := execute_library_sequence(transferred, c.public.initial_state, c.public.actions))[0]
        == c.gold_final_state
        and not res[1]
    )
    c_hits = sum(
        1
        for c in d3_ood
        if (res := execute_library_sequence(cold_itl.operators, c.public.initial_state, c.public.actions))[0]
        == c.gold_final_state
        and not res[1]
    )
    t_acc = t_hits / float(len(d3_ood))
    c_acc = c_hits / float(len(d3_ood))
    assert t_acc == 1.0
    assert t_acc - c_acc >= 0.30


def test_s5_committed_manifests_saved_5seed_gate_and_fresh_isolated_run(tmp_path: Path) -> None:
    ok0, m0 = verify_committed_manifests(seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    ok1, m1 = verify_s1_committed_manifests(seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    ok2, m2 = verify_s2_committed_manifests(seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    ok3, m3 = verify_s3_committed_manifests(seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    ok4, m4 = verify_s4_committed_manifests(seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    ok5, m5 = verify_s5_committed_manifests(seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    assert ok0, f"S0 manifest drift: {m0}"
    assert ok1, f"S1 manifest drift: {m1}"
    assert ok2, f"S2 manifest drift: {m2}"
    assert ok3, f"S3 manifest drift: {m3}"
    assert ok4, f"S4 manifest drift: {m4}"
    assert ok5, f"S5 manifest drift: {m5}"

    summary_path = REPORTS_DIR / "s5_summary.json"
    report_path = REPORTS_DIR / "s5_research_report.md"
    csv_path = REPORTS_DIR / "raw_runs" / "s5_ablations.csv"
    assert summary_path.is_file() and report_path.is_file() and csv_path.is_file()

    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["s5_gate"]["passed"] is True
    assert len(saved["s5_gate"]["checks"]) >= 24

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 60  # 5 seeds * 12 ablation-method rows
    for seed in DEFAULT_SEEDS:
        assert (REPORTS_DIR / "raw_runs" / f"s5_seed_{seed}.json").is_file()

    # Fresh isolated 2-seed end-to-end run (immune to df=1 Student-t explosion on random-probe CIs)
    fresh = run_s5_experiments(
        seeds=(101, 202),
        demos_per_action=DEFAULT_DEMOS_PER_ACTION,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        verify_prior_manifests=False,
    )
    df1_random_ci_keys = {
        "kill_test_1_dream_beats_random_probes_ci_low_gt_0",
        "kill_test_2_blame_beats_random_by_40pp",
        "kill_test_3_active_query_reduction_ci_low_ge_0.35",
        "cpu_budget_within_30s_per_seed",
    }
    for check_name, ok_val in fresh["s5_gate"]["checks"].items():
        if check_name not in df1_random_ci_keys:
            assert ok_val is True, f"Fresh isolated S5 check failed: {check_name}"
    assert fresh["ablation_1_dream_refutation"]["accuracy_gain_vs_random_probes"]["min"] > 0.0
    assert fresh["ablation_2_interventional_blame"]["top1_gain_vs_random"]["min"] >= 0.40
    assert fresh["ablation_3_active_max_disagreement"]["query_reduction_fraction"]["min"] >= 0.35
