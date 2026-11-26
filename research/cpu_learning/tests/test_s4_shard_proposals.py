"""Automated Stage S4 tests: SHARD 10,240-bit BSC Encoders, Multi-Table LSH Indexes,
Warm-Started Task A Recovery & Task B Multi-Step Planning, Task C Bind-Depth Geometry,
Novel Synthetic Domain Generalization, Ablations, AST Isolation, and 5-Seed Gates.

STRICT ISOLATION: Never imports ``hidden_actions`` or ``oracle``.
"""
from __future__ import annotations

import ast
import csv
import heapq
import json
from pathlib import Path

import pytest

from mirror_crystal.quotient_egraph_resonator import (
    D_UNIFIED,
    UNIFIED_MASK,
    UNIFIED_ODD_MASK,
    hv10240_of,
)
from research.cpu_learning.evaluation.run_experiments import (
    REPORTS_DIR,
    make_blackbox_environment,
    verify_committed_manifests,
)
from research.cpu_learning.evaluation.run_s1_experiments import verify_s1_committed_manifests
from research.cpu_learning.evaluation.run_s2_experiments import verify_s2_committed_manifests
from research.cpu_learning.evaluation.run_s3_experiments import verify_s3_committed_manifests
from research.cpu_learning.evaluation.run_s4_experiments import (
    audit_s4_ast_isolation_and_agnosticism,
    build_synthetic_distractor_operators,
    run_s4_experiments,
    verify_s4_committed_manifests,
)
from research.cpu_learning.itl.contrast_harness import ContrastRefinedITLLearner
from research.cpu_learning.itl.operator_card import (
    AttributeGuard,
    LiftedAttributeEdit,
    OperatorCard,
    RoleInequalityGuard,
)
from research.cpu_learning.ledger.episode_ledger import EpisodeLedger
from research.cpu_learning.shard.encoders import (
    _majority_bundle10240,
    cleanup_entity_ranking,
    effect_slot_key_hv,
    encode_delta_entity_slots_hv,
    encode_operator_card_hv,
    encode_role_bindings_hv,
    encode_sequence_hv,
    encode_sparse_delta_hv,
    encode_typed_state_hv,
    role_key_hv,
    unbind_role_slot_hv,
    unbind_step_from_sequence,
)
from research.cpu_learning.shard.episodic_index import (
    IndexFrozenError,
    SHARDEpisodicIndex,
    SHARDOperatorIndex,
    _MultiTableLSHCore,
)
from research.cpu_learning.shard.proposer import (
    SHARDProposalEngine,
    _state_diff_count,
    _StateBindingIndex,
    enumerate_compatible_bindings,
    run_bind_depth_geometry_probe,
)
from research.cpu_learning.simulator.generators import (
    DEFAULT_DEMOS_PER_ACTION,
    DEFAULT_SEEDS,
    generate_heldout_suite,
    generate_training_episodes,
)
from research.cpu_learning.simulator.state_graph import (
    Entity,
    SparseDelta,
    StateGraph,
    compute_sparse_delta,
)

PKG_ROOT = Path(__file__).resolve().parents[1]


def test_s4_ast_isolation_and_zero_hardcoded_domain_literals() -> None:
    audit = audit_s4_ast_isolation_and_agnosticism()
    assert audit["passed"] is True, f"S4 AST audit failed: {audit}"
    assert audit["import_violations"] == []
    assert audit["hardcode_violations"] == []
    assert audit["scanned_shard_files"] >= 4
    assert audit["scanned_itl_files"] >= 7
    assert audit["scanned_learner_files"] >= 15

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
        "source",
        "destination",
        "recipient",
        "target",
    }
    for fpath in sorted((PKG_ROOT / "shard").glob("*.py")):
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
        assert hits == set(), f"shard/{fpath.name} contained forbidden S0 literals: {sorted(hits)}"


def test_s4_bsc10240_encoders_exact_majority_overflow_safety_invariance_and_unbinding() -> None:
    # 1. Exact majority verification against integer bit-sum reference (odd, even, weighted, zero-weight)
    hvs = [hv10240_of(f"enc_test_{i}") for i in range(5)]
    for weights in ([1, 1, 1], [1, 1, 1, 1], [3, 2, 1, 1], [0, 2, 1, 0, 1]):
        pairs = list(zip(hvs, weights))
        got = _majority_bundle10240(pairs, empty_hv=0, bits=10)
        total_w = sum(w for w in weights if w > 0)
        ref = 0
        for b in range(D_UNIFIED):
            s = sum(w for hv, w in pairs if w > 0 and ((hv >> b) & 1))
            if s * 2 > total_w:
                ref |= 1 << b
            elif s * 2 == total_w and ((UNIFIED_ODD_MASK >> b) & 1):
                ref |= 1 << b
        assert got == ref

    # 2. Overflow safety regression: large count of identical vectors at bits=8 MUST NOT wrap around!
    v = hv10240_of("overflow_canary_vec")
    for count in (128, 256, 512):
        bundled = _majority_bundle10240([(v, 1)] * count, empty_hv=0, bits=8)
        assert bundled == (v & UNIFIED_MASK), f"Bit-plane wraparound detected at count={count}"

    # 3. Entity-ID invariance of encode_typed_state_hv and encode_sparse_delta_hv
    s_a = StateGraph.create(
        [Entity("wA", "wizard"), Entity("rA", "rune", (("charged", True),)), Entity("a1", "altar"), Entity("a2", "altar")],
        [("rA", "attuned_to", "wA"), ("rA", "anchored_at", "a1"), ("a2", "reachable", "true")],
    )
    s_a_after = s_a.with_changes(delete_facts=(("rA", "anchored_at", "a1"),), add_facts=(("rA", "anchored_at", "a2"),))
    d_a = compute_sparse_delta(s_a, s_a_after)

    s_b = StateGraph.create(
        [Entity("zZZ", "wizard"), Entity("k99", "rune", (("charged", True),)), Entity("p7", "altar"), Entity("q8", "altar")],
        [("k99", "attuned_to", "zZZ"), ("k99", "anchored_at", "p7"), ("q8", "reachable", "true")],
    )
    s_b_after = s_b.with_changes(delete_facts=(("k99", "anchored_at", "p7"),), add_facts=(("k99", "anchored_at", "q8"),))
    d_b = compute_sparse_delta(s_b, s_b_after)

    assert encode_typed_state_hv(s_a) == encode_typed_state_hv(s_b)
    assert encode_sparse_delta_hv(d_a, s_a) == encode_sparse_delta_hv(d_b, s_b)

    # 4. Role binding & delta slot XOR unbinding recovers exact entity IDs
    roles_hv = encode_role_bindings_hv({"caster": "wA", "relic": "rA", "origin": "a1", "dest": "a2"}, s_a)
    for rname, expected_eid in (("caster", "wA"), ("relic", "rA"), ("origin", "a1"), ("dest", "a2")):
        unbound = unbind_role_slot_hv(roles_hv, role_key_hv(rname))
        ranked = cleanup_entity_ranking(unbound, ["wA", "rA", "a1", "a2"])
        assert ranked[0][0] == expected_eid

    slot_hv = encode_delta_entity_slots_hv(d_a, s_a)
    top_del_obj = cleanup_entity_ranking(
        unbind_role_slot_hv(slot_hv, effect_slot_key_hv("del_obj", "anchored_at")), ["a1", "a2"]
    )[0][0]
    top_add_obj = cleanup_entity_ranking(
        unbind_role_slot_hv(slot_hv, effect_slot_key_hv("add_obj", "anchored_at")), ["a1", "a2"]
    )[0][0]
    assert top_del_obj == "a1" and top_add_obj == "a2"

    # 5. Sequence role permutation preserves order and repeats ([A, B, A, C] vs [C, A, B, A])
    cb = {lbl: hv10240_of(f"seq_sym:{lbl}") for lbl in ("A", "B", "C", "D", "E", "F")}
    seq1 = ["A", "B", "A", "C", "D", "B"]
    seq2 = list(reversed(seq1))
    hv1 = encode_sequence_hv([cb[x] for x in seq1], use_role_permutation=True)
    hv2 = encode_sequence_hv([cb[x] for x in seq2], use_role_permutation=True)
    assert hv1 != hv2
    for idx, expected in enumerate(seq1):
        lbl, sim = unbind_step_from_sequence(hv1, idx, cb, use_role_permutation=True)
        assert lbl == expected and sim > 1500
    for idx, expected in enumerate(seq2):
        lbl, sim = unbind_step_from_sequence(hv2, idx, cb, use_role_permutation=True)
        assert lbl == expected and sim > 1500


def test_s4_lsh_indexes_fallback_resorting_freeze_and_disk_ledger(tmp_path: Path) -> None:
    # 1. Regression test: include_fallback=True MUST scan missing vectors & re-sort even when >= top_k decoys hit buckets!
    core = _MultiTableLSHCore(num_tables=2, bits_per_table=8, seed_tag="fallback_regression")
    target = hv10240_of("fallback_target")
    lsh_mask = 0
    for tbl in core._bit_tables:
        for b in tbl:
            lsh_mask |= 1 << b
    decoy = ((~target) & UNIFIED_MASK & ~lsh_mask) | (target & lsh_mask)
    for _ in range(6):
        core.add_vector(decoy)
    near_exact = (target & ~lsh_mask) | ((~target) & lsh_mask)
    near_idx = core.add_vector(near_exact)

    # Without fallback, LSH bucket shortlist only sees the 6 colliding decoys
    no_fb = core.query_ranked_indices(target, top_k=5, include_fallback=False)
    assert all(i != near_idx for i, _ in no_fb)
    assert all(s < -9000 for _, s in no_fb)

    # With fallback, missing vector MUST be scored and sorted to Rank #1!
    with_fb = core.query_ranked_indices(target, top_k=5, include_fallback=True)
    assert len(with_fb) == 5
    assert with_fb[0][0] == near_idx
    assert with_fb[0][1] > 10000

    # 2. SHARDOperatorIndex duplicate detection, bit accounting, and freeze
    dummy_card = OperatorCard(
        card_id="op_zap_v1",
        action_id="zap",
        role_variables=("caster", "orb"),
        required_types=(("caster", frozenset({"wizard"})), ("orb", frozenset({"rune"}))),
        required_facts=(("?orb", "lit", "false"),),
        conditional_fact_guards=(),
        indirect_fact_guards=(),
        attribute_guards=(),
        inequality_guards=(),
        add_facts=(("?orb", "lit", "true"),),
        delete_facts=(("?orb", "lit", "false"),),
        attribute_edits=(),
    )
    op_idx = SHARDOperatorIndex(num_tables=8, bits_per_table=4, seed_tag="op_unit")
    op_idx.add_operator(dummy_card)
    with pytest.raises(ValueError):
        op_idx.add_operator(dummy_card)
    assert op_idx.index_bits() > D_UNIFIED
    assert op_idx.compressed_index_bits() > 0
    op_idx.freeze()
    assert op_idx.is_frozen is True
    with pytest.raises(IndexFrozenError):
        op_idx.add_operator(dummy_card)

    # 3. SHARDEpisodicIndex indexing directly from a frozen on-disk S1 EpisodeLedger + zero subsequent ledger reads
    train_eps = generate_training_episodes(101, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    db_path = tmp_path / "s4_ep_ledger.sqlite3"
    with EpisodeLedger(db_path) as ledger:
        for ep in train_eps:
            ledger.ingest_training_episode(ep, use_typed_action=True)
        ledger.freeze()

        ep_idx = SHARDEpisodicIndex(num_tables=12, bits_per_table=4, seed_tag="ep_disk")
        added = ep_idx.index_from_ledger(ledger, split_id="train", effect_only=True)
        assert added == 16  # 4 actions * 4 effect-bearing positive demos
        ep_idx.freeze()
        ledger.reset_audit_log()

        with pytest.raises(IndexFrozenError):
            ep_idx.add_episode(train_eps[0])
        with pytest.raises(IndexFrozenError):
            ep_idx.index_from_ledger(ledger)

        hits = ep_idx.query_by_delta(train_eps[0].sparse_delta, train_eps[0].before_state, top_k=3)
        assert len(hits) >= 1 and hits[0][0].action_id == train_eps[0].action.action_id
        assert ledger.audit_summary()["total_accesses"] == 0


def test_s4_novel_synthetic_domain_recovery_planning_and_canceling_enabler() -> None:
    """Prove SHARDProposalEngine generalizes to a 100% novel synthetic domain (36 ops, 0 S0 literals),
    rejects unsolvable deltas honestly, and solves both normal & canceling-enabler multi-step plans."""
    op_channel = OperatorCard(
        card_id="op_channel_v1",
        action_id="channel",
        role_variables=("caster", "relic", "origin", "dest"),
        required_types=(
            ("caster", frozenset({"wizard"})),
            ("relic", frozenset({"rune"})),
            ("origin", frozenset({"altar", "sanctum"})),
            ("dest", frozenset({"altar", "sanctum"})),
        ),
        required_facts=(
            ("?relic", "attuned_to", "?caster"),
            ("?relic", "anchored_at", "?origin"),
            ("?dest", "reachable", "true"),
        ),
        conditional_fact_guards=(),
        indirect_fact_guards=(),
        attribute_guards=(AttributeGuard("relic", "charged", True),),
        inequality_guards=(RoleInequalityGuard("origin", "dest"),),
        add_facts=(("?relic", "anchored_at", "?dest"),),
        delete_facts=(("?relic", "anchored_at", "?origin"),),
        attribute_edits=(),
    )
    op_bestow = OperatorCard(
        card_id="op_bestow_v1",
        action_id="bestow",
        role_variables=("caster", "relic", "heir"),
        required_types=(
            ("caster", frozenset({"wizard"})),
            ("relic", frozenset({"rune"})),
            ("heir", frozenset({"wizard"})),
        ),
        required_facts=(("?relic", "attuned_to", "?caster"),),
        conditional_fact_guards=(),
        indirect_fact_guards=(),
        attribute_guards=(),
        inequality_guards=(RoleInequalityGuard("caster", "heir"),),
        add_facts=(("?relic", "attuned_to", "?heir"),),
        delete_facts=(("?relic", "attuned_to", "?caster"),),
        attribute_edits=(),
    )
    op_unseal = OperatorCard(
        card_id="op_unseal_v1",
        action_id="unseal",
        role_variables=("caster", "shrine"),
        required_types=(("caster", frozenset({"wizard"})), ("shrine", frozenset({"sanctum"}))),
        required_facts=(("?shrine", "gate_open", "false"),),
        conditional_fact_guards=(),
        indirect_fact_guards=(),
        attribute_guards=(),
        inequality_guards=(),
        add_facts=(("?shrine", "gate_open", "true"), ("?shrine", "reachable", "true")),
        delete_facts=(("?shrine", "gate_open", "false"), ("?shrine", "reachable", "false")),
        attribute_edits=(LiftedAttributeEdit("shrine", "seal_label", "open"),),
    )
    op_reseal = OperatorCard(
        card_id="op_reseal_v1",
        action_id="reseal",
        role_variables=("caster", "shrine"),
        required_types=(("caster", frozenset({"wizard"})), ("shrine", frozenset({"sanctum"}))),
        required_facts=(("?shrine", "gate_open", "true"),),
        conditional_fact_guards=(),
        indirect_fact_guards=(),
        attribute_guards=(),
        inequality_guards=(),
        add_facts=(("?shrine", "gate_open", "false"), ("?shrine", "reachable", "false")),
        delete_facts=(("?shrine", "gate_open", "true"), ("?shrine", "reachable", "true")),
        attribute_edits=(LiftedAttributeEdit("shrine", "seal_label", "closed"),),
    )

    library: dict[str, OperatorCard] = {
        "channel": op_channel,
        "bestow": op_bestow,
        "unseal": op_unseal,
        "reseal": op_reseal,
    }
    # Add 32 novel-domain distractors (36 ops total)
    for k in range(32):
        aid = f"arcane_aux_{k:02d}"
        library[aid] = OperatorCard(
            card_id=f"op_{aid}_v1",
            action_id=aid,
            role_variables=("caster", "relic"),
            required_types=(("caster", frozenset({"wizard"})), ("relic", frozenset({"rune"}))),
            required_facts=(("?relic", "attuned_to", "?caster"),),
            conditional_fact_guards=(),
            indirect_fact_guards=(),
            attribute_guards=(),
            inequality_guards=(),
            add_facts=(("?relic", f"glyph_mark_{k:02d}", "true"),),
            delete_facts=(),
            attribute_edits=(),
        )

    engine = SHARDProposalEngine(library, episodic_index=None, seed_tag="novel_arcane_s4")
    engine.freeze()

    w0 = StateGraph.create(
        [
            Entity("mage_x", "wizard"),
            Entity("mage_y", "wizard"),
            Entity("rune_1", "rune", (("charged", True),)),
            Entity("rune_2", "rune", (("charged", True),)),
            Entity("altar_1", "altar"),
            Entity("sanctum_1", "sanctum", (("seal_label", "closed"),)),
        ],
        [
            ("rune_1", "attuned_to", "mage_x"),
            ("rune_1", "anchored_at", "altar_1"),
            ("rune_2", "attuned_to", "mage_y"),
            ("rune_2", "anchored_at", "altar_1"),
            ("altar_1", "reachable", "true"),
            ("sanctum_1", "reachable", "false"),
            ("sanctum_1", "gate_open", "false"),
        ],
    )

    # 1. Task A recovery on novel unseen action (1 proposal vs >>10 unguided!)
    r_step1 = op_unseal.apply(w0, {"caster": "mage_x", "shrine": "sanctum_1"})
    assert r_step1.applied and r_step1.after_state is not None
    rec_sh = engine.recover_transition_shard_guided(w0, r_step1.delta)
    rec_ug = engine.recover_transition_unguided(w0, r_step1.delta)
    assert rec_sh.solved and rec_sh.itl_verified is True
    assert rec_sh.action_id == "unseal"
    assert rec_sh.proposals_evaluated == 1
    assert rec_ug.proposals_evaluated > 10

    # 2. Adversarial unsolvable delta MUST return solved=False (never hallucinate!)
    impossible_delta = SparseDelta(added_facts=(("rune_1", "completely_unknown_pred", "xyz"),))
    rec_imp = engine.recover_transition_shard_guided(w0, impossible_delta)
    assert rec_imp.solved is False and rec_imp.itl_verified is True

    # 3. Task B normal depth-3 plan: unseal -> channel -> bestow
    s1 = r_step1.after_state
    s2 = op_channel.apply(
        s1, {"caster": "mage_x", "relic": "rune_1", "origin": "altar_1", "dest": "sanctum_1"}
    ).after_state
    assert s2 is not None
    s3 = op_bestow.apply(s2, {"caster": "mage_x", "relic": "rune_1", "heir": "mage_y"}).after_state
    assert s3 is not None
    plan_res = engine.plan_goal_shard_guided(w0, s3, max_depth=4, max_proposals=600)
    assert plan_res.solved is True and plan_res.itl_verified is True and plan_res.final_state == s3

    # 4. Adversarial Task B canceling-enabler plan: unseal -> channel -> reseal
    #    Notice: compute_sparse_delta(w0, s_cancel) ONLY shows rune_1 moved to sanctum_1 (NOT gate_open!),
    #    so phase=1 completeness fallback MUST discover unseal -> channel -> reseal!
    s_cancel = op_reseal.apply(s2, {"caster": "mage_x", "shrine": "sanctum_1"}).after_state
    assert s_cancel is not None
    cancel_delta = compute_sparse_delta(w0, s_cancel)
    assert all(p != "gate_open" for _, p, _ in cancel_delta.added_facts + cancel_delta.deleted_facts)
    plan_cancel = engine.plan_goal_shard_guided(w0, s_cancel, max_depth=4, max_proposals=1500)
    assert plan_cancel.solved is True and plan_cancel.final_state == s_cancel and len(plan_cancel.plan) == 3


def test_s4_ablations_no_episodic_distractor_targets_best_first_and_similarity_not_proof() -> None:
    train_eps = generate_training_episodes(101, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    suite = generate_heldout_suite(101, train_eps)
    s3 = ContrastRefinedITLLearner(seed=101, env=make_blackbox_environment())
    s3.fit(train_eps)
    s3.freeze()

    distractors = build_synthetic_distractor_operators(44)
    full_lib = dict(s3.operators)
    full_lib.update(distractors)

    # 1. Pure SHARDOperatorIndex (episodic_index=None) still gets 1.0 proposals on Task A and solves depth-4!
    eng_no_ep = SHARDProposalEngine(full_lib, episodic_index=None, seed_tag="s4_ablation_no_ep")
    eng_no_ep.freeze()
    for case in suite["heldout_unseen_names"][:10]:
        r = eng_no_ep.recover_transition_shard_guided(
            case.public.initial_state, case.hidden.gold_step_deltas[0]
        )
        assert r.solved and r.proposals_evaluated == 1
        assert r.predicted_after_state == case.hidden.gold_intermediate_states[0]

    for case in suite["heldout_comp_depth4"][:4]:
        p = eng_no_ep.plan_goal_shard_guided(
            case.public.initial_state, case.hidden.gold_final_state, max_depth=4, max_proposals=600
        )
        assert p.solved and p.final_state == case.hidden.gold_final_state
        assert p.proposals_evaluated < 100

    # 2. Distractor-operator ground-truth targets (absent from episodic_index) STILL solve in 1 proposal!
    ep_idx = SHARDEpisodicIndex(seed_tag="s4_ablation_ep")
    ep_idx.index_episodes(train_eps, effect_only=True)
    eng_with_ep = SHARDProposalEngine(full_lib, episodic_index=ep_idx, seed_tag="s4_ablation_with_ep")
    eng_with_ep.freeze()

    w0 = suite["heldout_unseen_names"][0].public.initial_state
    sidx = _StateBindingIndex(w0)
    tested_distractors = 0
    for aid in sorted(distractors)[:8]:
        card = distractors[aid]
        roles = next(enumerate_compatible_bindings(card, sidx), None)
        if roles is None:
            continue
        res = card.apply(w0, roles)
        assert res.applied and not res.delta.is_noop
        rec = eng_with_ep.recover_transition_shard_guided(w0, res.delta)
        assert rec.solved and rec.action_id == aid and rec.proposals_evaluated == 1
        tested_distractors += 1
    assert tested_distractors >= 6

    # 3. Similarity-only without ITL verification fails on unseen disjoint entities (similarity != proof)
    for case in suite["heldout_unseen_names"][:8]:
        ret_only = eng_with_ep.recover_transition_retrieval_only_no_itl(
            case.public.initial_state, case.hidden.gold_step_deltas[0]
        )
        assert ret_only.itl_verified is False
        assert ret_only.solved is False

    # 4. Fairness ablation: unguided Best-First WITH _state_diff_count (NO SHARD) still hits 600 cap on depth-4!
    case_d4 = suite["heldout_comp_depth4"][0]
    init_s, goal_s = case_d4.public.initial_state, case_d4.hidden.gold_final_state
    heap = [(_state_diff_count(init_s, goal_s), 0, 0, init_s)]
    vis = {init_s.fingerprint(): 0}
    props, tie, ubf_solved = 0, 1, False
    while heap and props < 600:
        _h, d, _, cur = heapq.heappop(heap)
        if d >= 4:
            continue
        cur_sidx = _StateBindingIndex(cur)
        for aid in sorted(full_lib):
            card = full_lib[aid]
            for roles in enumerate_compatible_bindings(card, cur_sidx):
                if props >= 600:
                    break
                props += 1
                res = card.apply(cur, roles)
                if not res.applied or res.after_state is None:
                    continue
                if res.after_state == goal_s:
                    ubf_solved = True
                    break
                fp = res.after_state.fingerprint()
                if d + 1 < 4 and vis.get(fp, 5) > d + 1:
                    vis[fp] = d + 1
                    heapq.heappush(heap, (_state_diff_count(res.after_state, goal_s), d + 1, tie, res.after_state))
                    tie += 1
            if ubf_solved or props >= 600:
                break
        if ubf_solved:
            break
    assert ubf_solved is False and props == 600


def test_s4_bind_depth_geometry_probe_reproduces_through_depth_6() -> None:
    r1 = run_bind_depth_geometry_probe(depths=(1, 2, 3, 4, 5, 6), trials_per_depth=25, codebook_size=64, seed=101)
    r2 = run_bind_depth_geometry_probe(depths=(1, 2, 3, 4, 5, 6), trials_per_depth=25, codebook_size=64, seed=101)
    assert r1 == r2
    assert len(r1) == 6
    for row in r1:
        assert row.permuted_cleanup_accuracy >= 0.99
    assert r1[-1].depth == 6
    assert r1[-1].permuted_cleanup_accuracy - r1[-1].naive_superposition_accuracy >= 0.60


def test_s4_committed_manifests_saved_5seed_gate_and_fresh_isolated_run(tmp_path: Path) -> None:
    ok0, m0 = verify_committed_manifests(seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    ok1, m1 = verify_s1_committed_manifests(seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    ok2, m2 = verify_s2_committed_manifests(seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    ok3, m3 = verify_s3_committed_manifests(seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    ok4, m4 = verify_s4_committed_manifests(seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    assert ok0, f"S0 manifest drift: {m0}"
    assert ok1, f"S1 manifest drift: {m1}"
    assert ok2, f"S2 manifest drift: {m2}"
    assert ok3, f"S3 manifest drift: {m3}"
    assert ok4, f"S4 manifest drift: {m4}"

    summary_path = REPORTS_DIR / "s4_summary.json"
    report_path = REPORTS_DIR / "s4_research_report.md"
    csv_path = REPORTS_DIR / "raw_runs" / "s4_per_split.csv"
    assert summary_path.is_file() and report_path.is_file() and csv_path.is_file()

    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["s4_gate"]["passed"] is True
    assert len(saved["s4_gate"]["checks"]) >= 20
    assert saved["shard_keep_or_remove_decision"] == "KEEP_SHARD_AS_PROPOSER"
    assert saved["operator_library_size"] == 48
    assert saved["task_a"]["methods"]["shard_guided_proposer"]["mean_collateral_rate"]["max"] == 0.0
    assert saved["task_b"]["methods"]["shard_guided_planner"]["mean_collateral_rate"]["max"] == 0.0

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 105  # 5 seeds * (4 Task-A methods * 3 splits + 3 Task-B methods * 3 splits)
    for seed in DEFAULT_SEEDS:
        assert (REPORTS_DIR / "raw_runs" / f"s4_seed_{seed}.json").is_file()

    # Fresh isolated 2-seed end-to-end execution test
    # Note: with n=2 (df=1), Student-t critical value is 12.706 so wall-clock speedup CIs can clip to 0;
    # verify every non-timing check strictly AND verify per-seed min CPU speedup >= 2.0x on both seeds!
    fresh = run_s4_experiments(
        seeds=(101, 202),
        demos_per_action=DEFAULT_DEMOS_PER_ACTION,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        verify_prior_manifests=False,
    )
    df1_timing_keys = {
        "five_distinct_fixed_seeds_present",
        "task_a_shard_cpu_speedup_ge_2x_vs_unguided_and_random",
        "task_b_shard_cpu_speedup_ge_2x_vs_bfs_and_random",
        "shard_keep_decision_confirmed",
        "cpu_budget_within_limit",
    }
    for check_name, ok_val in fresh["s4_gate"]["checks"].items():
        if check_name not in df1_timing_keys:
            assert ok_val is True, f"Fresh isolated S4 check failed: {check_name}"
    assert fresh["task_a"]["comparisons"]["cpu_speedup_vs_unguided"]["min"] >= 2.0
    assert fresh["task_a"]["comparisons"]["cpu_speedup_vs_random"]["min"] >= 2.0
    assert fresh["task_b"]["comparisons"]["cpu_speedup_vs_bfs"]["min"] >= 2.0
    assert fresh["task_b"]["comparisons"]["cpu_speedup_vs_random_walk"]["min"] >= 2.0
