"""Comprehensive automated tests for Stage S0 simulator, splits, audits, baselines, and canary."""
from __future__ import annotations

import ast
import json
import random
from pathlib import Path

import pytest

from mirror_crystal.hdc_associative_engine import D, MASK
from research.cpu_learning.evaluation.baselines import (
    CurrentMirrorSolverBaseline,
    EpisodeLookupBaseline,
    NoOpIdentityBaseline,
    SHARDHDCRetrievalBaseline,
    SmallCPUNeuralReferenceBaseline,
    StructuralKNNBaseline,
)
from research.cpu_learning.evaluation.metrics import (
    TaskPrediction,
    aggregate_split_scores,
    compute_collateral_changes,
    score_operator_recovery,
    score_task,
    summarize_across_seeds,
)
from research.cpu_learning.evaluation.oracle import HiddenOracleBaseline
from research.cpu_learning.evaluation.run_experiments import (
    DATA_DIR,
    RAW_RUNS_DIR,
    REPORTS_DIR,
    run_s0_experiments,
    verify_committed_manifests,
)
from research.cpu_learning.evaluation.shard_canary import (
    SignedBitSlicedAccumulator,
    run_shard_arithmetic_canary,
)
from research.cpu_learning.simulator.generators import (
    DEFAULT_SEEDS,
    HELDOUT_POOL,
    TEMPORAL_POOL,
    TRAIN_POOL,
    UNSEEN_EVAL_SPLITS,
    build_action_invocation,
    build_split_manifests,
    generate_heldout_suite,
    generate_training_episodes,
    verify_pool_disjointness,
)
from research.cpu_learning.simulator.hidden_actions import (
    HIDDEN_ACTION_NAMES,
    apply_hidden_action,
    apply_hidden_sequence,
)
from research.cpu_learning.simulator.state_graph import (
    ActionInvocation,
    Entity,
    EvaluationCase,
    EvaluationTaskHiddenLabel,
    EvaluationTaskPublic,
    ObservationMask,
    SparseDelta,
    StateGraph,
    compute_sparse_delta,
)

PKG_ROOT = Path(__file__).resolve().parents[1]


def _referenced_import_targets(source_text: str) -> set[str]:
    """Collect all static and dynamic import targets including relative from-imports."""
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


def _has_forbidden_evaluator_import(targets: set[str]) -> bool:
    for tgt in targets:
        parts = tgt.split(".")
        if "hidden_actions" in parts or "oracle" in parts:
            return True
    return False


def test_hidden_actions_strict_import_isolation() -> None:
    """Non-evaluator modules MUST NEVER import hidden_actions or oracle (static or dynamic)."""
    # Self-test scanner against relative-import and dynamic-import bypass attempts
    assert _has_forbidden_evaluator_import(_referenced_import_targets("from . import hidden_actions"))
    assert _has_forbidden_evaluator_import(
        _referenced_import_targets("from research.cpu_learning.simulator import hidden_actions")
    )
    assert _has_forbidden_evaluator_import(
        _referenced_import_targets("import importlib; importlib.import_module('research.cpu_learning.simulator.hidden_actions')")
    )
    assert _has_forbidden_evaluator_import(
        _referenced_import_targets("__import__('research.cpu_learning.evaluation.oracle')")
    )

    evaluator_allowlist = {
        Path("simulator/hidden_actions.py"),
        Path("simulator/generators.py"),
        Path("evaluation/oracle.py"),
        Path("evaluation/metrics.py"),
        Path("evaluation/run_experiments.py"),
        Path("tests/test_s0_simulator_and_harness.py"),
    }
    scanned = 0
    for fpath in sorted(PKG_ROOT.rglob("*.py")):
        rel = fpath.relative_to(PKG_ROOT)
        if rel in evaluator_allowlist:
            continue
        scanned += 1
        targets = _referenced_import_targets(fpath.read_text(encoding="utf-8"))
        assert not _has_forbidden_evaluator_import(targets), (
            f"{rel} illegally referenced evaluator-only module: {sorted(targets)}"
        )
    assert scanned >= 4

    # Public evaluation tasks serialized to dict must never expose gold/hidden keys
    eps = generate_training_episodes(101)
    suite = generate_heldout_suite(101, eps)
    for split_cases in suite.values():
        pub_json = json.dumps(split_cases[0].public.to_dict())
        assert '"gold_' not in pub_json
        assert '"hidden_' not in pub_json


def test_state_graph_canonicalization_and_observation_mask_edge_cases() -> None:
    with pytest.raises(ValueError):
        Entity("", "object")
    with pytest.raises(ValueError):
        Entity("x", "")
    with pytest.raises(ValueError):
        ActionInvocation("move", (), b"abc", (("action", 2, 1),))
    with pytest.raises(ValueError):
        ActionInvocation("move", (), b"abc", (("action", 0, 10),))

    e1 = Entity("cup_1", "object", (("color", "red"), ("portable", True)))
    e2 = Entity("table_1", "surface", (("color", "blue"),))
    s_a = StateGraph.create([e1, e2], [("cup_1", "located_on", "table_1"), ("table_1", "accessible", "true")])
    s_b = StateGraph.create([e2, e1], [("table_1", "accessible", "true"), ("cup_1", "located_on", "table_1")])
    assert s_a == s_b
    assert s_a.fingerprint() == s_b.fingerprint()
    assert s_a.canonical_json() == s_b.canonical_json()

    # Partial observation mask must ignore unobserved entity/predicate changes
    s_after = s_a.with_changes(
        delete_facts=(("table_1", "accessible", "true"),),
        add_facts=(("table_1", "accessible", "false"),),
    )
    mask_cup_only = ObservationMask(
        observed_entities=frozenset({"cup_1"}),
        observed_predicates=frozenset({"located_on"}),
        fully_observed=False,
    )
    delta_masked = compute_sparse_delta(s_a, s_after, mask=mask_cup_only)
    assert delta_masked.is_noop
    assert delta_masked.preserved_fact_count == 1

    # Blind mask (fully_observed=False with empty sets) must observe NOTHING
    blind_mask = ObservationMask(
        observed_entities=frozenset(),
        observed_predicates=frozenset(),
        fully_observed=False,
    )
    assert not blind_mask.is_entity_observed("cup_1")
    assert not blind_mask.is_fact_observed(("cup_1", "located_on", "table_1"))
    assert compute_sparse_delta(s_a, s_after, mask=blind_mask).preserved_fact_count == 0

    # Predicate-only partial mask observes only the specified predicate across all entities
    pred_only_mask = ObservationMask(
        observed_entities=frozenset(),
        observed_predicates=frozenset({"located_on"}),
        fully_observed=False,
    )
    assert pred_only_mask.is_fact_observed(("cup_1", "located_on", "table_1"))
    assert not pred_only_mask.is_fact_observed(("table_1", "accessible", "true"))

    # Duplicate attribute keys, reserved 'type' key, duplicate entities, and duplicate roles must raise
    with pytest.raises(ValueError):
        Entity("cup_1", "object", (("color", "red"), ("color", "blue")))
    with pytest.raises(ValueError):
        Entity("cup_1", "object", (("type", "surface"),))
    with pytest.raises(ValueError):
        StateGraph.create([e1, Entity("cup_1", "object", (("color", "green"),))], [])
    with pytest.raises(ValueError):
        ActionInvocation("open", (("actor", "a1"), ("actor", "a2")), b"a1 open b1", (("action", 3, 7),))

    # SparseDelta attribute deletion must round-trip cleanly (apply_to(before) == after)
    s_attr_removed = StateGraph.create([e1.without_attr("color"), e2], s_a.facts)
    d_attr_rem = compute_sparse_delta(s_a, s_attr_removed)
    assert d_attr_rem.apply_to(s_a) == s_attr_removed
    assert len(SparseDelta(added_facts=(("a", "p", "b"), ("a", "p", "b"))).added_facts) == 1


def test_hidden_actions_preconditions_guards_and_non_commutation() -> None:
    entities = {
        "a1": Entity("a1", "agent"),
        "a2": Entity("a2", "agent"),
        "o1": Entity("o1", "object", (("color", "red"), ("portable", True))),
        "heavy": Entity("heavy", "object", (("color", "blue"), ("portable", False))),
        "s1": Entity("s1", "surface"),
        "s2": Entity("s2", "surface"),
        "s_bad": Entity("s_bad", "surface"),
        "box1": Entity("box1", "container", (("status", "closed"),)),
        "box_locked": Entity("box_locked", "container", (("status", "locked"),)),
        "distractor": Entity("distractor", "object", (("color", "green"), ("portable", True))),
    }
    facts = {
        ("o1", "located_on", "s1"),
        ("o1", "owned_by", "a1"),
        ("heavy", "located_on", "s1"),
        ("heavy", "owned_by", "a1"),
        ("distractor", "located_on", "s2"),
        ("distractor", "owned_by", "a2"),
        ("s1", "accessible", "true"),
        ("s2", "accessible", "true"),
        ("s_bad", "accessible", "false"),
        ("box1", "accessible", "true"),
        ("box1", "is_open", "false"),
        ("box1", "is_locked", "false"),
        ("box_locked", "accessible", "true"),
        ("box_locked", "is_open", "false"),
        ("box_locked", "is_locked", "true"),
    }
    w = StateGraph.create(entities, facts)

    # Blocked guards must produce zero state changes
    blocked_calls = [
        build_action_invocation("move", {"actor": "a1", "object": "heavy", "source": "s1", "destination": "s2"}),
        build_action_invocation("move", {"actor": "a1", "object": "o1", "source": "s1", "destination": "s_bad"}),
        build_action_invocation("move", {"actor": "a1", "object": "o1", "source": "s1", "destination": "box1"}),
        build_action_invocation("move", {"actor": "a2", "object": "o1", "source": "s1", "destination": "s2"}),
        build_action_invocation("move", {"actor": "a1", "object": "o1", "source": "s1", "destination": "s1"}),
        build_action_invocation("give", {"actor": "a1", "object": "o1", "recipient": "a1"}),
        build_action_invocation("give", {"actor": "a2", "object": "o1", "recipient": "a1"}),
        build_action_invocation("open", {"actor": "a1", "target": "box_locked"}),
        build_action_invocation("open", {"actor": "a1", "target": "s1"}),
        build_action_invocation("lock", {"actor": "a1", "target": "box_locked"}),
    ]
    for call in blocked_calls:
        res = apply_hidden_action(w, call)
        assert not res.applied
        assert res.after_state == w
        assert res.delta.is_noop

    # Unknown action is cleanly blocked
    unknown = ActionInvocation("teleport", (("actor", "a1"),), b"a1 teleport", (("action", 3, 11), ("actor", 0, 2)))
    res_u = apply_hidden_action(w, unknown)
    assert not res_u.applied and res_u.after_state == w

    # Non-commuting open vs lock
    act_open = build_action_invocation("open", {"actor": "a1", "target": "box1"})
    act_lock = build_action_invocation("lock", {"actor": "a1", "target": "box1"})
    s_ol, _, _, app_ol, _ = apply_hidden_sequence(w, (act_open, act_lock))
    s_lo, _, _, app_lo, _ = apply_hidden_sequence(w, (act_lock, act_open))
    assert app_ol == (True, False)
    assert app_lo == (True, False)
    assert s_ol != s_lo
    assert s_ol.has_fact("box1", "is_open", "true")
    assert s_lo.has_fact("box1", "is_locked", "true")
    # Distractor completely preserved
    assert s_ol.has_fact("distractor", "located_on", "s2")
    assert s_lo.has_fact("distractor", "owned_by", "a2")


def test_generator_disjointness_byte_spans_and_determinism() -> None:
    cert = verify_pool_disjointness()
    assert cert["disjoint_verified"] is True

    with pytest.raises(ValueError):
        generate_training_episodes(101, demos_per_action=2)
    with pytest.raises(ValueError):
        generate_training_episodes(101, demos_per_action=9)
    assert len(generate_training_episodes(101, demos_per_action=3)) == 12
    assert len(generate_training_episodes(101, demos_per_action=8)) == 32

    train_eps = generate_training_episodes(101, demos_per_action=6)
    suite = generate_heldout_suite(101, train_eps)

    train_roots = TRAIN_POOL.all_entity_roots()
    heldout_roots = HELDOUT_POOL.all_entity_roots()
    temporal_roots = TEMPORAL_POOL.all_entity_roots()

    for ep in train_eps:
        for eid, _ in ep.before_state.entities:
            root = eid.rsplit("_", 1)[0]
            assert root in train_roots
            assert root not in heldout_roots
            assert root not in temporal_roots
        # Verify every byte span slices the exact UTF-8 substring
        raw = ep.action.raw_instruction_bytes
        spans = ep.action.span_dict()
        act_s, act_e = spans["action"]
        assert raw[act_s:act_e].decode("utf-8") == ep.action.action_id
        for rk, rv in ep.action.roles:
            s_i, e_i = spans[rk]
            assert raw[s_i:e_i].decode("utf-8") == rv

    # Verify no role_badge confounder, open-container give positive, portable variation, and no context leakage
    actor_badges = {
        ep.before_state.get_entity(ep.action.role_dict()["actor"]).get("role_badge")
        for ep in train_eps
        if not ep.sparse_delta.is_noop
    }
    assert len(actor_badges) >= 2, f"Expected varied actor role_badge across training positives, got {actor_badges}"
    train_portables = {
        ent.get("portable")
        for ep in train_eps
        for _, ent in ep.before_state.entities
        if ent.entity_type == "object"
    }
    assert train_portables == {True, False}
    give_open_container_pos = [
        ep
        for ep in train_eps
        if ep.action.action_id == "give"
        and not ep.sparse_delta.is_noop
        and any(
            ep.before_state.get_entity(loc).entity_type == "container"
            for _, _, loc in ep.before_state.facts_for(ep.action.role_dict()["object"], "located_on")
        )
    ]
    assert len(give_open_container_pos) >= 1
    for ep in train_eps:
        assert "expected_transition_kind" not in dict(ep.context)

    # Adversarial substring-collision byte-span tests (whole-token word-boundary verification)
    for act_id, roles, tv in (
        ("lock", {"actor": "box_locked", "target": "box1"}, 0),
        ("open", {"actor": "a", "target": "box1"}, 1),
        ("move", {"actor": "s1", "object": "o", "source": "s1", "destination": "s2"}, 1),
    ):
        inv = build_action_invocation(act_id, roles, template_variant=tv)
        raw = inv.raw_instruction_bytes
        for key, (s_i, e_i) in inv.span_dict().items():
            expected_tok = act_id if key == "action" else roles[key]
            assert raw[s_i:e_i].decode("utf-8") == expected_tok
            assert s_i == 0 or raw[s_i - 1 : s_i] == b" "
            assert e_i == len(raw) or raw[e_i : e_i + 1] == b" "

    # Verify disjoint roots AND exact word-boundary UTF-8 byte spans across ALL held-out splits
    for split_name in UNSEEN_EVAL_SPLITS:
        expected_roots = temporal_roots if split_name == "heldout_temporal_frozen" else heldout_roots
        for case in suite[split_name]:
            for eid, _ in case.public.initial_state.entities:
                root = eid.rsplit("_", 1)[0]
                assert root in expected_roots
                assert root not in train_roots
            for act in case.public.actions:
                raw = act.raw_instruction_bytes
                for key, (s_i, e_i) in act.span_dict().items():
                    expected_tok = act.action_id if key == "action" else act.role_dict()[key]
                    assert raw[s_i:e_i].decode("utf-8") == expected_tok
                    assert s_i == 0 or raw[s_i - 1 : s_i] == b" "
                    assert e_i == len(raw) or raw[e_i : e_i + 1] == b" " 

    # Double-generation hash determinism
    tm1, hm1 = build_split_manifests((101, 202), demos_per_action=6)
    tm2, hm2 = build_split_manifests((101, 202), demos_per_action=6)
    assert tm1 == tm2
    assert hm1 == hm2


def test_retrieval_audit_abstention_and_collateral_metrics() -> None:
    train_eps = generate_training_episodes(101, demos_per_action=6)
    suite = generate_heldout_suite(101, train_eps)
    train_keys = frozenset(ep.state_action_key() for ep in train_eps)

    lookup = EpisodeLookupBaseline(seed=101)
    lookup.fit(train_eps)

    # Canary split: 100% retrieval hit, 0 learning-scored tasks
    canary_scores = [
        score_task(c, lookup.predict(c.public), train_keys, is_canary_split=True)
        for c in suite["canary_exact_memorization"]
    ]
    c_agg = aggregate_split_scores(canary_scores)
    assert c_agg["retrieval_hits"] == len(canary_scores)
    assert c_agg["learning_scored_tasks"] == 0
    assert c_agg["raw_exact_accuracy_including_retrieval"] == 1.0

    # Adversarial check: solver hides retrieval_hits=0 while accessing memory on exact train key
    canary_case = suite["canary_exact_memorization"][0]
    sneaky_pred = TaskPrediction(
        predicted_final_state=canary_case.hidden.gold_final_state,
        abstained=False,
        retrieval_hits=0,
        memory_accesses=1,
    )
    sneaky_score = score_task(canary_case, sneaky_pred, train_keys, is_canary_split=False)
    assert sneaky_score.is_retrieval_hit is True
    assert sneaky_score.exact_train_key_overlap is True
    assert sneaky_score.excluded_from_learning_score is True

    # Even a dishonest solver reporting memory_accesses=0 on an exact training key is excluded from learning score
    dishonest_pred = TaskPrediction(
        predicted_final_state=canary_case.hidden.gold_final_state,
        abstained=False,
        retrieval_hits=0,
        memory_accesses=0,
    )
    dishonest_score = score_task(canary_case, dishonest_pred, train_keys, is_canary_split=False)
    assert dishonest_score.exact_train_key_overlap is True
    assert dishonest_score.excluded_from_learning_score is True

    # Multi-step retrieval audit: step 0 unseen, step 1 lands on an exact training (state, action) key
    train_ep0 = train_eps[0]
    noop_act = build_action_invocation(
        "move",
        {
            "actor": train_ep0.action.role_dict()["actor"],
            "object": train_ep0.action.role_dict()["object"],
            "source": train_ep0.action.role_dict()["source"],
            "destination": train_ep0.action.role_dict()["source"],
        },
    )
    ms_final, ms_inters, ms_deltas, ms_app, ms_reasons = apply_hidden_sequence(
        train_ep0.before_state, (noop_act, train_ep0.action)
    )
    ms_case = EvaluationCase(
        public=EvaluationTaskPublic(
            task_id="multistep_exact_step1",
            split_id="synthetic",
            generator_id="test",
            generator_version="v1",
            seed=101,
            depth=2,
            initial_state=train_ep0.before_state,
            actions=(noop_act, train_ep0.action),
            observation_mask=ObservationMask(fully_observed=True),
        ),
        hidden=EvaluationTaskHiddenLabel(
            task_id="multistep_exact_step1",
            gold_final_state=ms_final,
            gold_intermediate_states=ms_inters,
            gold_step_deltas=ms_deltas,
            gold_step_applied=ms_app,
            gold_block_reasons=ms_reasons,
        ),
    )
    ms_pred = TaskPrediction(
        predicted_final_state=ms_final,
        abstained=False,
        retrieval_hits=0,
        memory_accesses=2,
        step_predictions=ms_inters,
    )
    ms_score = score_task(ms_case, ms_pred, train_keys, is_canary_split=False)
    assert ms_score.exact_train_key_overlap is True
    assert ms_score.is_retrieval_hit is True
    assert ms_score.excluded_from_learning_score is True

    # Abstention on a blocked guard-contrast task MUST NOT count as exact match
    blocked_case = next(
        c for c in suite["heldout_guard_contrasts"] if not c.hidden.gold_step_applied[0]
    )
    abstain_pred = TaskPrediction(predicted_final_state=None, abstained=True)
    abstain_score = score_task(blocked_case, abstain_pred, train_keys)
    assert abstain_score.exact_final_match is False
    assert abstain_score.collateral_changes == 0

    # Collateral error detection: deleting preserved fact + injecting spurious fact + editing untouched attr
    unseen_case = suite["heldout_unseen_names"][0]
    gold = unseen_case.hidden.gold_final_state
    preserved_fact = next(iter(unseen_case.public.initial_state.facts & gold.facts))
    corrupted = gold.with_changes(
        delete_facts=(preserved_fact,),
        add_facts=(("alien_entity_9", "located_on", "alien_table_9"),),
    )
    col_cnt, unrel_cnt, col_rate = compute_collateral_changes(
        unseen_case.public.initial_state, gold, corrupted, abstained=False
    )
    assert col_cnt == 2
    assert unrel_cnt >= 1
    assert col_rate > 0.0

    # Collateral audit must ALSO catch unrelated corruptions on touched entities, type mutations, & hallucinated entities
    e_cup = Entity("cup_1", "object", (("color", "red"), ("portable", True)))
    e_tbl = Entity("table_1", "surface", (("color", "blue"),))
    e_shf = Entity("shelf_1", "surface", (("color", "green"),))
    e_dsk = Entity("desk_1", "surface", (("color", "yellow"),))
    s_init = StateGraph.create(
        [e_cup, e_tbl, e_shf, e_dsk],
        [("cup_1", "located_on", "table_1"), ("cup_1", "owned_by", "alice_1"), ("table_1", "accessible", "true")],
    )
    s_gold = s_init.with_changes(
        delete_facts=(("cup_1", "located_on", "table_1"),),
        add_facts=(("cup_1", "located_on", "shelf_1"),),
    )
    # Primary effect error ONLY (wrong destination desk_1) -> 0 collateral
    s_primary_err = s_init.with_changes(
        delete_facts=(("cup_1", "located_on", "table_1"),),
        add_facts=(("cup_1", "located_on", "desk_1"),),
    )
    c_prim, _, _ = compute_collateral_changes(s_init, s_gold, s_primary_err, abstained=False)
    assert c_prim == 0
    # Touched-entity attr corruption (color + portable) + unrelated predicates (is_locked, owned_by)
    # + untouched entity_type mutation (table_1->agent) + hallucinated entity -> 6 collateral changes
    s_multi_corrupt = StateGraph.create(
        [
            Entity("cup_1", "object", (("color", "purple"), ("portable", False))),
            Entity("table_1", "agent", (("color", "blue"),)),
            e_shf,
            e_dsk,
            Entity("hallucinated_9", "object"),
        ],
        s_gold.facts | {("cup_1", "is_locked", "true"), ("cup_1", "owned_by", "evil_bot")},
    )
    c_multi, _, c_multi_rate = compute_collateral_changes(s_init, s_gold, s_multi_corrupt, abstained=False)
    assert c_multi == 6
    assert c_multi_rate > 0.0


def test_solver_freeze_and_operator_recovery_metrics() -> None:
    train_eps = generate_training_episodes(101, demos_per_action=6)
    solvers = [
        HiddenOracleBaseline(seed=101),
        NoOpIdentityBaseline(seed=101),
        EpisodeLookupBaseline(seed=101),
        StructuralKNNBaseline(seed=101),
        SHARDHDCRetrievalBaseline(seed=101),
        CurrentMirrorSolverBaseline(seed=101),
        SmallCPUNeuralReferenceBaseline(seed=101, epochs=3),
    ]
    for solver in solvers:
        solver.fit(train_eps)
        solver.freeze()
        with pytest.raises(RuntimeError):
            solver.fit(train_eps)

    full_oracle_schemas = HiddenOracleBaseline(seed=101).recovered_operator_schemas()
    oracle_rec = score_operator_recovery(full_oracle_schemas)
    assert oracle_rec["precision"] == 1.0
    assert oracle_rec["recall"] == 1.0
    assert oracle_rec["f1"] == 1.0

    # Omitting required_types / required_attr_guards MUST lower recall below 1.0
    stripped = {
        k: {ck: cv for ck, cv in v.items() if ck not in ("required_types", "required_attr_guards")}
        for k, v in full_oracle_schemas.items()
    }
    stripped_rec = score_operator_recovery(stripped)
    assert stripped_rec["precision"] == 1.0
    assert stripped_rec["recall"] < 1.0

    # Hallucinating an extra unknown operator MUST lower precision below 1.0
    with_extra = dict(full_oracle_schemas)
    with_extra["teleport"] = {"add_facts": [["?x", "located_on", "?y"]]}
    extra_rec = score_operator_recovery(with_extra)
    assert extra_rec["precision"] < 1.0
    assert extra_rec["recall"] == 1.0

    empty_rec = score_operator_recovery({})
    assert empty_rec["precision"] == 0.0
    assert empty_rec["recall"] == 0.0

    # Empty-fit baselines report 0 MDL bits cleanly
    for cls in (EpisodeLookupBaseline, StructuralKNNBaseline, SHARDHDCRetrievalBaseline, SmallCPUNeuralReferenceBaseline):
        sol = cls(seed=101)
        sol.fit([])
        assert sol.bit_budget(0, 0).compressed_model_mdl_bits == 0

    # Bounded [0, 1] CI clips high proportions cleanly while preserving exact ci95_half
    ci = summarize_across_seeds([1.0, 0.95, 1.0, 0.95, 1.0], bounds=(0.0, 1.0))
    assert ci["n"] == 5
    assert ci["ci95_half"] > 0.03
    assert 0.0 <= ci["ci95_low"] < ci["mean"] < ci["ci95_high"] <= 1.0


def test_signed_bit_sliced_accumulator_is_bit_exact() -> None:
    rng = random.Random(2026)
    acc = SignedBitSlicedAccumulator(bits=10)
    ref = [0] * D
    for _ in range(30):
        v = rng.getrandbits(D) & MASK
        if rng.random() < 0.5:
            acc.add_bipolar(v)
            for i in range(D):
                ref[i] += 1 if ((v >> i) & 1) else -1
        else:
            acc.sub_bipolar(v)
            for i in range(D):
                ref[i] -= 1 if ((v >> i) & 1) else -1
    expected_hv = 0
    for i in range(D):
        if ref[i] >= 0:
            expected_hv |= 1 << i
    assert acc.sign_hv() == expected_hv

    # Overflow safety: 4-bit accumulator auto-extends cleanly past +/-7 without sign wraparound
    acc_pos = SignedBitSlicedAccumulator(bits=4)
    for _ in range(50):
        acc_pos.add_bipolar(MASK)
    assert acc_pos.K > 4
    assert acc_pos.sign_hv() == MASK

    acc_neg = SignedBitSlicedAccumulator(bits=4)
    for _ in range(50):
        acc_neg.add_bipolar(0)
    assert acc_neg.K > 4
    assert acc_neg.sign_hv() == 0


def test_shard_arithmetic_canary_reproduces_and_confirms_modular_failure() -> None:
    res1 = run_shard_arithmetic_canary(seeds=DEFAULT_SEEDS)
    res2 = run_shard_arithmetic_canary(seeds=DEFAULT_SEEDS)
    assert res1["per_seed"] == res2["per_seed"]

    tasks = res1["summary_by_task"]
    assert tasks["carry"]["shard_bit_perceptron"]["mean"] >= 0.85
    assert tasks["comparison"]["shard_bit_perceptron"]["mean"] >= 0.75
    # Modular addition must remain a hard failure canary for pure similarity & bit-perceptron
    assert tasks["modular_addition"]["hdc_pure_similarity"]["mean"] < 0.25
    assert tasks["modular_addition"]["shard_bit_perceptron"]["mean"] < 0.25
    assert tasks["modular_addition"]["learned_readout_bits"] == 10 * D


def test_s0_saved_artifacts_and_tmp_end_to_end_run(tmp_path: Path) -> None:
    # Verify committed S0 artifacts pass all S0 gates
    assert (DATA_DIR / "preregistration.json").is_file()
    assert (DATA_DIR / "train_manifest.json").is_file()
    assert (DATA_DIR / "heldout_manifest.json").is_file()
    assert (REPORTS_DIR / "summary.json").is_file()
    assert (REPORTS_DIR / "research_report.md").is_file()
    assert (RAW_RUNS_DIR / "s0_per_split.csv").is_file()
    assert (RAW_RUNS_DIR / "shard_canary_5seed.json").is_file()
    for seed in DEFAULT_SEEDS:
        assert (RAW_RUNS_DIR / f"s0_seed_{seed}.json").is_file()

    # Committed manifests must match freshly generated manifests bit-for-bit
    manifests_ok, manifest_diffs = verify_committed_manifests()
    assert manifests_ok, f"Committed manifest drift detected: {manifest_diffs}"

    summary = json.loads((REPORTS_DIR / "summary.json").read_text(encoding="utf-8"))
    assert summary["s0_gate"]["passed"] is True
    assert len(summary["s0_gate"]["checks"]) >= 11
    assert summary["solvers"]["hand_coded_hidden_oracle"]["unseen_learning_exact_accuracy"]["mean"] == 1.0
    assert summary["solvers"]["episode_lookup"]["unseen_learning_exact_accuracy"]["mean"] == 0.0

    # End-to-end fresh 5-seed run (demos_per_action=4) into isolated tmp_path
    mini = run_s0_experiments(
        seeds=(101, 202, 303, 404, 505),
        demos_per_action=4,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        include_mirror_solver=False,
    )
    assert mini["s0_gate"]["passed"] is True
    assert (tmp_path / "reports" / "summary.json").is_file()
    assert (tmp_path / "reports" / "research_report.md").is_file()
