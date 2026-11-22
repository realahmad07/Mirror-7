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
    ObservationMask,
    StateGraph,
    compute_sparse_delta,
)

PKG_ROOT = Path(__file__).resolve().parents[1]


def _imported_module_names(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.add(node.module)
    return found


def test_hidden_actions_strict_import_isolation() -> None:
    """Non-evaluator modules MUST NEVER import hidden_actions or oracle."""
    learner_side_files = (
        PKG_ROOT / "simulator" / "__init__.py",
        PKG_ROOT / "simulator" / "state_graph.py",
        PKG_ROOT / "evaluation" / "baselines.py",
        PKG_ROOT / "evaluation" / "shard_canary.py",
    )
    for fpath in learner_side_files:
        mods = _imported_module_names(fpath)
        assert not any("hidden_actions" in m for m in mods), f"{fpath} imported hidden_actions!"
        assert not any(m.endswith(".oracle") or m == "oracle" for m in mods), f"{fpath} imported oracle!"

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

    for split_name in UNSEEN_EVAL_SPLITS:
        expected_roots = temporal_roots if split_name == "heldout_temporal_frozen" else heldout_roots
        for case in suite[split_name]:
            for eid, _ in case.public.initial_state.entities:
                root = eid.rsplit("_", 1)[0]
                assert root in expected_roots
                assert root not in train_roots

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
    assert sneaky_score.excluded_from_learning_score is True

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

    oracle_rec = score_operator_recovery(HiddenOracleBaseline(seed=101).recovered_operator_schemas())
    assert oracle_rec["precision"] == 1.0
    assert oracle_rec["recall"] == 1.0
    assert oracle_rec["f1"] == 1.0

    empty_rec = score_operator_recovery({})
    assert empty_rec["precision"] == 0.0
    assert empty_rec["recall"] == 0.0

    ci = summarize_across_seeds([0.8, 0.9, 0.85, 0.95, 0.9])
    assert ci["n"] == 5
    assert ci["ci95_low"] < ci["mean"] < ci["ci95_high"]


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

    summary = json.loads((REPORTS_DIR / "summary.json").read_text(encoding="utf-8"))
    assert summary["s0_gate"]["passed"] is True
    assert summary["solvers"]["hand_coded_hidden_oracle"]["unseen_learning_exact_accuracy"]["mean"] == 1.0
    assert summary["solvers"]["episode_lookup"]["unseen_learning_exact_accuracy"]["mean"] == 0.0

    # End-to-end fresh run into isolated tmp_path (fast 1-seed check)
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
