"""Automated Stage S3 tests: Controlled Single-Factor Contrasts, Repeated-Evidence Guard
Acquisition, Shadow Gating, Append-Only OperatorVersionStore & Rollback, Disjunctive Joint
Ablation Integrity, Novel Synthetic Domain Generalization, AST Isolation, and 5-Seed Gates.

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
    evaluate_solver_on_seed,
    make_blackbox_environment,
    verify_committed_manifests,
)
from research.cpu_learning.evaluation.run_s1_experiments import verify_s1_committed_manifests
from research.cpu_learning.evaluation.run_s2_experiments import verify_s2_committed_manifests
from research.cpu_learning.evaluation.run_s3_experiments import (
    S3_INACCESSIBLE_SOURCE_SPLIT,
    S2PassiveITLBaseline,
    audit_s3_ast_isolation_and_agnosticism,
    generate_inaccessible_source_contrast_split,
    run_s3_experiments,
    verify_s3_committed_manifests,
)
from research.cpu_learning.itl.contrast_harness import (
    ContrastRefinedITLLearner,
    UnderconstrainedContrastRepairLearner,
    acquire_guards_from_contrasts,
    build_probe_action,
    generate_domain_agnostic_contrast_probes,
    run_contrast_audit,
    strip_operator_guards,
)
from research.cpu_learning.itl.operator_card import OperatorCard, TypeTaxonomy
from research.cpu_learning.itl.operator_inducer import ITLOperatorLearner
from research.cpu_learning.itl.version_store import (
    OperatorVersionRecord,
    OperatorVersionStore,
    VersionStoreError,
    VersionStoreFrozenError,
)
from research.cpu_learning.ledger.episode_ledger import EpisodeLedger
from research.cpu_learning.simulator.generators import (
    DEFAULT_DEMOS_PER_ACTION,
    DEFAULT_SEEDS,
    HELDOUT_POOL,
    TRAIN_POOL,
    generate_heldout_suite,
    generate_training_episodes,
)
from research.cpu_learning.simulator.state_graph import (
    ActionInvocation,
    Entity,
    EvaluationTaskPublic,
    ObservationMask,
    SparseDelta,
    StateGraph,
    TrainingEpisode,
    compute_sparse_delta,
)

PKG_ROOT = Path(__file__).resolve().parents[1]


def _make_ep(
    ep_id: str,
    before: StateGraph,
    after: StateGraph,
    action: ActionInvocation,
    *,
    domain: str = "synthetic_s3_v1",
) -> TrainingEpisode:
    mask = ObservationMask(fully_observed=True)
    return TrainingEpisode(
        episode_id=ep_id,
        before_state=before,
        action=action,
        observation_mask=mask,
        after_state=after,
        sparse_delta=compute_sparse_delta(before, after, mask=mask),
        context=(("domain", domain),),
        observation_source="synthetic_s3_test",
        split_id="synthetic_train",
        generator_id="synth_s3_v1",
        generator_version="v1",
        seed=777,
        timestamp="2026-11-26T00:00:00Z",
    )


def test_s3_ast_isolation_zero_hardcoded_literals_and_blackbox_env() -> None:
    audit = audit_s3_ast_isolation_and_agnosticism()
    assert audit["passed"] is True, f"S3 AST audit failed: {audit}"
    assert audit["import_violations"] == []
    assert audit["hardcode_violations"] == []
    assert audit["scanned_itl_files"] >= 6
    assert audit["scanned_learner_files"] >= 11
    assert audit["scanned_non_evaluator_files"] >= 14

    # Independent stricter string-constant audit across all new S3 ITL modules
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
        "actor",
        "destination",
        "recipient",
        "target",
    }
    for rel_name in ("itl/version_store.py", "itl/contrast_harness.py"):
        src = (PKG_ROOT / rel_name).read_text(encoding="utf-8")
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
        assert hits == set(), f"{rel_name} contained forbidden S0 literals: {sorted(hits)}"

    # BlackBoxSimulatorEnvironment must expose strictly (StateGraph, SparseDelta), count queries, and freeze
    env = make_blackbox_environment()
    assert env.query_count == 0
    assert env.is_frozen is False
    train_eps = generate_training_episodes(101, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    ep0 = train_eps[0]
    after, delta = env.step(ep0.before_state, ep0.action)
    assert isinstance(after, StateGraph)
    assert isinstance(delta, SparseDelta)
    assert after == ep0.after_state
    assert delta == ep0.sparse_delta
    assert env.query_count == 1
    env.freeze()
    assert env.is_frozen is True
    with pytest.raises(RuntimeError):
        env.step(ep0.before_state, ep0.action)
    assert env.query_count == 1


def test_s3_operator_version_store_lifecycle_shadow_guard_rollback_and_serialization() -> None:
    def _dummy_card(aid: str, ver: int, parent: int | None = None, req: tuple[tuple[str, str, str], ...] = ()) -> OperatorCard:
        return OperatorCard(
            card_id=f"op_{aid}_v{ver}",
            action_id=aid,
            role_variables=("actor", "item"),
            required_types=(("actor", frozenset({"mage"})), ("item", frozenset({"orb"}))),
            required_facts=req,
            conditional_fact_guards=(),
            indirect_fact_guards=(),
            attribute_guards=(),
            inequality_guards=(),
            add_facts=(("?item", "lit", "true"),),
            delete_facts=(("?item", "lit", "false"),),
            attribute_edits=(),
            version=ver,
            parent_version=parent,
        )

    c1 = _dummy_card("ignite", 1, None, (("?item", "lit", "false"), ("?item", "dry", "true")))
    c2 = _dummy_card("ignite", 2, 1, (("?item", "lit", "false"),))
    c3_shadow = _dummy_card("ignite", 3, 2, (("?item", "lit", "false"), ("?actor", "focused", "true")))

    # Record validation negative cases
    with pytest.raises(VersionStoreError):
        OperatorVersionRecord(card=c1, status="invalid")  # type: ignore[arg-type]
    with pytest.raises(VersionStoreError):
        OperatorVersionRecord(card=c1, status="shadow", contrast_pass_rate=1.2)
    with pytest.raises(VersionStoreError):
        OperatorVersionRecord(card=c1, status="shadow", contrast_pass_rate=-0.01)

    store = OperatorVersionStore()
    assert store.next_version("ignite") == 1
    assert store.get_active("ignite") is None

    r1 = store.register_shadow(c1, evidence={"step": "init"}, contrast_pass_rate=0.95, notes="v1")
    assert r1.status == "shadow"
    assert store.get_active("ignite") is None

    # Monotonic version enforcement
    with pytest.raises(VersionStoreError):
        store.register_shadow(c1)

    p1 = store.promote("ignite", 1, contrast_pass_rate=0.98, notes="promote_v1")
    assert p1.status == "promoted"
    assert store.get_active("ignite") is not None
    assert store.get_active("ignite").card.version == 1

    # Promote v2 -> retires v1 without deleting it
    store.register_shadow(c2, evidence={"step": "ablate"}, contrast_pass_rate=1.0, notes="v2")
    p2 = store.promote("ignite", 2, notes="promote_v2")
    assert p2.status == "promoted"
    assert store.get_version("ignite", 1).status == "retired"
    assert store.get_version("ignite", 2).status == "promoted"
    assert len(store.get_history("ignite")) == 2

    # Register v3 as unpromoted shadow (e.g. 1-shot fluke)
    store.register_shadow(c3_shadow, evidence={"step": "1shot"}, contrast_pass_rate=0.5, notes="shadow_1shot")
    assert len(store.get_shadows("ignite")) == 1

    # ADVERSARIAL: rollback() MUST reject unpromoted shadow version (cannot bypass promote() gate!)
    with pytest.raises(VersionStoreError):
        store.rollback("ignite", 3, reason="illegal_shadow_bypass")
    assert store.get_active("ignite").card.version == 2
    assert store.get_version("ignite", 3).status == "shadow"

    # Rollback to retired v1 -> restores v1, retires v2, keeps v3 shadow, deletes NOTHING
    rb1 = store.rollback("ignite", 1, reason="revert_regression")
    assert rb1.card.version == 1
    assert rb1.status == "promoted"
    assert store.get_version("ignite", 2).status == "retired"
    assert store.get_version("ignite", 3).status == "shadow"
    assert len(store.get_history("ignite")) == 3

    # Unknown action / version errors
    with pytest.raises(VersionStoreError):
        store.promote("nonexistent", 1)
    with pytest.raises(VersionStoreError):
        store.promote("ignite", 99)
    with pytest.raises(VersionStoreError):
        store.rollback("nonexistent", 1)
    with pytest.raises(VersionStoreError):
        store.rollback("ignite", 99)

    # Freeze enforcement + bit-exact round-trip
    store.freeze()
    assert store.is_frozen is True
    with pytest.raises(VersionStoreFrozenError):
        store.register_shadow(_dummy_card("ignite", 4, 1))
    with pytest.raises(VersionStoreFrozenError):
        store.promote("ignite", 2)
    with pytest.raises(VersionStoreFrozenError):
        store.rollback("ignite", 2)

    raw_dict = store.to_dict()
    reloaded = OperatorVersionStore.from_dict(raw_dict)
    assert reloaded.canonical_json() == store.canonical_json()
    assert reloaded.is_frozen is True
    assert len(reloaded.transition_log()) == len(store.transition_log())

    # Corrupted serialization rejection (2 promoted versions or non-monotonic version)
    bad_two_promoted = store.to_dict()
    bad_two_promoted["histories"]["ignite"][1]["lifecycle_state"] = "promoted"
    with pytest.raises(VersionStoreError):
        OperatorVersionStore.from_dict(bad_two_promoted)

    bad_non_mono = store.to_dict()
    bad_non_mono["histories"]["ignite"][1]["card"]["version"] = 1
    with pytest.raises(VersionStoreError):
        OperatorVersionStore.from_dict(bad_non_mono)


def test_s3_disjunctive_ablation_joint_consistency_and_min_support_guard() -> None:
    """Adversarial test: when two clauses form a disjunctive dependency (C1 OR C2),
    joint ablation MUST NOT drop both AND AblationClauseReport + version_store evidence
    MUST strictly match the actually pruned clauses (never falsely mark a retained clause refuted)."""

    # --- Case 1: Fact OR Fact disjunction (glow=='on' OR hum=='on') ---
    class FactOrFactEnv:
        def step(
            self, state: StateGraph, action: ActionInvocation, mask: ObservationMask | None = None
        ) -> tuple[StateGraph, SparseDelta]:
            orb = action.role_dict()["orb"]
            if (state.has_fact(orb, "glow", "on") or state.has_fact(orb, "hum", "on")) and state.has_fact(
                orb, "charged", "false"
            ):
                after = state.with_changes(
                    delete_facts=((orb, "charged", "false"),),
                    add_facts=((orb, "charged", "true"),),
                )
                return after, compute_sparse_delta(state, after, mask=mask)
            return state.copy(), compute_sparse_delta(state, state, mask=mask)

    env_ff = FactOrFactEnv()
    eps_ff: list[TrainingEpisode] = []
    for i in range(4):
        m, o1, o2 = f"mage_{i}", f"orbA_{i}", f"orbB_{i}"
        w = StateGraph.create(
            [Entity(m, "mage"), Entity(o1, "orb"), Entity(o2, "orb")],
            [
                (o1, "glow", "on"),
                (o1, "hum", "on"),
                (o1, "charged", "false"),
                (o2, "glow", "off"),
                (o2, "hum", "off"),
                (o2, "charged", "true"),
            ],
        )
        act = build_probe_action("cast", {"mage": m, "orb": o1})
        after, _ = env_ff.step(w, act)
        eps_ff.append(_make_ep(f"ff_{i}", w, after, act))

    learner_ff = ContrastRefinedITLLearner(seed=1, env=env_ff, taxonomy=TypeTaxonomy())
    learner_ff.fit(eps_ff)
    card_ff = learner_ff.operators["cast"]
    # Exactly ONE of glow/hum is pruned, and the OTHER is retained!
    kept_ff = set(card_ff.required_facts)
    assert ("?orb", "charged", "false") in kept_ff
    assert (("?orb", "glow", "on") in kept_ff) ^ (("?orb", "hum", "on") in kept_ff)

    # Reports and version_store evidence MUST strictly agree with kept_ff!
    rep_by_clause = {r.clause: r for r in learner_ff.ablation_reports}
    retained_disj = ("?orb", "hum", "on") if ("?orb", "hum", "on") in kept_ff else ("?orb", "glow", "on")
    pruned_disj = ("?orb", "glow", "on") if retained_disj == ("?orb", "hum", "on") else ("?orb", "hum", "on")
    assert rep_by_clause[pruned_disj].refuted is True
    assert rep_by_clause[retained_disj].refuted is False
    assert rep_by_clause[retained_disj].reason == "retained_joint_disjunctive_dependency"
    assert rep_by_clause[retained_disj].contradiction_count >= 1
    assert learner_ff.version_store.get_active("cast").evidence["refuted_clauses"] == [list(pruned_disj)]

    # --- Case 2: Fact OR Attribute disjunction (glow=='on' OR warm==True) with probe_attribute_guards=True ---
    class FactOrAttrEnv:
        def step(
            self, state: StateGraph, action: ActionInvocation, mask: ObservationMask | None = None
        ) -> tuple[StateGraph, SparseDelta]:
            orb = action.role_dict()["orb"]
            eorb = state.get_entity(orb)
            if (
                eorb is not None
                and (state.has_fact(orb, "glow", "on") or eorb.get("warm") is True)
                and state.has_fact(orb, "charged", "false")
            ):
                after = state.with_changes(
                    delete_facts=((orb, "charged", "false"),),
                    add_facts=((orb, "charged", "true"),),
                )
                return after, compute_sparse_delta(state, after, mask=mask)
            return state.copy(), compute_sparse_delta(state, state, mask=mask)

    env_fa = FactOrAttrEnv()
    eps_fa: list[TrainingEpisode] = []
    for i in range(4):
        m, o1, o2, o3 = f"mage_{i}", f"orbA_{i}", f"orbB_{i}", f"orbC_{i}"
        # Include o3 with (glow='on', charged='false', warm=False) so warm is NOT functionally entailed by facts!
        w = StateGraph.create(
            [
                Entity(m, "mage"),
                Entity(o1, "orb", (("warm", True),)),
                Entity(o2, "orb", (("warm", False),)),
                Entity(o3, "orb", (("warm", False),)),
            ],
            [
                (o1, "glow", "on"),
                (o1, "charged", "false"),
                (o2, "glow", "off"),
                (o2, "charged", "true"),
                (o3, "glow", "on"),
                (o3, "charged", "false"),
            ],
        )
        act = build_probe_action("cast", {"mage": m, "orb": o1})
        after, _ = env_fa.step(w, act)
        eps_fa.append(_make_ep(f"fa_{i}", w, after, act))

    learner_fa = ContrastRefinedITLLearner(
        seed=2, env=env_fa, probe_attribute_guards=True, taxonomy=TypeTaxonomy()
    )
    learner_fa.fit(eps_fa)
    card_fa = learner_fa.operators["cast"]
    # glow fact pruned -> warm attribute guard MUST be retained by joint ablation!
    assert ("?orb", "glow", "on") not in card_fa.required_facts
    assert any(ag.attribute == "warm" and ag.expected_value is True for ag in card_fa.attribute_guards)
    attr_rep = next(r for r in learner_fa.ablation_reports if r.clause_kind == "attribute_guard")
    assert attr_rep.refuted is False
    assert attr_rep.reason == "retained_joint_disjunctive_dependency"

    # --- Case 3: Insufficient distinct worlds (< min_ablation_support) MUST NOT refute ---
    learner_low = ContrastRefinedITLLearner(
        seed=3, env=env_ff, min_ablation_support=2, taxonomy=TypeTaxonomy()
    )
    learner_low.fit(eps_ff[:1])  # Only 1 world!
    assert len(learner_low.version_store.get_history("cast")) == 1
    assert ("?orb", "glow", "on") in learner_low.operators["cast"].required_facts
    assert ("?orb", "hum", "on") in learner_low.operators["cast"].required_facts


def test_s3_one_shot_shadow_gate_duplicate_rejection_and_mixed_support_promotion() -> None:
    train_eps = generate_training_episodes(101, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    env = make_blackbox_environment()
    refined = ContrastRefinedITLLearner(seed=101, env=env)
    refined.fit(train_eps)

    aid = "move"
    stripped = strip_operator_guards(refined.operators[aid], version=1)
    pos_obs = refined._grouped_pos_obs[aid]
    corpus_stats = refined._corpus_stats

    vs = OperatorVersionStore()
    vs.register_shadow(stripped, notes="v1_stripped")
    vs.promote(aid, 1, notes="v1_active")

    probes = generate_domain_agnostic_contrast_probes(aid, stripped, pos_obs, corpus_stats)
    probe_env = make_blackbox_environment()
    failing: list[tuple[StateGraph, ActionInvocation]] = []
    for st, ac in probes:
        pred = stripped.apply(st, ac.role_dict())
        _after, delta = probe_env.step(st, ac)
        if pred.applied and delta.is_noop:
            failing.append((st, ac))
    assert len(failing) >= 4

    # 1. Single 1-shot failure -> stays shadow, never promotes
    rep1 = acquire_guards_from_contrasts(
        aid, vs, pos_obs, [failing[0]], corpus_stats, make_blackbox_environment(), min_guard_support=2
    )
    assert rep1.promoted is False
    assert rep1.shadow_version_registered == 2
    assert rep1.active_version_after == 1
    assert vs.get_active(aid).card.version == 1
    assert vs.get_version(aid, 2).status == "shadow"

    # 2. Adversarial duplicate inflation: repeating the exact same failing probe 3x MUST stay shadow
    rep_dup = acquire_guards_from_contrasts(
        aid,
        vs,
        pos_obs,
        [failing[0], failing[0], failing[0]],
        corpus_stats,
        make_blackbox_environment(),
        min_guard_support=2,
    )
    assert rep_dup.promoted is False
    assert rep_dup.independent_overgeneralizations == 1
    assert rep_dup.active_version_after == 1
    assert vs.get_active(aid).card.version == 1

    # 3. Attempting rollback to 1-shot shadow v2 MUST raise VersionStoreError
    with pytest.raises(VersionStoreError):
        vs.rollback(aid, 2, reason="cannot_activate_1shot_shadow")

    # 4. Mixed-support test: pick 2 failures for portable==False + 1 failure for dest.accessible==false
    portable_fails = [
        (st, ac)
        for st, ac in failing
        if st.get_entity(ac.role_dict()["object"]).get("portable") is False
    ]
    dest_inacc_fails = [
        (st, ac)
        for st, ac in failing
        if st.has_fact(ac.role_dict()["destination"], "accessible", "false")
        and st.get_entity(ac.role_dict()["object"]).get("portable") is True
    ]
    assert len(portable_fails) >= 2 and len(dest_inacc_fails) >= 1

    rep_mix = acquire_guards_from_contrasts(
        aid,
        vs,
        pos_obs,
        [portable_fails[0], portable_fails[1], dest_inacc_fails[0]],
        corpus_stats,
        make_blackbox_environment(),
        min_guard_support=2,
    )
    assert rep_mix.promoted is True
    promoted_card = vs.get_active(aid).card
    # Supported guard (portable==True) IS promoted; 1-shot guard (dest accessible==true) stays shadow-only!
    assert any(g.attribute == "portable" and g.expected_value is True for g in promoted_card.attribute_guards)
    assert ("?destination", "accessible", "true") not in promoted_card.required_facts
    assert any(g["clause"] == ["?destination", "accessible", "true"] for g in rep_mix.shadow_only_guards)
    # Prior v1 retired intact, v2 & v3 shadows retained intact (zero deletion)
    assert vs.get_version(aid, 1).status == "retired"
    assert vs.get_version(aid, 2).status == "shadow"
    assert vs.get_version(aid, 3).status == "shadow"
    assert len(vs.get_history(aid)) == 4


def test_s3_novel_synthetic_arcane_domain_ablation_repair_and_contrast_audit() -> None:
    """Prove Stage S3 ablation, guard repair, and ContrastAuditReport generalize with ZERO
    hardcoded literals to a 100% novel synthetic domain with unseen actions, types, and predicates."""

    class ArcaneEnv:
        def __init__(self) -> None:
            self.queries = 0

        def step(
            self, state: StateGraph, action: ActionInvocation, mask: ObservationMask | None = None
        ) -> tuple[StateGraph, SparseDelta]:
            self.queries += 1
            r = action.role_dict()
            aid = action.action_id
            if aid == "channel":
                w, rel, org, dst = r.get("caster"), r.get("relic"), r.get("origin"), r.get("dest")
                ew, er, eo, ed = (
                    state.get_entity(w or ""),
                    state.get_entity(rel or ""),
                    state.get_entity(org or ""),
                    state.get_entity(dst or ""),
                )
                if not (ew and er and eo and ed):
                    return state.copy(), compute_sparse_delta(state, state, mask=mask)
                if ew.entity_type != "wizard" or er.entity_type != "rune":
                    return state.copy(), compute_sparse_delta(state, state, mask=mask)
                if eo.entity_type not in ("altar", "sanctum") or ed.entity_type not in ("altar", "sanctum"):
                    return state.copy(), compute_sparse_delta(state, state, mask=mask)
                if org == dst or er.get("charged") is not True:
                    return state.copy(), compute_sparse_delta(state, state, mask=mask)
                if not (state.has_fact(rel, "attuned_to", w) and state.has_fact(rel, "anchored_at", org)):
                    return state.copy(), compute_sparse_delta(state, state, mask=mask)
                # Notice: origin.reachable is NOT required; only dest.reachable == true!
                if not state.has_fact(dst, "reachable", "true"):
                    return state.copy(), compute_sparse_delta(state, state, mask=mask)
                if eo.entity_type == "sanctum" and not (
                    state.has_fact(org, "gate_open", "true") and state.has_fact(org, "ward_active", "false")
                ):
                    return state.copy(), compute_sparse_delta(state, state, mask=mask)
                if ed.entity_type == "sanctum" and not (
                    state.has_fact(dst, "gate_open", "true") and state.has_fact(dst, "ward_active", "false")
                ):
                    return state.copy(), compute_sparse_delta(state, state, mask=mask)
                after = state.with_changes(
                    delete_facts=((rel, "anchored_at", org),),
                    add_facts=((rel, "anchored_at", dst),),
                )
                return after, compute_sparse_delta(state, after, mask=mask)

            if aid == "bestow":
                w, rel, h = r.get("caster"), r.get("relic"), r.get("heir")
                ew, er, eh = (
                    state.get_entity(w or ""),
                    state.get_entity(rel or ""),
                    state.get_entity(h or ""),
                )
                if not (ew and er and eh) or ew.entity_type != "wizard" or eh.entity_type != "wizard" or er.entity_type != "rune":
                    return state.copy(), compute_sparse_delta(state, state, mask=mask)
                if w == h or not state.has_fact(rel, "attuned_to", w):
                    return state.copy(), compute_sparse_delta(state, state, mask=mask)
                for _s, _p, loc in state.facts_for(rel, "anchored_at"):
                    eloc = state.get_entity(loc)
                    if eloc and eloc.entity_type == "sanctum":
                        if not (
                            state.has_fact(loc, "gate_open", "true")
                            and state.has_fact(loc, "ward_active", "false")
                        ):
                            return state.copy(), compute_sparse_delta(state, state, mask=mask)
                after = state.with_changes(
                    delete_facts=((rel, "attuned_to", w),),
                    add_facts=((rel, "attuned_to", h),),
                )
                return after, compute_sparse_delta(state, after, mask=mask)

            if aid == "unseal":
                w, sc = r.get("caster"), r.get("shrine_target")
                ew, esc = state.get_entity(w or ""), state.get_entity(sc or "")
                if not (ew and esc) or ew.entity_type != "wizard" or esc.entity_type != "sanctum":
                    return state.copy(), compute_sparse_delta(state, state, mask=mask)
                if not (
                    state.has_fact(sc, "reachable", "true")
                    and state.has_fact(sc, "gate_open", "false")
                    and state.has_fact(sc, "ward_active", "false")
                ):
                    return state.copy(), compute_sparse_delta(state, state, mask=mask)
                after = state.with_changes(
                    delete_facts=((sc, "gate_open", "false"),),
                    add_facts=((sc, "gate_open", "true"),),
                    attribute_edits=((sc, "seal_label", "open"),),
                )
                return after, compute_sparse_delta(state, after, mask=mask)

            if aid == "ward":
                w, sc = r.get("caster"), r.get("shrine_target")
                ew, esc = state.get_entity(w or ""), state.get_entity(sc or "")
                if not (ew and esc) or ew.entity_type != "wizard" or esc.entity_type != "sanctum":
                    return state.copy(), compute_sparse_delta(state, state, mask=mask)
                if not (
                    state.has_fact(sc, "reachable", "true")
                    and state.has_fact(sc, "gate_open", "false")
                    and state.has_fact(sc, "ward_active", "false")
                ):
                    return state.copy(), compute_sparse_delta(state, state, mask=mask)
                after = state.with_changes(
                    delete_facts=((sc, "ward_active", "false"),),
                    add_facts=((sc, "ward_active", "true"),),
                    attribute_edits=((sc, "seal_label", "warded"),),
                )
                return after, compute_sparse_delta(state, after, mask=mask)

            return state.copy(), compute_sparse_delta(state, state, mask=mask)

    def make_world(
        sfx: str,
        *,
        charged0: bool = True,
        charged1: bool = False,
        sc_gate: str = "true",
        sc_ward: str = "false",
        al0_reach: str = "true",
        al1_reach: str = "true",
        r0_at_sc: bool = False,
        aura: str = "gold",
        heir_aura: str = "silver",
    ) -> tuple[StateGraph, dict[str, str]]:
        w0, w1 = f"mageA_{sfx}", f"mageB_{sfx}"
        r0, r1 = f"sigil0_{sfx}", f"sigil1_{sfx}"
        al0, al1 = f"altar0_{sfx}", f"altar1_{sfx}"
        sc0 = f"sanctum0_{sfx}"
        sl = "warded" if sc_ward == "true" else ("open" if sc_gate == "true" else "closed")
        ents = [
            Entity(w0, "wizard", (("aura", aura),)),
            Entity(w1, "wizard", (("aura", heir_aura),)),
            Entity(r0, "rune", (("charged", charged0),)),
            Entity(r1, "rune", (("charged", charged1),)),
            Entity(al0, "altar", (("stone", aura),)),
            Entity(al1, "altar", (("stone", "obsidian"),)),
            Entity(sc0, "sanctum", (("stone", aura), ("seal_label", sl))),
        ]
        loc0 = sc0 if r0_at_sc else al0
        facts = [
            (r0, "attuned_to", w0),
            (r0, "anchored_at", loc0),
            (r1, "attuned_to", w1),
            (r1, "anchored_at", al1),
            (al0, "reachable", al0_reach),
            (al1, "reachable", al1_reach),
            (sc0, "reachable", "true"),
            (sc0, "gate_open", sc_gate),
            (sc0, "ward_active", sc_ward),
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

    setup_env = ArcaneEnv()
    train_eps: list[TrainingEpisode] = []
    colors = ("gold", "ruby", "jade", "cobalt")

    def add_ep(eid: str, w: StateGraph, aid: str, roles: dict[str, str]) -> None:
        act = build_probe_action(aid, roles)
        after, _ = setup_env.step(w, act)
        train_eps.append(_make_ep(eid, w, after, act, domain="synthetic_arcane_s3"))

    for i in range(4):
        w, ids = make_world(
            f"ch{i}", charged1=(i % 2 == 0), sc_gate="true", sc_ward="false", r0_at_sc=(i == 3), aura=colors[i]
        )
        src = ids["sc0"] if i == 3 else ids["al0"]
        dst = ids["sc0"] if i == 2 else ids["al1"]
        add_ep(f"ch_p{i}", w, "channel", {"caster": ids["w0"], "relic": ids["r0"], "origin": src, "dest": dst})
    w, ids = make_world("chn0", al1_reach="false")
    add_ep("ch_n0", w, "channel", {"caster": ids["w0"], "relic": ids["r0"], "origin": ids["al0"], "dest": ids["al1"]})
    w, ids = make_world("chn1", sc_gate="false")
    add_ep("ch_n1", w, "channel", {"caster": ids["w0"], "relic": ids["r0"], "origin": ids["al0"], "dest": ids["sc0"]})
    w, ids = make_world("chn2", charged0=False)
    add_ep("ch_n2", w, "channel", {"caster": ids["w0"], "relic": ids["r0"], "origin": ids["al0"], "dest": ids["al1"]})

    # Notice: bestow varies charged0=(i % 2 == 0) in positives, AND also test incidental attr ablation separately
    for i in range(4):
        w, ids = make_world(
            f"bs{i}", charged0=(i % 2 == 0), sc_gate="true", sc_ward="false", r0_at_sc=(i == 3), aura=colors[i]
        )
        add_ep(f"bs_p{i}", w, "bestow", {"caster": ids["w0"], "relic": ids["r0"], "heir": ids["w1"]})
    w, ids = make_world("bsn0")
    add_ep("bs_n0", w, "bestow", {"caster": ids["w1"], "relic": ids["r0"], "heir": ids["w0"]})
    w, ids = make_world("bsn1", sc_gate="false", sc_ward="false", r0_at_sc=True)
    add_ep("bs_n1", w, "bestow", {"caster": ids["w0"], "relic": ids["r0"], "heir": ids["w1"]})

    for i in range(4):
        w, ids = make_world(f"us{i}", sc_gate="false", sc_ward="false", aura=colors[i])
        add_ep(f"us_p{i}", w, "unseal", {"caster": ids["w0"], "shrine_target": ids["sc0"]})
    w, ids = make_world("usn0", sc_gate="false", sc_ward="true")
    add_ep("us_n0", w, "unseal", {"caster": ids["w0"], "shrine_target": ids["sc0"]})
    w, ids = make_world("usn1", sc_gate="true", sc_ward="false")
    add_ep("us_n1", w, "unseal", {"caster": ids["w0"], "shrine_target": ids["sc0"]})

    for i in range(4):
        w, ids = make_world(f"wd{i}", sc_gate="false", sc_ward="false", aura=colors[i])
        add_ep(f"wd_p{i}", w, "ward", {"caster": ids["w0"], "shrine_target": ids["sc0"]})
    w, ids = make_world("wdn0", sc_gate="true", sc_ward="false")
    add_ep("wd_n0", w, "ward", {"caster": ids["w0"], "shrine_target": ids["sc0"]})
    w, ids = make_world("wdn1", sc_gate="false", sc_ward="true")
    add_ep("wd_n1", w, "ward", {"caster": ids["w0"], "shrine_target": ids["sc0"]})

    tax = TypeTaxonomy.from_mapping({"shrine": frozenset({"altar", "sanctum"})})

    # 1. S2 passive over-constrains channel with incidental (?origin, reachable, true)
    passive = ITLOperatorLearner(seed=7, taxonomy=tax)
    passive.fit(train_eps)
    passive.freeze()
    assert ("?origin", "reachable", "true") in passive.operators["channel"].required_facts

    # 2. S3 ContrastRefinedITLLearner actively refutes (?origin, reachable, true) & promotes v2
    env_ref = ArcaneEnv()
    refined = ContrastRefinedITLLearner(seed=7, env=env_ref, taxonomy=tax)
    refined.fit(train_eps)
    refined.freeze()
    queries_after_fit = env_ref.queries
    assert queries_after_fit > 0
    assert ("?origin", "reachable", "true") not in refined.operators["channel"].required_facts
    assert ("?dest", "reachable", "true") in refined.operators["channel"].required_facts
    assert [(r.card.version, r.status) for r in refined.version_store.get_history("channel")] == [
        (1, "retired"),
        (2, "promoted"),
    ]

    # Unseen unreachable-origin channel task: passive fails (predicts no-op), S3 refined matches gold!
    w_unr, ids_u = make_world("UNSEEN_UNR", al0_reach="false", al1_reach="true", aura="emerald", heir_aura="violet")
    act_unr = build_probe_action(
        "channel",
        {"caster": ids_u["w0"], "relic": ids_u["r0"], "origin": ids_u["al0"], "dest": ids_u["al1"]},
    )
    gold_unr, _ = ArcaneEnv().step(w_unr, act_unr)
    task_unr = EvaluationTaskPublic(
        "synth_unr", "synth_eval", "synth", "v1", 7, 1, w_unr, (act_unr,), ObservationMask(fully_observed=True)
    )
    pred_p = passive.predict(task_unr)
    pred_r = refined.predict(task_unr)
    assert pred_p.predicted_final_state == w_unr  # Passive wrongly blocked
    assert pred_r.abstained is False and pred_r.predicted_final_state == gold_unr
    assert env_ref.queries == queries_after_fit  # Zero test-time env queries!

    # 3. UnderconstrainedContrastRepairLearner strips & repairs ALL 4 novel operators to exact schemas!
    rep = UnderconstrainedContrastRepairLearner(seed=7, env=ArcaneEnv(), taxonomy=tax)
    rep.fit(train_eps)
    rep.freeze()
    for aid in sorted(refined.operators):
        assert rep.operators[aid].to_schema_dict(tax) == refined.operators[aid].to_schema_dict(tax)

    # 4. Independent ContrastAuditReport on novel synthetic domain
    audit = run_contrast_audit(refined.operators, refined._grouped_pos_obs, ArcaneEnv())
    assert audit.all_checks_passed is True
    assert audit.role_swap_probes.exact_final_accuracy == 1.0
    assert audit.commutator_order_probes.exact_final_accuracy == 1.0
    assert audit.commutator_order_probes.non_commuting_pairs_detected >= 2
    assert audit.repeated_idempotent_probes.exact_final_accuracy == 1.0
    for d in ("depth_1", "depth_2", "depth_4", "depth_6"):
        assert audit.depth_stratified_verification[d].exact_final_accuracy == 1.0
        assert audit.depth_stratified_verification[d].intermediate_step_accuracy == 1.0


def test_s3_freeze_zero_post_freeze_env_queries_disk_ledger_rollback_and_edge_cases(tmp_path: Path) -> None:
    train_eps = generate_training_episodes(202, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    suite = generate_heldout_suite(202, train_eps)
    inacc_cases = generate_inaccessible_source_contrast_split(202)

    db_path = tmp_path / "s3_train_ledger.sqlite3"
    with EpisodeLedger(db_path) as ledger:
        for ep in train_eps:
            ledger.ingest_training_episode(ep, use_typed_action=True)
        ledger.freeze()

        env = make_blackbox_environment()
        learner = ContrastRefinedITLLearner(seed=202, env=env)
        learner.fit_from_ledger(ledger, split_id="train")

        # Verify live rollback BEFORE freeze reverts & restores inaccessible-source behavior
        sample_inacc = inacc_cases[0]
        assert learner.predict(sample_inacc.public).predicted_final_state == sample_inacc.hidden.gold_final_state
        learner.rollback_operator("move", 1, reason="verify_revert")
        assert learner.predict(sample_inacc.public).predicted_final_state == sample_inacc.public.initial_state
        learner.rollback_operator("move", 2, reason="verify_restore")
        assert learner.predict(sample_inacc.public).predicted_final_state == sample_inacc.hidden.gold_final_state

        # Grab a guarded_env handle BEFORE freeze to test post-freeze trap
        stale_guarded = learner._guarded_env()

        learner.freeze()
        fit_queries = env.query_count
        assert fit_queries == learner.fit_env_queries and fit_queries > 0
        ledger.reset_audit_log()

        # Post-freeze mutation guards MUST all raise
        with pytest.raises(RuntimeError):
            learner.fit(train_eps)
        with pytest.raises(RuntimeError):
            learner.fit_from_ledger(ledger)
        with pytest.raises(RuntimeError):
            learner.acquire_guards("move", [])
        with pytest.raises(RuntimeError):
            learner.rollback_operator("move", 1)
        with pytest.raises(VersionStoreFrozenError):
            learner.version_store.rollback("move", 1)
        with pytest.raises(RuntimeError):
            stale_guarded.step(sample_inacc.public.initial_state, sample_inacc.public.actions[0])
        assert env.query_count == fit_queries  # Stale guarded env was blocked BEFORE touching env!

        # Test-time predictions on depth-6 and inaccessible-source split: 0 env queries, 0 ledger reads!
        for case in list(suite["heldout_comp_depth6"][:5]) + list(inacc_cases[:5]):
            pred = learner.predict(case.public)
            assert pred.abstained is False
            assert pred.retrieval_hits == 0
            assert pred.memory_accesses == 0
            assert pred.predicted_final_state == case.hidden.gold_final_state
            assert pred.step_predictions == case.hidden.gold_intermediate_states

        assert env.query_count == fit_queries
        assert ledger.audit_summary()["total_accesses"] == 0

        # Partial ObservationMask hiding a required fact MUST abstain honestly
        open_case = next(c for c in suite["heldout_unseen_names"] if c.public.actions[0].action_id == "open")
        partial_mask = ObservationMask(
            observed_entities=frozenset(eid for eid, _ in open_case.public.initial_state.entities),
            observed_predicates=frozenset({"is_open", "accessible"}),  # hides is_locked
            fully_observed=False,
        )
        masked_task = EvaluationTaskPublic(
            "masked_s3",
            "partial_probe",
            "test",
            "v1",
            202,
            1,
            open_case.public.initial_state,
            open_case.public.actions,
            partial_mask,
        )
        assert learner.predict(masked_task).abstained is True

    # Empty-fit edge cases on both S3 learners
    for cls in (ContrastRefinedITLLearner, UnderconstrainedContrastRepairLearner):
        empty_env = make_blackbox_environment()
        empty_solver = cls(seed=202, env=empty_env)
        empty_solver.fit([])
        empty_solver.freeze()
        assert empty_solver.operators == {}
        assert empty_solver.fit_env_queries == 0
        assert empty_env.query_count == 0
        assert empty_solver.bit_budget(0, 0).compressed_model_mdl_bits == 0
        assert empty_solver.predict(suite["heldout_unseen_names"][0].public).abstained is True


def test_s3_inaccessible_source_split_integrity_and_seed101_head_to_head() -> None:
    train_eps = generate_training_episodes(101, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    train_keys = frozenset(ep.state_action_key() for ep in train_eps)
    train_roots = TRAIN_POOL.all_entity_roots()
    heldout_roots = HELDOUT_POOL.all_entity_roots()

    inacc = generate_inaccessible_source_contrast_split(101)
    assert len(inacc) == 16
    src_cont_count = 0
    for case in inacc:
        # Strict disjointness & zero train-key overlap
        for eid, _ in case.public.initial_state.entities:
            root = eid.rsplit("_", 1)[0]
            assert root in heldout_roots and root not in train_roots
        act = case.public.actions[0]
        assert (case.public.initial_state.fingerprint(), act.action_id, act.roles) not in train_keys
        # Word-boundary UTF-8 byte spans
        raw = act.raw_instruction_bytes
        for key, (s_i, e_i) in act.span_dict().items():
            tok = act.action_id if key == "action" else act.role_dict()[key]
            assert raw[s_i:e_i].decode("utf-8") == tok
            assert s_i == 0 or raw[s_i - 1 : s_i] == b" "
            assert e_i == len(raw) or raw[e_i : e_i + 1] == b" "
        # Source is truly inaccessible and gold applied
        src_id = act.role_dict()["source"]
        assert case.public.initial_state.has_fact(src_id, "accessible", "false")
        assert case.hidden.gold_step_applied == (True,)
        if case.public.initial_state.get_entity(src_id).entity_type == "container":
            src_cont_count += 1
    assert src_cont_count == 8

    # Head-to-head on seed 101 (including raw UTF-8 E2E mode)
    s3_res = evaluate_solver_on_seed(
        ContrastRefinedITLLearner(seed=101, env=make_blackbox_environment()), seed=101
    )
    s3_e2e_res = evaluate_solver_on_seed(
        ContrastRefinedITLLearner(seed=101, env=make_blackbox_environment(), raw_utf8_parse_mode=True),
        seed=101,
    )
    s2_res = evaluate_solver_on_seed(S2PassiveITLBaseline(seed=101), seed=101)
    rep_res = evaluate_solver_on_seed(
        UnderconstrainedContrastRepairLearner(seed=101, env=make_blackbox_environment()), seed=101
    )

    assert s3_res["operator_recovery"]["precision"] == 1.0
    assert s3_res["operator_recovery"]["recall"] == 1.0
    assert s3_res["operator_recovery"]["precision"] > s2_res["operator_recovery"]["precision"]
    assert s3_e2e_res["unseen_overall"]["learning_exact_accuracy"] == 1.0
    assert s3_e2e_res["operator_recovery"]["precision"] == 1.0
    assert rep_res["operator_recovery"]["precision"] == 1.0
    assert rep_res["operator_recovery"]["recall"] == 1.0


def test_s3_committed_manifests_saved_5seed_gate_and_fresh_isolated_run(tmp_path: Path) -> None:
    ok0, m0 = verify_committed_manifests(seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    ok1, m1 = verify_s1_committed_manifests(seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    ok2, m2 = verify_s2_committed_manifests(seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    ok3, m3 = verify_s3_committed_manifests(seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION)
    assert ok0, f"S0 manifest drift: {m0}"
    assert ok1, f"S1 manifest drift: {m1}"
    assert ok2, f"S2 manifest drift: {m2}"
    assert ok3, f"S3 manifest drift: {m3}"

    summary_path = REPORTS_DIR / "s3_summary.json"
    report_path = REPORTS_DIR / "s3_research_report.md"
    csv_path = REPORTS_DIR / "raw_runs" / "s3_per_split.csv"
    assert summary_path.is_file() and report_path.is_file() and csv_path.is_file()

    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["s3_gate"]["passed"] is True
    assert len(saved["s3_gate"]["checks"]) >= 25
    assert saved["s3_audit_summary"]["max_test_time_env_queries"] == 0
    assert saved["s3_audit_summary"]["max_post_freeze_env_queries"] == 0
    assert saved["s3_audit_summary"]["one_shot_rejection_seeds_passed"] == 5
    assert saved["s3_audit_summary"]["version_store_audit_seeds_passed"] == 5
    assert saved["s3_audit_summary"]["contrast_audit_seeds_passed"] == 5

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 275  # 5 seeds * 5 solvers * (10 unseen + 1 canary)
    for seed in DEFAULT_SEEDS:
        raw_p = REPORTS_DIR / "raw_runs" / f"s3_seed_{seed}.json"
        assert raw_p.is_file()

    # Fresh isolated 2-seed end-to-end execution test
    fresh = run_s3_experiments(
        seeds=(101, 202),
        demos_per_action=DEFAULT_DEMOS_PER_ACTION,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        verify_prior_manifests=False,
    )
    for check_name, ok_val in fresh["s3_gate"]["checks"].items():
        if check_name != "five_distinct_fixed_seeds_present":
            assert ok_val is True, f"Fresh isolated S3 check failed: {check_name}"
