"""Stage S5 RDC Refutation: Mechanism 1 (Dream-Refutation) & Mechanism 2 (Max-Disagreement Active Querying).

STRICT ISOLATION & CRITICAL SAFEGUARDS (PDF p. 8):
- A system-generated "dream" is ONLY a proposed counterfactual test — NEVER ground truth!
  Every dream MUST be evaluated by an independent black-box ``TransitionEnvironment``
  (``self_certified_dreams == 0`` ALWAYS).
- Contains ZERO hardcoded domain action, predicate, attribute, or entity-type literals.
- NEVER imports ``hidden_actions``, ``oracle``, or ``generators``.
- All promoted revisions pass through ``OperatorVersionStore`` (retiring prior versions
  reversibly without deletion) and enforce strict ``freeze()`` discipline.
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from research.cpu_learning.itl.contrast_harness import (
    TransitionEnvironment,
    _clone_card_with_updates,
    _distinct_positive_worlds,
    _matches_expected_transition,
    _merge_guards_into_card,
    _recompute_card_confidence,
    acquire_guards_from_contrasts,
    build_probe_action,
    generate_domain_agnostic_contrast_probes,
)
from research.cpu_learning.itl.delta_extractor import (
    CorpusStatistics,
    ExtractedEpisodeObservation,
)
from research.cpu_learning.itl.operator_card import (
    AttributeGuard,
    ConditionalRoleFactGuard,
    IndirectRoleFactGuard,
    OperatorCard,
    RoleInequalityGuard,
)
from research.cpu_learning.itl.operator_inducer import _is_attr_functionally_entailed
from research.cpu_learning.itl.role_unifier import (
    extract_direct_lifted_facts,
    extract_entity_literal_facts,
    extract_one_hop_links,
    extract_role_attribute_map,
)
from research.cpu_learning.itl.version_store import (
    OperatorVersionStore,
    VersionStoreFrozenError,
)
from research.cpu_learning.rdc.mdl_judge import (
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    MDLExample,
    evaluate_library_mdl,
)
from research.cpu_learning.simulator.state_graph import (
    ActionInvocation,
    SparseDelta,
    StateGraph,
    canonical_json_bytes,
    sha256_hex,
)

RDC_REFUTER_SCHEMA_VERSION: str = "s5-rdc-refuter-v1"


@dataclass(frozen=True)
class DreamProposal:
    """A system-synthesized boundary probe (NEVER ground truth until judged by an independent env)."""

    probe_id: str
    action_id: str
    before_state: StateGraph
    action: ActionInvocation
    dream_kind: str
    target_clause: tuple[str, ...] = ()


@dataclass(frozen=True)
class DreamRefutationReport:
    """Auditable outcome of running DreamRefuter against an independent black-box environment."""

    mode: str
    probe_budget: int
    dreams_proposed: int
    dreams_verified_by_env: int
    self_certified_dreams: int
    overgeneralization_counterexamples: int
    overconstraint_counterexamples: int
    promoted_versions_by_action: dict[str, int]
    j_before: float
    j_after: float
    schema_version: str = RDC_REFUTER_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "probe_budget": self.probe_budget,
            "dreams_proposed": self.dreams_proposed,
            "dreams_verified_by_env": self.dreams_verified_by_env,
            "self_certified_dreams": self.self_certified_dreams,
            "overgeneralization_counterexamples": self.overgeneralization_counterexamples,
            "overconstraint_counterexamples": self.overconstraint_counterexamples,
            "promoted_versions_by_action": dict(self.promoted_versions_by_action),
            "j_before": round(self.j_before, 4),
            "j_after": round(self.j_after, 4),
        }


def _probe_key(state: StateGraph, act: ActionInvocation) -> str:
    return sha256_hex(
        canonical_json_bytes(
            {
                "state_fp": state.fingerprint(),
                "action_id": act.action_id,
                "roles": dict(act.roles),
            }
        )
    )


def synthesize_boundary_dreams(
    cards: Mapping[str, OperatorCard],
    grouped_pos_obs: Mapping[str, Sequence[ExtractedEpisodeObservation]],
    corpus_stats: CorpusStatistics,
) -> list[DreamProposal]:
    """Synthesize interleaved domain-agnostic boundary dreams across all actions.

    Covers BOTH:
    1. Over-constraint ablation dreams: flipping each direct literal precondition
       ``(?role, pred, val)`` on ``card`` across distinct base worlds.
    2. Over-generalization boundary dreams: single-factor structural perturbations
       (role equality, relation swap, attributes, literal facts, subtype invariants,
       and 1-hop neighbor facts).

    Interleaves round-robin across actions and perturbation families so tight probe
    budgets deliver broad coverage across all operators and guard classes.
    """
    streams: list[list[DreamProposal]] = []
    seen_keys: set[str] = set()
    seq_counter = 0

    for aid in sorted(cards):
        card = cards[aid]
        pos_obs = grouped_pos_obs.get(aid, ())
        distinct_worlds = _distinct_positive_worlds(pos_obs)
        if not distinct_worlds:
            continue

        # Stream 1: Over-constraint direct literal ablation dreams
        ablation_stream: list[DreamProposal] = []
        for subj_t, pred, val in card.required_facts:
            if not subj_t.startswith("?") or val.startswith("?"):
                continue
            role = subj_t[1:]
            for base_state, ep in distinct_worlds:
                eid = ep.roles.get(role)
                if eid is None or not base_state.has_fact(eid, pred, val):
                    continue
                etype = ep.before_frame.entity_type_of(eid) or ""
                alts = sorted(
                    (
                        corpus_stats.literal_fact_domain_for_type(etype, pred)
                        | corpus_stats.global_literal_fact_domain(pred)
                    )
                    - {val}
                )
                for alt_v in alts:
                    cf_state = base_state.with_changes(
                        delete_facts=((eid, pred, val),),
                        add_facts=((eid, pred, alt_v),),
                    )
                    act = build_probe_action(aid, ep.roles)
                    k = _probe_key(cf_state, act)
                    if k not in seen_keys:
                        seen_keys.add(k)
                        seq_counter += 1
                        ablation_stream.append(
                            DreamProposal(
                                probe_id=f"dream_ablate_{aid}_{seq_counter:04d}",
                                action_id=aid,
                                before_state=cf_state,
                                action=act,
                                dream_kind="overconstraint_ablation",
                                target_clause=(subj_t, pred, val),
                            )
                        )
        if ablation_stream:
            streams.append(ablation_stream)

        # Stream 2: Over-generalization single-factor boundary dreams
        raw_contrasts = generate_domain_agnostic_contrast_probes(aid, card, pos_obs, corpus_stats)
        overgen_stream: list[DreamProposal] = []
        for cf_state, act in raw_contrasts:
            k = _probe_key(cf_state, act)
            if k not in seen_keys:
                seen_keys.add(k)
                seq_counter += 1
                overgen_stream.append(
                    DreamProposal(
                        probe_id=f"dream_guard_{aid}_{seq_counter:04d}",
                        action_id=aid,
                        before_state=cf_state,
                        action=act,
                        dream_kind="overgeneralization_boundary",
                    )
                )
        if overgen_stream:
            streams.append(overgen_stream)

    # Round-robin interleave across streams so every action & dream kind gets budget early
    interleaved: list[DreamProposal] = []
    max_len = max((len(s) for s in streams), default=0)
    for idx in range(max_len):
        for st in streams:
            if idx < len(st):
                interleaved.append(st[idx])
    return interleaved


def synthesize_random_world_probes(
    cards: Mapping[str, OperatorCard],
    grouped_pos_obs: Mapping[str, Sequence[ExtractedEpisodeObservation]],
    corpus_stats: CorpusStatistics,
    *,
    count: int,
    seed: int = 0,
) -> list[DreamProposal]:
    """Generate unguided random state-action perturbations (fair control baseline vs DreamRefuter)."""
    rng = random.Random(seed)
    action_ids = sorted(cards.keys())
    if not action_ids:
        return []

    proposals: list[DreamProposal] = []
    seen_keys: set[str] = set()
    attempts = 0
    max_attempts = max(count * 8, 200)

    while len(proposals) < count and attempts < max_attempts:
        attempts += 1
        aid = rng.choice(action_ids)
        pos_obs = grouped_pos_obs.get(aid, ())
        if not pos_obs:
            continue
        ep = rng.choice(pos_obs)
        base_state = ep.before_frame.to_observed_state_graph()
        ent_list = list(base_state.entities)
        if not ent_list:
            continue
        roles = dict(ep.roles)

        # Unguided random world mutation: perturb a random entity's attribute, fact, or role
        mut_kind = rng.randint(0, 3)
        cf_state = base_state
        if mut_kind == 0:
            # Random entity literal fact flip (often hits an unrelated distractor entity!)
            eid, ent = rng.choice(ent_list)
            lit_facts = list(extract_entity_literal_facts(ep.before_frame, eid))
            if lit_facts:
                pred, val = rng.choice(lit_facts)
                alts = sorted(
                    (
                        corpus_stats.literal_fact_domain_for_type(ent.entity_type, pred)
                        | corpus_stats.global_literal_fact_domain(pred)
                    )
                    - {val}
                )
                if alts:
                    cf_state = base_state.with_changes(
                        delete_facts=((eid, pred, val),),
                        add_facts=((eid, pred, rng.choice(alts)),),
                    )
        elif mut_kind == 1:
            # Random entity attribute flip (often hits decorative color/distractor!)
            eid, ent = rng.choice(ent_list)
            if ent.attributes:
                ak, av = rng.choice(ent.attributes)
                alts = [
                    v
                    for v in corpus_stats.attr_domain_for_types(frozenset({ent.entity_type}), ak)
                    if v != av
                ]
                if alts:
                    cf_state = base_state.with_changes(attribute_edits=((eid, ak, rng.choice(alts)),))
        elif mut_kind == 2:
            # Random role substitution with a random entity of any matching type
            rname = rng.choice(sorted(roles.keys()))
            allowed = cards[aid].allowed_types_map().get(rname, frozenset())
            matching = [eid for eid, ent in ent_list if not allowed or ent.entity_type in allowed]
            if matching:
                roles[rname] = rng.choice(matching)
        else:
            # Unperturbed positive world (random baseline may query already-valid worlds)
            pass

        act = build_probe_action(aid, roles)
        k = _probe_key(cf_state, act)
        if k not in seen_keys:
            seen_keys.add(k)
            proposals.append(
                DreamProposal(
                    probe_id=f"rand_probe_{len(proposals):04d}",
                    action_id=aid,
                    before_state=cf_state,
                    action=act,
                    dream_kind="random_perturbation",
                )
            )
    return proposals


def _find_sole_violated_direct_literal(
    card: OperatorCard,
    state: StateGraph,
    roles: Mapping[str, str],
) -> tuple[str, str, str] | None:
    """Return the unique non-deleted direct literal in ``card.required_facts`` violated by
    ``(state, roles)`` iff relaxing ONLY that clause makes all other preconditions on ``card`` hold."""
    deleted_set = set(card.delete_facts)
    violated_lits: list[tuple[str, str, str]] = []
    for s_t, pred, val in card.required_facts:
        if not (s_t.startswith("?") and not val.startswith("?")):
            continue
        eid = roles.get(s_t[1:])
        if eid is None or not state.has_fact(eid, pred, val):
            if (s_t, pred, val) in deleted_set:
                return None
            violated_lits.append((s_t, pred, val))
    if len(violated_lits) != 1:
        return None
    sole = violated_lits[0]
    relaxed = _clone_card_with_updates(
        card,
        version=card.version,
        parent_version=card.parent_version,
        required_facts=tuple(f for f in card.required_facts if f != sole),
    )
    ok, _reason, unknown = relaxed.check_preconditions(state, roles)
    return sole if (ok and not unknown) else None


class DreamRefuter:
    """Mechanism 1: Independent-Environment Dream Refutation & Versioned Refinement.

    Never self-certifies a dream: every proposed dream is evaluated strictly by the
    injected black-box ``TransitionEnvironment.step(...)``, and revisions are only
    promoted when repeated independent counterexamples justify them and MDL ``J`` drops.
    """

    def __init__(
        self,
        *,
        min_ablation_support: int = 2,
        min_guard_support: int = 2,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
    ) -> None:
        self.min_ablation_support = int(min_ablation_support)
        self.min_guard_support = int(min_guard_support)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def refine_library(
        self,
        version_store: OperatorVersionStore,
        grouped_pos_obs: Mapping[str, Sequence[ExtractedEpisodeObservation]],
        grouped_neg_obs: Mapping[str, Sequence[ExtractedEpisodeObservation]],
        corpus_stats: CorpusStatistics,
        env: TransitionEnvironment,
        *,
        probe_budget: int = 240,
        mode: str = "rdc_dream_refutation",
        random_seed: int = 0,
    ) -> DreamRefutationReport:
        """Run budgeted dream refutation (or random-probe control) against ``env``."""
        if version_store.is_frozen and mode != "plain_mdl_no_dreams" and probe_budget > 0:
            raise VersionStoreFrozenError(
                "DreamRefuter cannot refine a frozen OperatorVersionStore"
            )
        active_cards = version_store.active_cards()
        if mode == "plain_mdl_no_dreams" or probe_budget <= 0:
            base_examples = [
                MDLExample(
                    example_id=ep.episode_id,
                    before_state=ep.before_frame.to_observed_state_graph(),
                    actions=(build_probe_action(ep.action_id, ep.roles),),
                    observed_after_state=ep.after_frame.to_observed_state_graph(),
                    mask=ep.before_frame.observation_mask,
                )
                for obs_list in list(grouped_pos_obs.values()) + list(grouped_neg_obs.values())
                for ep in obs_list
            ]
            sc = evaluate_library_mdl(active_cards, base_examples, beta=self.beta, gamma=self.gamma)
            return DreamRefutationReport(
                mode=mode,
                probe_budget=0,
                dreams_proposed=0,
                dreams_verified_by_env=0,
                self_certified_dreams=0,
                overgeneralization_counterexamples=0,
                overconstraint_counterexamples=0,
                promoted_versions_by_action={aid: c.version for aid, c in active_cards.items()},
                j_before=sc.j_score,
                j_after=sc.j_score,
            )

        if mode == "rdc_random_probes":
            dreams = synthesize_random_world_probes(
                active_cards,
                grouped_pos_obs,
                corpus_stats,
                count=probe_budget,
                seed=random_seed,
            )
        else:
            dreams = synthesize_boundary_dreams(active_cards, grouped_pos_obs, corpus_stats)

        selected_dreams = dreams[:probe_budget]

        # Execute EVERY selected dream strictly through the independent black-box env (0 self-certification!)
        verified_examples: list[MDLExample] = []
        for obs_list in list(grouped_pos_obs.values()) + list(grouped_neg_obs.values()):
            for ep in obs_list:
                verified_examples.append(
                    MDLExample(
                        example_id=ep.episode_id,
                        before_state=ep.before_frame.to_observed_state_graph(),
                        actions=(build_probe_action(ep.action_id, ep.roles),),
                        observed_after_state=ep.after_frame.to_observed_state_graph(),
                        mask=ep.before_frame.observation_mask,
                    )
                )

        # Replay cache so guard acquisition reuses the exact env responses without extra env calls
        cached_responses: dict[str, tuple[StateGraph, SparseDelta]] = {}
        overgen_by_action: dict[str, list[tuple[StateGraph, ActionInvocation]]] = {}
        ablation_support: dict[tuple[str, tuple[str, str, str]], set[str]] = {}
        ablation_contras: dict[tuple[str, tuple[str, str, str]], int] = {}
        overgen_count = 0
        overcon_count = 0
        env_queries_used = 0

        for dream in selected_dreams:
            card = active_cards.get(dream.action_id)
            if card is None:
                continue
            env_after, env_delta = env.step(dream.before_state, dream.action)
            env_queries_used += 1
            pk = _probe_key(dream.before_state, dream.action)
            cached_responses[pk] = (env_after, env_delta)
            verified_examples.append(
                MDLExample(
                    example_id=dream.probe_id,
                    before_state=dream.before_state,
                    actions=(dream.action,),
                    observed_after_state=env_after,
                )
            )
            roles_d = dream.action.role_dict()
            pred = card.apply(dream.before_state, roles_d)
            if pred.applied and env_delta.is_noop:
                overgen_count += 1
                overgen_by_action.setdefault(dream.action_id, []).append(
                    (dream.before_state, dream.action)
                )
            elif not pred.applied:
                if not env_delta.is_noop:
                    overcon_count += 1
                # Infer single-clause direct literal ablation identically for ALL probe sources
                sole_clause = _find_sole_violated_direct_literal(card, dream.before_state, roles_d)
                if sole_clause is not None:
                    clause_key = (dream.action_id, sole_clause)
                    if (not env_delta.is_noop) and _matches_expected_transition(
                        card, dream.before_state, roles_d, env_after, env_delta
                    ):
                        ablation_support.setdefault(clause_key, set()).add(
                            dream.before_state.fingerprint()
                        )
                    else:
                        ablation_contras[clause_key] = ablation_contras.get(clause_key, 0) + 1

        # Cached black-box replay adapter (strictly returns ONLY the independent env responses already observed!)
        class _CachedEnvReplay:
            def step(
                self,
                state: StateGraph,
                action: ActionInvocation,
                mask: Any = None,
            ) -> tuple[StateGraph, SparseDelta]:
                key = _probe_key(state, action)
                return cached_responses[key]

        replay_env = _CachedEnvReplay()

        # Pre-compute candidate refuted clauses (supp >= min_ablation_support and 0 contradictions)
        cand_prune_by_action: dict[str, list[tuple[str, str, str]]] = {}
        for aid in sorted(active_cards):
            card = active_cards[aid]
            cand_prune: list[tuple[str, str, str]] = []
            for f_clause in card.required_facts:
                ckey = (aid, f_clause)
                supp = len(ablation_support.get(ckey, ()))
                contra = ablation_contras.get(ckey, 0)
                if supp >= self.min_ablation_support and contra == 0:
                    cand_prune.append(f_clause)
            if cand_prune:
                cand_prune_by_action[aid] = cand_prune

        # Pass 1: Over-generalization guard acquisition FIRST (strict repeated-evidence
        #         min_guard_support gate!) so missing guards are repaired before evaluating
        #         whether an incidental precondition can be safely pruned against neg_obs.
        for aid in sorted(overgen_by_action):
            probes_here = overgen_by_action[aid]
            if not probes_here:
                continue
            acquire_guards_from_contrasts(
                aid,
                version_store,
                grouped_pos_obs.get(aid, ()),
                probes_here,
                corpus_stats,
                replay_env,
                min_guard_support=self.min_guard_support,
                neg_obs=grouped_neg_obs.get(aid, ()),
                refuted_facts=frozenset(cand_prune_by_action.get(aid, ())),
            )

        # Pass 2: Over-constraint pruning on the freshly guard-repaired active cards
        #         (requires >= min_ablation_support distinct worlds, 0 contradictions,
        #         joint neg_obs consistency, and joint counterfactual verification for >= 2 clauses)
        repaired_cards = version_store.active_cards()
        for aid in sorted(repaired_cards):
            card = repaired_cards[aid]
            cand_prune = [
                f_clause
                for f_clause in cand_prune_by_action.get(aid, ())
                if f_clause in card.required_facts
            ]
            if not cand_prune:
                continue

            safe_prune: list[tuple[str, str, str]] = []
            distinct_w = _distinct_positive_worlds(grouped_pos_obs.get(aid, ()))
            for f_clause in cand_prune:
                trial_group = safe_prune + [f_clause]
                trial = _clone_card_with_updates(
                    card,
                    version=card.version,
                    parent_version=card.parent_version,
                    required_facts=tuple(f for f in card.required_facts if f not in set(trial_group)),
                )
                neg_ok = all(
                    not trial.check_preconditions(
                        nep.before_frame.to_observed_state_graph(),
                        nep.roles,
                        mask=nep.before_frame.observation_mask,
                    )[0]
                    for nep in grouped_neg_obs.get(aid, ())
                )
                if not neg_ok:
                    continue
                if len(trial_group) == 1:
                    safe_prune.append(f_clause)
                    continue
                # Verify simultaneous flip of >= 2 clauses against independent env (within remaining budget)
                joint_ok = 0
                joint_contra = 0
                for base_s, ep in distinct_w:
                    if env_queries_used >= probe_budget:
                        break
                    del_f = []
                    add_f = []
                    applicable = True
                    for st_t, p_t, v_t in trial_group:
                        eid = ep.roles.get(st_t[1:])
                        if eid is None or not base_s.has_fact(eid, p_t, v_t):
                            applicable = False
                            break
                        et = ep.before_frame.entity_type_of(eid) or ""
                        alts = sorted(
                            (
                                corpus_stats.literal_fact_domain_for_type(et, p_t)
                                | corpus_stats.global_literal_fact_domain(p_t)
                            )
                            - {v_t}
                        )
                        if not alts:
                            applicable = False
                            break
                        del_f.append((eid, p_t, v_t))
                        add_f.append((eid, p_t, alts[0]))
                    if not applicable:
                        continue
                    cf_s = base_s.with_changes(delete_facts=del_f, add_facts=add_f)
                    act_j = build_probe_action(aid, ep.roles)
                    cf_aft, cf_del = env.step(cf_s, act_j)
                    env_queries_used += 1
                    verified_examples.append(
                        MDLExample(
                            example_id=f"dream_joint_{aid}_{env_queries_used:04d}",
                            before_state=cf_s,
                            actions=(act_j,),
                            observed_after_state=cf_aft,
                        )
                    )
                    if _matches_expected_transition(card, cf_s, ep.roles, cf_aft, cf_del):
                        joint_ok += 1
                    else:
                        joint_contra += 1
                if joint_contra == 0 and joint_ok >= self.min_ablation_support:
                    safe_prune.append(f_clause)

            to_prune = safe_prune
            if to_prune:
                new_ver = version_store.next_version(aid)
                pruned_card = _clone_card_with_updates(
                    card,
                    version=new_ver,
                    parent_version=card.version,
                    required_facts=tuple(f for f in card.required_facts if f not in set(to_prune)),
                )
                conf = _recompute_card_confidence(
                    pruned_card, grouped_pos_obs.get(aid, ()), grouped_neg_obs.get(aid, ())
                )
                pruned_card = _clone_card_with_updates(
                    pruned_card,
                    version=pruned_card.version,
                    parent_version=pruned_card.parent_version,
                    confidence=conf,
                )
                # Verify strict MDL J improvement before promoting
                trial_lib = dict(version_store.active_cards())
                trial_lib[aid] = pruned_card
                trial_mdl = evaluate_library_mdl(
                    trial_lib, verified_examples, beta=self.beta, gamma=self.gamma
                )
                cur_mdl = evaluate_library_mdl(
                    version_store.active_cards(),
                    verified_examples,
                    beta=self.beta,
                    gamma=self.gamma,
                )
                version_store.register_shadow(
                    pruned_card,
                    evidence={
                        "source": "s5_rdc_dream_overconstraint_pruning",
                        "refuted_clauses": [list(c) for c in sorted(to_prune)],
                        "j_before": cur_mdl.j_score,
                        "j_after": trial_mdl.j_score,
                    },
                    contrast_pass_rate=trial_mdl.exact_match_rate,
                    notes="rdc_dream_ablation_candidate",
                )
                if trial_mdl.j_score < cur_mdl.j_score:
                    version_store.promote(
                        aid,
                        pruned_card.version,
                        notes="promoted_by_rdc_dream_refutation",
                    )

        final_cards = version_store.active_cards()
        before_mdl = evaluate_library_mdl(
            active_cards, verified_examples, beta=self.beta, gamma=self.gamma
        )
        after_mdl = evaluate_library_mdl(
            final_cards, verified_examples, beta=self.beta, gamma=self.gamma
        )

        return DreamRefutationReport(
            mode=mode,
            probe_budget=probe_budget,
            dreams_proposed=env_queries_used,
            dreams_verified_by_env=env_queries_used,
            self_certified_dreams=0,
            overgeneralization_counterexamples=overgen_count,
            overconstraint_counterexamples=overcon_count,
            promoted_versions_by_action={aid: c.version for aid, c in sorted(final_cards.items())},
            j_before=before_mdl.j_score,
            j_after=after_mdl.j_score,
        )


# ---------------------------------------------------------------------------
# Mechanism 2: Maximum-Disagreement Version-Space Active Query Selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActiveQueryResult:
    """Audit outcome of active query selection over a hypothesis version-space."""

    strategy: str
    action_id: str
    initial_hypotheses: int
    surviving_hypotheses: int
    queries_used: int
    queries_to_unique_or_target: int
    selected_card: OperatorCard
    query_disagreement_trace: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "action_id": self.action_id,
            "initial_hypotheses": self.initial_hypotheses,
            "surviving_hypotheses": self.surviving_hypotheses,
            "queries_used": self.queries_used,
            "queries_to_unique_or_target": self.queries_to_unique_or_target,
            "query_disagreement_trace": [round(x, 6) for x in self.query_disagreement_trace],
        }


def build_hypothesis_version_space(
    base_core_card: OperatorCard,
    pos_obs: Sequence[ExtractedEpisodeObservation],
    corpus_stats: CorpusStatistics,
    *,
    init_neg_obs: Sequence[ExtractedEpisodeObservation] = (),
    max_optional_clauses: int = 8,
) -> list[OperatorCard]:
    """Construct a finite committee of candidate ``OperatorCard`` hypotheses consistent with ``pos_obs``.

    Extracts all candidate optional guards holding across ``pos_obs`` (direct literal facts,
    conditional role facts, 1-hop indirect facts, non-entailed attribute guards, and role
    inequality guards) and enumerates their subsets (up to ``2**max_optional_clauses``),
    retaining all hypotheses consistent with ``init_neg_obs``. 100% domain-agnostic.
    """
    if not pos_obs:
        return [base_core_card]

    allowed_types = base_core_card.allowed_types_map()
    role_vars = list(base_core_card.role_variables)
    deleted_set = set(base_core_card.delete_facts)
    existing_req = set(base_core_card.required_facts)

    direct_sets = [extract_direct_lifted_facts(ep.before_frame, ep.roles) for ep in pos_obs]
    pos_common_direct = frozenset.intersection(*direct_sets) if direct_sets else frozenset()

    optional_clauses: list[tuple[str, Any]] = []
    seen_clauses: set[tuple[str, Any]] = set()

    def _add_opt(kind: str, payload: Any) -> None:
        item = (kind, payload)
        if item not in seen_clauses:
            seen_clauses.add(item)
            optional_clauses.append(item)

    # 1. Direct literal facts not in delete_facts and not already on base_core_card
    for s_t, pred, o_t in sorted(pos_common_direct):
        if s_t.startswith("?") and not o_t.startswith("?") and (s_t, pred, o_t) not in deleted_set:
            if (s_t, pred, o_t) not in existing_req:
                _add_opt("required_fact", (s_t, pred, o_t))

    # 2. Role inequality guards between type-compatible roles
    effect_orient: dict[frozenset[str], tuple[str, str]] = {}
    for ds, dp, do in base_core_card.delete_facts:
        for as_, ap, ao in base_core_card.add_facts:
            if dp != ap:
                continue
            if ds == as_ and do != ao and do.startswith("?") and ao.startswith("?"):
                effect_orient[frozenset({do[1:], ao[1:]})] = (do[1:], ao[1:])
            elif do == ao and ds != as_ and ds.startswith("?") and as_.startswith("?"):
                effect_orient[frozenset({ds[1:], as_[1:]})] = (ds[1:], as_[1:])
    existing_ineq = {
        (min(g.role_a, g.role_b), max(g.role_a, g.role_b))
        for g in base_core_card.inequality_guards
    }
    for i in range(len(role_vars)):
        for j in range(i + 1, len(role_vars)):
            r1, r2 = role_vars[i], role_vars[j]
            if not (allowed_types.get(r1, frozenset()) & allowed_types.get(r2, frozenset())):
                continue
            if (min(r1, r2), max(r1, r2)) in existing_ineq:
                continue
            if all(ep.roles.get(r1) != ep.roles.get(r2) for ep in pos_obs):
                ra, rb = effect_orient.get(frozenset({r1, r2}), (r1, r2))
                _add_opt("inequality", RoleInequalityGuard(role_a=ra, role_b=rb))

    # 3. Non-entailed varying attribute guards
    pos_attr_maps = [extract_role_attribute_map(ep.before_frame, ep.roles) for ep in pos_obs]
    existing_attrs = set(base_core_card.attribute_guards)
    for rname in role_vars:
        common_ks: set[str] | None = None
        for amap in pos_attr_maps:
            ks = set(amap.get(rname, {}).keys())
            common_ks = ks if common_ks is None else (common_ks & ks)
        if not common_ks:
            continue
        r_types = allowed_types.get(rname, frozenset())
        role_lits = frozenset(
            (pred, obj)
            for subj, pred, obj in pos_common_direct
            if subj == f"?{rname}" and not obj.startswith("?")
        )
        for ak in sorted(common_ks):
            vals = {amap[rname][ak] for amap in pos_attr_maps}
            if len(vals) != 1:
                continue
            exp_v = next(iter(vals))
            if len(corpus_stats.attr_domain_for_types(r_types, ak)) < 2:
                continue
            if _is_attr_functionally_entailed(r_types, ak, exp_v, role_lits, corpus_stats):
                continue
            if not corpus_stats.is_attr_edited_on_types(r_types, ak):
                if corpus_stats.other_varying_pos_role_slots(
                    r_types, ak, exclude_action=base_core_card.action_id, exclude_role=rname
                ):
                    continue
            ag = AttributeGuard(role=rname, attribute=ak, expected_value=exp_v)
            if ag not in existing_attrs:
                _add_opt("attribute", ag)

    # 4. Conditional subtype guards for multi-type roles
    existing_cond = set(base_core_card.conditional_fact_guards)
    for rname in role_vars:
        r_types = allowed_types.get(rname, frozenset())
        if len(r_types) <= 1:
            continue
        uncond_preds = {pred for subj, pred, _obj in pos_common_direct if subj == f"?{rname}"}
        preds_by_sub: dict[str, set[str]] = {}
        invs_by_sub: dict[str, set[tuple[str, str]]] = {}
        for sub_t in sorted(r_types):
            sub_eps = [ep for ep in pos_obs if ep.before_frame.entity_type_of(ep.roles[rname]) == sub_t]
            if not sub_eps:
                continue
            lsets = [extract_entity_literal_facts(ep.before_frame, ep.roles[rname]) for ep in sub_eps]
            inter = set(lsets[0])
            for ls in lsets[1:]:
                inter &= ls
            invs_by_sub[sub_t] = inter
            preds_by_sub[sub_t] = {p for ls in lsets for p, _v in ls}
        for sub_t, inv_fs in sorted(invs_by_sub.items()):
            other_ps: set[str] | None = None
            for ot, ops in preds_by_sub.items():
                if ot == sub_t:
                    continue
                other_ps = set(ops) if other_ps is None else (other_ps & ops)
            shared = other_ps or set()
            for pred, val in sorted(inv_fs):
                if pred in uncond_preds or pred in shared:
                    continue
                cg = ConditionalRoleFactGuard(role=rname, when_entity_type=sub_t, predicate=pred, value=val)
                if cg not in existing_cond:
                    _add_opt("conditional_fact", cg)

    # 5. Indirect 1-hop neighbor guards
    existing_ind = set(base_core_card.indirect_fact_guards)
    pos_links_by_key: dict[tuple[str, str], list[tuple[ExtractedEpisodeObservation, str, str]]] = {}
    for pep in pos_obs:
        for link in extract_one_hop_links(pep.before_frame, pep.roles):
            pos_links_by_key.setdefault((link.role, link.via_predicate), []).append(
                (pep, link.neighbor_id, link.neighbor_type)
            )
    for (rname, via_pred), entries in sorted(pos_links_by_key.items()):
        for ntype in sorted({nt for _p, _nid, nt in entries}):
            same_t = [(pep, nid) for pep, nid, nt in entries if nt == ntype]
            lsets = [extract_entity_literal_facts(pep.before_frame, nid) for pep, nid in same_t]
            if not lsets:
                continue
            pos_invs = set(lsets[0])
            for ls in lsets[1:]:
                pos_invs &= ls
            other_ps = {
                p
                for pep, nid, nt in entries
                if nt != ntype
                for p, _v in extract_entity_literal_facts(pep.before_frame, nid)
            }
            cand_invs = {(p, v) for p, v in pos_invs if p not in other_ps}
            for pred, val in sorted(cand_invs):
                ig = IndirectRoleFactGuard(
                    role=rname,
                    via_predicate=via_pred,
                    when_neighbor_type=ntype,
                    predicate=pred,
                    value=val,
                )
                if ig not in existing_ind:
                    _add_opt("indirect_fact", ig)

    capped = optional_clauses[:max_optional_clauses]
    hypotheses: list[OperatorCard] = []
    h_idx = 0
    for r in range(len(capped) + 1):
        for subset in itertools.combinations(capped, r):
            h_idx += 1
            cand = _merge_guards_into_card(
                base_core_card,
                subset,
                version=h_idx,
                parent_version=base_core_card.version,
            )
            if init_neg_obs:
                ok_negs = all(
                    not cand.check_preconditions(
                        nep.before_frame.to_observed_state_graph(),
                        nep.roles,
                        mask=nep.before_frame.observation_mask,
                    )[0]
                    for nep in init_neg_obs
                )
                if not ok_negs:
                    continue
            hypotheses.append(cand)

    return hypotheses if hypotheses else [base_core_card]


def score_probe_disagreement(
    state: StateGraph,
    action: ActionInvocation,
    hypotheses: Sequence[OperatorCard],
) -> float:
    """Compute exact Shannon entropy over competing hypotheses' predicted outcomes on ``(state, action)``."""
    n = len(hypotheses)
    if n <= 1:
        return 0.0
    roles = action.role_dict()
    vote_counts: dict[tuple[bool, SparseDelta | None], int] = {}
    effect_cache: dict[tuple[Any, Any, Any], tuple[bool, SparseDelta | None]] = {}
    for h in hypotheses:
        ok, _reason, unknown = h.check_preconditions(state, roles)
        if unknown or not ok:
            sig: tuple[bool, SparseDelta | None] = (False, None)
        else:
            ekey = (h.add_facts, h.delete_facts, h.attribute_edits)
            cached = effect_cache.get(ekey)
            if cached is None:
                res = h.apply(state, roles)
                cached = (res.applied, res.delta if res.applied else None)
                effect_cache[ekey] = cached
            sig = cached
        vote_counts[sig] = vote_counts.get(sig, 0) + 1
    if len(vote_counts) <= 1:
        return 0.0
    ent = 0.0
    for cnt in vote_counts.values():
        p = cnt / float(n)
        ent -= p * math.log2(p)
    return ent


class MaxDisagreementQueryEngine:
    """Mechanism 2: Version-Space Query-by-Committee Active Query Selection vs Random Control."""

    def run_active_selection(
        self,
        action_id: str,
        initial_hypotheses: Sequence[OperatorCard],
        candidate_pool: Sequence[tuple[StateGraph, ActionInvocation]],
        env: TransitionEnvironment,
        *,
        strategy: str = "max_disagreement_query",
        max_queries: int = 64,
        seed: int = 0,
        target_eval_fn: Any = None,
        target_accuracy: float = 0.98,
    ) -> ActiveQueryResult:
        """Iteratively select probes from ``candidate_pool`` to narrow ``initial_hypotheses``.

        - ``strategy="max_disagreement_query"``: selects the unqueried probe maximizing
          Shannon entropy disagreement across surviving hypotheses.
        - ``strategy="random_query"``: selects unqueried probes uniformly at random without
          replacement (fair control baseline on identical hypothesis space & probe pool).
        """
        if not initial_hypotheses:
            raise ValueError("initial_hypotheses must be non-empty")
        surviving: list[OperatorCard] = list(initial_hypotheses)
        n_init = len(surviving)
        pool_indices = list(range(len(candidate_pool)))
        rng = random.Random(seed)
        if strategy == "random_query":
            rng.shuffle(pool_indices)

        queried_indices: set[int] = set()
        zero_disagreement_indices: set[int] = set()
        disagreement_trace: list[float] = []
        queries_used = 0
        queries_to_target: int | None = None

        # Precompute deterministic MDL-minimal sort keys once (avoids repeated JSON hashing)
        rep_sort_keys: dict[int, tuple[int, str]] = {
            id(c): (
                len(c.required_facts)
                + len(c.conditional_fact_guards)
                + len(c.indirect_fact_guards)
                + len(c.attribute_guards)
                + len(c.inequality_guards),
                c.fingerprint(),
            )
            for c in surviving
        }

        def _representative_card(hyps: Sequence[OperatorCard]) -> OperatorCard:
            return min(hyps, key=lambda c: rep_sort_keys[id(c)])

        if target_eval_fn is not None and surviving:
            if target_eval_fn(_representative_card(surviving)) >= target_accuracy:
                queries_to_target = 0

        while queries_used < max_queries and len(surviving) > 1 and len(queried_indices) < len(candidate_pool):
            chosen_idx: int | None = None
            chosen_dis = 0.0

            if strategy == "max_disagreement_query":
                best_score = -1.0
                for idx in pool_indices:
                    if idx in queried_indices or idx in zero_disagreement_indices:
                        continue
                    st, act = candidate_pool[idx]
                    sc = score_probe_disagreement(st, act, surviving)
                    if sc <= 0.0:
                        zero_disagreement_indices.add(idx)
                        continue
                    if sc > best_score:
                        best_score = sc
                        chosen_idx = idx
                        chosen_dis = sc
                if chosen_idx is None or best_score <= 0.0:
                    break
            else:
                for idx in pool_indices:
                    if idx not in queried_indices:
                        chosen_idx = idx
                        break
                if chosen_idx is None:
                    break
                st0, act0 = candidate_pool[chosen_idx]
                chosen_dis = score_probe_disagreement(st0, act0, surviving)

            queried_indices.add(chosen_idx)
            st_q, act_q = candidate_pool[chosen_idx]
            env_after, env_delta = env.step(st_q, act_q)
            queries_used += 1
            disagreement_trace.append(chosen_dis)

            roles_q = act_q.role_dict()
            env_applied = not env_delta.is_noop
            effect_match_cache: dict[tuple[Any, Any, Any], bool] = {}
            next_surviving: list[OperatorCard] = []
            for h in surviving:
                ok, _reason, unknown = h.check_preconditions(st_q, roles_q)
                if unknown:
                    continue
                if not ok:
                    if (not env_applied) and st_q == env_after:
                        next_surviving.append(h)
                    continue
                ekey = (h.add_facts, h.delete_facts, h.attribute_edits)
                matched_after = effect_match_cache.get(ekey)
                if matched_after is None:
                    res = h.apply(st_q, roles_q)
                    matched_after = (
                        (not res.abstained)
                        and res.applied == env_applied
                        and res.after_state == env_after
                    )
                    effect_match_cache[ekey] = matched_after
                if matched_after:
                    next_surviving.append(h)

            prev_len = len(surviving)
            if next_surviving:
                surviving = next_surviving

            if queries_to_target is None and len(surviving) < prev_len:
                if target_eval_fn is not None:
                    if target_eval_fn(_representative_card(surviving)) >= target_accuracy:
                        queries_to_target = queries_used
                elif len(surviving) == 1:
                    queries_to_target = queries_used

            if len(surviving) == 1 and queries_to_target is not None:
                break

        rep_card = _representative_card(surviving)
        if queries_to_target is None:
            queries_to_target = queries_used

        return ActiveQueryResult(
            strategy=strategy,
            action_id=action_id,
            initial_hypotheses=n_init,
            surviving_hypotheses=len(surviving),
            queries_used=queries_used,
            queries_to_unique_or_target=queries_to_target,
            selected_card=rep_card,
            query_disagreement_trace=tuple(disagreement_trace),
        )
