"""Stage S3 Domain-Agnostic Controlled Contrast Harness, Guard Refiner & Contrast-Refined ITL Learner.

STRICT ISOLATION (Stage S3):
- Contains ZERO hardcoded action names, ZERO hardcoded predicate names, ZERO hardcoded
  attribute names, and ZERO hardcoded entity-type names.
- NEVER imports ``hidden_actions``, ``oracle``, or ``generators``.
- Interacts with any environment strictly through the public black-box
  ``TransitionEnvironment`` protocol returning only ``(after_state, sparse_delta)``.
- Enforces strict freeze discipline: ZERO environment queries and ZERO episode-ledger
  reads after ``freeze()`` or during ``predict()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from research.cpu_learning.frames.state_frame import TypedStateFrame
from research.cpu_learning.itl.delta_extractor import (
    CorpusStatistics,
    ExtractedEpisodeObservation,
    collect_corpus_statistics,
    extract_observations_from_ledger,
    partition_positive_negative,
)
from research.cpu_learning.itl.operator_card import (
    DEFAULT_WORLD_TYPE_TAXONOMY,
    AttributeGuard,
    ConditionalRoleFactGuard,
    IndirectRoleFactGuard,
    OperatorCard,
    RoleInequalityGuard,
    TypeTaxonomy,
    _resolve_var,
)
from research.cpu_learning.itl.operator_inducer import (
    ITLOperatorLearner,
    _is_attr_functionally_entailed,
)
from research.cpu_learning.itl.role_unifier import (
    LiftedDelta,
    extract_direct_lifted_facts,
    extract_entity_literal_facts,
    extract_one_hop_links,
    extract_role_attribute_map,
    lift_sparse_delta,
)
from research.cpu_learning.itl.version_store import (
    DEFAULT_VERSION_TIMESTAMP,
    OperatorVersionStore,
    VersionStoreError,
)
from research.cpu_learning.ledger.episode_ledger import EpisodeLedger
from research.cpu_learning.simulator.state_graph import (
    ActionInvocation,
    EvaluationTaskPublic,
    ObservationMask,
    SparseDelta,
    StateGraph,
    TrainingEpisode,
    canonical_json_bytes,
    compute_sparse_delta,
    sha256_hex,
)

CONTRAST_AUDIT_SCHEMA_VERSION: str = "s3-contrast-audit-v1"


@runtime_checkable
class TransitionEnvironment(Protocol):
    """Public black-box transition interface returning only ``(after_state, sparse_delta)``."""

    def step(
        self,
        state: StateGraph,
        action: ActionInvocation,
        mask: ObservationMask | None = None,
    ) -> tuple[StateGraph, SparseDelta]:
        ...


def build_probe_action(action_id: str, roles: Mapping[str, str]) -> ActionInvocation:
    """Construct a domain-agnostic synthetic ActionInvocation for black-box contrast probing."""
    ordered_roles = tuple(sorted((str(k), str(v)) for k, v in roles.items()))
    raw_text = action_id + "".join(f" {k}={v}" for k, v in ordered_roles)
    raw_bytes = raw_text.encode("utf-8")
    act_len = len(action_id.encode("utf-8"))
    return ActionInvocation(
        action_id=action_id,
        roles=ordered_roles,
        raw_instruction_bytes=raw_bytes,
        byte_spans=(("action", 0, act_len),),
    )


def _state_action_fingerprint(state: StateGraph, action_id: str, roles: Mapping[str, str]) -> str:
    payload = {
        "state_fp": state.fingerprint(),
        "action_id": action_id,
        "roles": dict(sorted((str(k), str(v)) for k, v in roles.items())),
    }
    return sha256_hex(canonical_json_bytes(payload))


def _apply_card_effects_unconditionally(
    card: OperatorCard,
    state: StateGraph,
    roles: Mapping[str, str],
) -> StateGraph:
    """Apply only the lifted add/delete/attribute effects of ``card`` to ``state``."""
    del_concrete = tuple(
        (_resolve_var(s, roles), p, _resolve_var(o, roles)) for s, p, o in card.delete_facts
    )
    add_concrete = tuple(
        (_resolve_var(s, roles), p, _resolve_var(o, roles)) for s, p, o in card.add_facts
    )
    edit_concrete = tuple(
        (roles[e.role], e.attribute, e.new_value) for e in card.attribute_edits
    )
    return state.with_changes(
        delete_facts=del_concrete,
        add_facts=add_concrete,
        attribute_edits=edit_concrete,
    )


def _recompute_card_confidence(
    card: OperatorCard,
    pos_obs: Sequence[ExtractedEpisodeObservation],
    neg_obs: Sequence[ExtractedEpisodeObservation],
) -> float:
    total = len(pos_obs) + len(neg_obs)
    if total == 0:
        return 1.0
    matched = 0
    for ep in pos_obs:
        res = card.apply(
            ep.before_frame.to_observed_state_graph(),
            ep.roles,
            mask=ep.before_frame.observation_mask,
        )
        if res.applied and res.delta == ep.sparse_delta:
            matched += 1
    for ep in neg_obs:
        res = card.apply(
            ep.before_frame.to_observed_state_graph(),
            ep.roles,
            mask=ep.before_frame.observation_mask,
        )
        if not res.applied and not res.abstained:
            matched += 1
    return matched / float(total)


def _clone_card_with_updates(
    base: OperatorCard,
    *,
    version: int,
    parent_version: int | None,
    required_facts: tuple[tuple[str, str, str], ...] | None = None,
    conditional_fact_guards: tuple[ConditionalRoleFactGuard, ...] | None = None,
    indirect_fact_guards: tuple[IndirectRoleFactGuard, ...] | None = None,
    attribute_guards: tuple[AttributeGuard, ...] | None = None,
    inequality_guards: tuple[RoleInequalityGuard, ...] | None = None,
    confidence: float | None = None,
    support_negative: int | None = None,
) -> OperatorCard:
    return OperatorCard(
        card_id=f"op_{base.action_id}_v{version}",
        action_id=base.action_id,
        role_variables=base.role_variables,
        required_types=base.required_types,
        required_facts=base.required_facts if required_facts is None else required_facts,
        conditional_fact_guards=(
            base.conditional_fact_guards
            if conditional_fact_guards is None
            else conditional_fact_guards
        ),
        indirect_fact_guards=(
            base.indirect_fact_guards if indirect_fact_guards is None else indirect_fact_guards
        ),
        attribute_guards=base.attribute_guards if attribute_guards is None else attribute_guards,
        inequality_guards=(
            base.inequality_guards if inequality_guards is None else inequality_guards
        ),
        add_facts=base.add_facts,
        delete_facts=base.delete_facts,
        attribute_edits=base.attribute_edits,
        preserved_invariants=base.preserved_invariants,
        scope=base.scope,
        confidence=base.confidence if confidence is None else confidence,
        support_positive=base.support_positive,
        support_negative=(
            base.support_negative if support_negative is None else support_negative
        ),
        version=version,
        parent_version=parent_version,
        provenance_episode_ids=base.provenance_episode_ids,
        provenance_record_hashes=base.provenance_record_hashes,
    )


def strip_operator_guards(card: OperatorCard, *, version: int = 1) -> OperatorCard:
    """Intentionally construct an under-constrained candidate card missing selective guards.

    Strips all attribute, conditional, indirect, and inequality guards, plus all direct
    literal ``required_facts`` not present in ``delete_facts`` (keeping only role types,
    role-to-role relational links, effect-deleted literals, and effects). 100% domain-agnostic.
    """
    deleted_set = set(card.delete_facts)
    kept_facts = tuple(
        f
        for f in card.required_facts
        if f[2].startswith("?") or f in deleted_set
    )
    return _clone_card_with_updates(
        card,
        version=version,
        parent_version=None,
        required_facts=kept_facts,
        conditional_fact_guards=(),
        indirect_fact_guards=(),
        attribute_guards=(),
        inequality_guards=(),
        confidence=1.0,
    )


@dataclass(frozen=True)
class AblationClauseReport:
    """Audit record for a single-factor precondition ablation probe."""

    action_id: str
    clause_kind: str
    clause: tuple[str, ...]
    contrasting_values_tested: tuple[str, ...]
    distinct_base_worlds_tested: int
    confirmed_effect_worlds: int
    contradiction_count: int
    refuted: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "clause_kind": self.clause_kind,
            "clause": list(self.clause),
            "contrasting_values_tested": list(self.contrasting_values_tested),
            "distinct_base_worlds_tested": self.distinct_base_worlds_tested,
            "confirmed_effect_worlds": self.confirmed_effect_worlds,
            "contradiction_count": self.contradiction_count,
            "refuted": self.refuted,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GuardAcquisitionReport:
    """Audit record for repeated-evidence guard acquisition (shadow vs promote)."""

    action_id: str
    initial_version: int
    active_version_after: int
    shadow_version_registered: int | None
    promoted: bool
    total_probes: int
    independent_overgeneralizations: int
    min_guard_support: int
    acquired_guards: tuple[dict[str, Any], ...]
    shadow_only_guards: tuple[dict[str, Any], ...]
    contrast_pass_rate_before: float
    contrast_pass_rate_after: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "initial_version": self.initial_version,
            "active_version_after": self.active_version_after,
            "shadow_version_registered": self.shadow_version_registered,
            "promoted": self.promoted,
            "total_probes": self.total_probes,
            "independent_overgeneralizations": self.independent_overgeneralizations,
            "min_guard_support": self.min_guard_support,
            "acquired_guards": [dict(g) for g in self.acquired_guards],
            "shadow_only_guards": [dict(g) for g in self.shadow_only_guards],
            "contrast_pass_rate_before": round(self.contrast_pass_rate_before, 6),
            "contrast_pass_rate_after": round(self.contrast_pass_rate_after, 6),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ProbeCategorySummary:
    """Structured summary for one contrast-audit probe family."""

    category: str
    total_probes: int
    exact_final_matches: int
    exact_final_accuracy: float
    total_steps: int
    intermediate_step_matches: int
    intermediate_step_accuracy: float
    all_steps_exact_matches: int
    all_steps_exact_accuracy: float
    non_commuting_pairs_detected: int = 0
    total_pairs: int = 0
    passed: bool = True
    details: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "total_probes": self.total_probes,
            "exact_final_matches": self.exact_final_matches,
            "exact_final_accuracy": round(self.exact_final_accuracy, 6),
            "total_steps": self.total_steps,
            "intermediate_step_matches": self.intermediate_step_matches,
            "intermediate_step_accuracy": round(self.intermediate_step_accuracy, 6),
            "all_steps_exact_matches": self.all_steps_exact_matches,
            "all_steps_exact_accuracy": round(self.all_steps_exact_accuracy, 6),
            "non_commuting_pairs_detected": self.non_commuting_pairs_detected,
            "total_pairs": self.total_pairs,
            "passed": self.passed,
            "details": [dict(d) for d in self.details],
        }


@dataclass(frozen=True)
class ContrastAuditReport:
    """JSON-serializable Stage S3 contrast audit summary across all probe families."""

    role_swap_probes: ProbeCategorySummary
    commutator_order_probes: ProbeCategorySummary
    repeated_idempotent_probes: ProbeCategorySummary
    depth_stratified_verification: dict[str, ProbeCategorySummary]
    ablation_reports: tuple[AblationClauseReport, ...] = ()
    guard_acquisition_reports: tuple[GuardAcquisitionReport, ...] = ()
    all_checks_passed: bool = True
    schema_version: str = CONTRAST_AUDIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "all_checks_passed": self.all_checks_passed,
            "role_swap_probes": self.role_swap_probes.to_dict(),
            "commutator_order_probes": self.commutator_order_probes.to_dict(),
            "repeated_idempotent_probes": self.repeated_idempotent_probes.to_dict(),
            "depth_stratified_verification": {
                k: v.to_dict() for k, v in sorted(self.depth_stratified_verification.items())
            },
            "ablation_reports": [r.to_dict() for r in self.ablation_reports],
            "guard_acquisition_reports": [r.to_dict() for r in self.guard_acquisition_reports],
        }


def _distinct_positive_worlds(
    pos_obs: Sequence[ExtractedEpisodeObservation],
) -> list[tuple[StateGraph, ExtractedEpisodeObservation]]:
    seen: set[str] = set()
    out: list[tuple[StateGraph, ExtractedEpisodeObservation]] = []
    for ep in pos_obs:
        sg = ep.before_frame.to_observed_state_graph()
        fp = sg.fingerprint()
        if fp not in seen:
            seen.add(fp)
            out.append((sg, ep))
    return out


def _matches_expected_transition(
    card: OperatorCard,
    cf_state: StateGraph,
    roles: Mapping[str, str],
    actual_after: StateGraph,
    actual_delta: SparseDelta,
) -> bool:
    if actual_delta.is_noop:
        return False
    expected_after = _apply_card_effects_unconditionally(card, cf_state, roles)
    expected_delta = compute_sparse_delta(cf_state, expected_after)
    if actual_after != expected_after or actual_delta != expected_delta:
        return False
    cf_before_frame = TypedStateFrame.from_state_graph(cf_state)
    cf_after_frame = TypedStateFrame.from_state_graph(actual_after)
    actual_lifted = lift_sparse_delta(actual_delta, roles, cf_before_frame, cf_after_frame)
    expected_lifted = LiftedDelta(
        add_facts=frozenset(card.add_facts),
        delete_facts=frozenset(card.delete_facts),
        attribute_edits=frozenset(card.attribute_edits),
    )
    return actual_lifted == expected_lifted


def refine_operator_by_single_factor_ablation(
    card: OperatorCard,
    pos_obs: Sequence[ExtractedEpisodeObservation],
    neg_obs: Sequence[ExtractedEpisodeObservation],
    corpus_stats: CorpusStatistics,
    env: TransitionEnvironment,
    *,
    min_support: int = 2,
    probe_attribute_guards: bool = False,
    next_version: int | None = None,
) -> tuple[OperatorCard | None, tuple[AblationClauseReport, ...]]:
    """Probe each direct literal precondition by single-factor counterfactual ablation.

    For every direct literal precondition ``(?role, pred, val)`` on ``card``:
    - Synthesizes minimal counterfactual states flipping ``val`` to observed contrasting
      value(s) from ``corpus_stats`` across distinct base worlds from ``pos_obs``.
    - Executes ``env.step(...)`` on each counterfactual.
    - Refutes the clause ONLY when ``>= min_support`` distinct base worlds ALL produce the
      exact expected lifted & concrete delta with ``0`` contradictions AND removing the
      clause preserves 100% consistency on ``neg_obs``.
    - When multiple clauses are individually refuted, verifies simultaneous joint ablation
      to protect against disjunctive dependencies.
    """
    distinct_worlds = _distinct_positive_worlds(pos_obs)
    reports: list[AblationClauseReport] = []
    individually_refuted: list[tuple[int, str, Any]] = []

    for subj_term, pred, val in card.required_facts:
        if not subj_term.startswith("?") or val.startswith("?"):
            continue
        role = subj_term[1:]
        tested_vals_set: set[str] = set()
        worlds_tested = 0
        confirmed_worlds = 0
        contradictions = 0

        for base_state, ep in distinct_worlds:
            eid = ep.roles.get(role)
            if eid is None or not base_state.has_fact(eid, pred, val):
                continue
            etype = ep.before_frame.entity_type_of(eid) or ""
            domain_vals = (
                corpus_stats.literal_fact_domain_for_type(etype, pred)
                | corpus_stats.global_literal_fact_domain(pred)
            )
            contrasting = sorted(v for v in domain_vals if v != val)
            if not contrasting:
                continue

            worlds_tested += 1
            world_ok = True
            for alt_val in contrasting:
                tested_vals_set.add(alt_val)
                cf_state = base_state.with_changes(
                    delete_facts=((eid, pred, val),),
                    add_facts=((eid, pred, alt_val),),
                )
                act = build_probe_action(card.action_id, ep.roles)
                cf_after, cf_delta = env.step(cf_state, act)
                if not _matches_expected_transition(card, cf_state, ep.roles, cf_after, cf_delta):
                    world_ok = False
                    contradictions += 1
            if world_ok:
                confirmed_worlds += 1

        refuted = contradictions == 0 and confirmed_worlds >= min_support
        reason = "refuted_incidental_invariant" if refuted else (
            "necessary_precondition_confirmed"
            if contradictions > 0
            else "insufficient_contrasting_support"
        )

        # Guard against breaking any existing negative demonstration
        if refuted and neg_obs:
            trial_facts = tuple(f for f in card.required_facts if f != (subj_term, pred, val))
            trial_card = _clone_card_with_updates(
                card,
                version=card.version,
                parent_version=card.parent_version,
                required_facts=trial_facts,
            )
            for nep in neg_obs:
                ok, _r, _u = trial_card.check_preconditions(
                    nep.before_frame.to_observed_state_graph(),
                    nep.roles,
                    mask=nep.before_frame.observation_mask,
                )
                if ok:
                    refuted = False
                    contradictions += 1
                    reason = "retained_required_by_training_negative"
                    break

        rep_idx = len(reports)
        reports.append(
            AblationClauseReport(
                action_id=card.action_id,
                clause_kind="required_fact",
                clause=(subj_term, pred, val),
                contrasting_values_tested=tuple(sorted(tested_vals_set)),
                distinct_base_worlds_tested=worlds_tested,
                confirmed_effect_worlds=confirmed_worlds,
                contradiction_count=contradictions,
                refuted=refuted,
                reason=reason,
            )
        )
        if refuted:
            individually_refuted.append((rep_idx, "required_fact", (subj_term, pred, val)))

    if probe_attribute_guards:
        allowed_map = card.allowed_types_map()
        for ag in card.attribute_guards:
            r_types = allowed_map.get(ag.role, frozenset())
            domain_vals = corpus_stats.attr_domain_for_types(r_types, ag.attribute)
            contrasting_vals = [v for v in domain_vals if v != ag.expected_value]
            contrasting_vals.sort(key=lambda x: str(x))
            tested_strs: set[str] = set()
            worlds_tested = 0
            confirmed_worlds = 0
            contradictions = 0

            for base_state, ep in distinct_worlds:
                eid = ep.roles.get(ag.role)
                if eid is None or not contrasting_vals:
                    continue
                ent = base_state.get_entity(eid)
                if ent is None or ent.get(ag.attribute) != ag.expected_value:
                    continue
                worlds_tested += 1
                world_ok = True
                for alt_val in contrasting_vals:
                    tested_strs.add(str(alt_val))
                    cf_state = base_state.with_changes(
                        attribute_edits=((eid, ag.attribute, alt_val),)
                    )
                    act = build_probe_action(card.action_id, ep.roles)
                    cf_after, cf_delta = env.step(cf_state, act)
                    if not _matches_expected_transition(
                        card, cf_state, ep.roles, cf_after, cf_delta
                    ):
                        world_ok = False
                        contradictions += 1
                if world_ok:
                    confirmed_worlds += 1

            refuted = contradictions == 0 and confirmed_worlds >= min_support
            reason = "refuted_incidental_attribute" if refuted else (
                "necessary_attribute_guard_confirmed"
                if contradictions > 0
                else "insufficient_contrasting_support"
            )
            if refuted and neg_obs:
                trial_attrs = tuple(g for g in card.attribute_guards if g != ag)
                trial_card = _clone_card_with_updates(
                    card,
                    version=card.version,
                    parent_version=card.parent_version,
                    attribute_guards=trial_attrs,
                )
                for nep in neg_obs:
                    ok, _r, _u = trial_card.check_preconditions(
                        nep.before_frame.to_observed_state_graph(),
                        nep.roles,
                        mask=nep.before_frame.observation_mask,
                    )
                    if ok:
                        refuted = False
                        contradictions += 1
                        reason = "retained_required_by_training_negative"
                        break
            rep_idx = len(reports)
            reports.append(
                AblationClauseReport(
                    action_id=card.action_id,
                    clause_kind="attribute_guard",
                    clause=ag.to_schema_clause(),
                    contrasting_values_tested=tuple(sorted(tested_strs)),
                    distinct_base_worlds_tested=worlds_tested,
                    confirmed_effect_worlds=confirmed_worlds,
                    contradiction_count=contradictions,
                    refuted=refuted,
                    reason=reason,
                )
            )
            if refuted:
                individually_refuted.append((rep_idx, "attribute_guard", ag))

    if not individually_refuted:
        return (None, tuple(reports))

    # Unified joint refutation verification across BOTH facts and attribute guards + joint neg_obs check
    allowed_map = card.allowed_types_map()
    safe_facts: list[tuple[str, str, str]] = []
    safe_attrs: list[AttributeGuard] = []

    for rep_idx, kind, obj in individually_refuted:
        cand_facts = safe_facts + ([obj] if kind == "required_fact" else [])
        cand_attrs = safe_attrs + ([obj] if kind == "attribute_guard" else [])

        if len(cand_facts) + len(cand_attrs) == 1:
            if kind == "required_fact":
                safe_facts.append(obj)
            else:
                safe_attrs.append(obj)
            continue

        # 1. Joint training-negative consistency check on combined pruned clause set
        neg_conflict = False
        if neg_obs:
            cand_f_set = set(cand_facts)
            cand_a_set = set(cand_attrs)
            trial_card = _clone_card_with_updates(
                card,
                version=card.version,
                parent_version=card.parent_version,
                required_facts=tuple(f for f in card.required_facts if f not in cand_f_set),
                attribute_guards=tuple(g for g in card.attribute_guards if g not in cand_a_set),
            )
            for nep in neg_obs:
                ok, _r, _u = trial_card.check_preconditions(
                    nep.before_frame.to_observed_state_graph(),
                    nep.roles,
                    mask=nep.before_frame.observation_mask,
                )
                if ok:
                    neg_conflict = True
                    break

        if neg_conflict:
            old = reports[rep_idx]
            reports[rep_idx] = AblationClauseReport(
                action_id=old.action_id,
                clause_kind=old.clause_kind,
                clause=old.clause,
                contrasting_values_tested=old.contrasting_values_tested,
                distinct_base_worlds_tested=old.distinct_base_worlds_tested,
                confirmed_effect_worlds=old.confirmed_effect_worlds,
                contradiction_count=old.contradiction_count + 1,
                refuted=False,
                reason="retained_joint_training_negative_conflict",
            )
            continue

        # 2. Simultaneous counterfactual probe flipping ALL candidate facts AND attribute guards
        joint_confirmed = 0
        joint_contra = 0
        for base_state, ep in distinct_worlds:
            del_f: list[tuple[str, str, str]] = []
            add_f: list[tuple[str, str, str]] = []
            attr_edits: list[tuple[str, str, Any]] = []
            applicable = True

            for s_t, p, v in cand_facts:
                eid = ep.roles.get(s_t[1:])
                if eid is None or not base_state.has_fact(eid, p, v):
                    applicable = False
                    break
                etype = ep.before_frame.entity_type_of(eid) or ""
                alts = sorted(
                    (
                        corpus_stats.literal_fact_domain_for_type(etype, p)
                        | corpus_stats.global_literal_fact_domain(p)
                    )
                    - {v}
                )
                if not alts:
                    applicable = False
                    break
                del_f.append((eid, p, v))
                add_f.append((eid, p, alts[0]))

            if applicable:
                for ag_item in cand_attrs:
                    eid = ep.roles.get(ag_item.role)
                    if eid is None:
                        applicable = False
                        break
                    ent = base_state.get_entity(eid)
                    if ent is None or ent.get(ag_item.attribute) != ag_item.expected_value:
                        applicable = False
                        break
                    r_types = allowed_map.get(ag_item.role, frozenset())
                    alt_attr_vals = sorted(
                        (
                            v
                            for v in corpus_stats.attr_domain_for_types(r_types, ag_item.attribute)
                            if v != ag_item.expected_value
                        ),
                        key=lambda x: str(x),
                    )
                    if not alt_attr_vals:
                        applicable = False
                        break
                    attr_edits.append((eid, ag_item.attribute, alt_attr_vals[0]))

            if not applicable:
                continue

            cf_state = base_state.with_changes(
                delete_facts=del_f,
                add_facts=add_f,
                attribute_edits=attr_edits,
            )
            act = build_probe_action(card.action_id, ep.roles)
            cf_after, cf_delta = env.step(cf_state, act)
            if _matches_expected_transition(card, cf_state, ep.roles, cf_after, cf_delta):
                joint_confirmed += 1
            else:
                joint_contra += 1

        if joint_contra == 0 and joint_confirmed >= min_support:
            if kind == "required_fact":
                safe_facts.append(obj)
            else:
                safe_attrs.append(obj)
        else:
            old = reports[rep_idx]
            reports[rep_idx] = AblationClauseReport(
                action_id=old.action_id,
                clause_kind=old.clause_kind,
                clause=old.clause,
                contrasting_values_tested=old.contrasting_values_tested,
                distinct_base_worlds_tested=old.distinct_base_worlds_tested,
                confirmed_effect_worlds=old.confirmed_effect_worlds,
                contradiction_count=old.contradiction_count + max(1, joint_contra),
                refuted=False,
                reason=(
                    "retained_joint_disjunctive_dependency"
                    if joint_contra > 0
                    else "retained_insufficient_joint_support"
                ),
            )

    safe_fact_set = set(safe_facts)
    safe_attr_set = set(safe_attrs)
    if not safe_fact_set and not safe_attr_set:
        return (None, tuple(reports))

    new_ver = next_version if next_version is not None else (card.version + 1)
    revised = _clone_card_with_updates(
        card,
        version=new_ver,
        parent_version=card.version,
        required_facts=tuple(f for f in card.required_facts if f not in safe_fact_set),
        attribute_guards=tuple(g for g in card.attribute_guards if g not in safe_attr_set),
    )
    conf = _recompute_card_confidence(revised, pos_obs, neg_obs)
    revised = _clone_card_with_updates(
        revised,
        version=revised.version,
        parent_version=revised.parent_version,
        confidence=conf,
    )
    return (revised, tuple(reports))


def _serialize_guard_candidate(kind: str, payload: Any) -> dict[str, Any]:
    if kind == "required_fact":
        return {"kind": kind, "clause": list(payload)}
    return {"kind": kind, "clause": list(payload.to_schema_clause()), "spec": payload.to_dict()}


def _extract_discriminating_guards_for_failure(
    active_card: OperatorCard,
    neg_frame: TypedStateFrame,
    neg_roles: Mapping[str, str],
    pos_obs: Sequence[ExtractedEpisodeObservation],
    corpus_stats: CorpusStatistics,
    pos_common_direct: frozenset[tuple[str, str, str]],
    allowed_types: Mapping[str, frozenset[str]],
    refuted_facts: frozenset[tuple[str, str, str]],
) -> set[tuple[str, Any]]:
    """Find all candidate guards across all 5 families that hold on ALL ``pos_obs`` and
    are violated by the over-generalization failure ``(neg_frame, neg_roles)``."""
    non_attr_candidates: set[tuple[str, Any]] = set()
    attr_candidates: set[tuple[str, Any]] = set()

    role_vars = list(active_card.role_variables)
    existing_ineq = {
        (min(g.role_a, g.role_b), max(g.role_a, g.role_b))
        for g in active_card.inequality_guards
    }

    # 1. Missing RoleInequalityGuard (canonically orient delete-role -> add-role when paired in effects)
    effect_role_orientations: dict[frozenset[str], tuple[str, str]] = {}
    for ds, dp, do in active_card.delete_facts:
        for as_, ap, ao in active_card.add_facts:
            if dp != ap:
                continue
            if ds == as_ and do != ao and do.startswith("?") and ao.startswith("?"):
                r_del, r_add = do[1:], ao[1:]
                effect_role_orientations[frozenset({r_del, r_add})] = (r_del, r_add)
            elif do == ao and ds != as_ and ds.startswith("?") and as_.startswith("?"):
                r_del, r_add = ds[1:], as_[1:]
                effect_role_orientations[frozenset({r_del, r_add})] = (r_del, r_add)

    for i in range(len(role_vars)):
        for j in range(i + 1, len(role_vars)):
            r1, r2 = role_vars[i], role_vars[j]
            if not (allowed_types.get(r1, frozenset()) & allowed_types.get(r2, frozenset())):
                continue
            if (min(r1, r2), max(r1, r2)) in existing_ineq:
                continue
            if all(ep.roles.get(r1) != ep.roles.get(r2) for ep in pos_obs):
                if neg_roles.get(r1) is not None and neg_roles.get(r1) == neg_roles.get(r2):
                    ra, rb = effect_role_orientations.get(frozenset({r1, r2}), (r1, r2))
                    non_attr_candidates.add(("inequality", RoleInequalityGuard(role_a=ra, role_b=rb)))

    # 2. Missing direct required_facts (both role-to-role and literal)
    neg_direct = extract_direct_lifted_facts(neg_frame, neg_roles)
    existing_req = set(active_card.required_facts)
    for fact_clause in sorted(pos_common_direct - neg_direct - existing_req - refuted_facts):
        non_attr_candidates.add(("required_fact", fact_clause))

    # 3. Missing ConditionalRoleFactGuard (subtype-specific invariants for multi-type roles)
    existing_cond = set(active_card.conditional_fact_guards)
    for rname in role_vars:
        r_types = allowed_types.get(rname, frozenset())
        if len(r_types) <= 1:
            continue
        neg_eid = neg_roles.get(rname)
        if neg_eid is None:
            continue
        neg_etype = neg_frame.entity_type_of(neg_eid)
        if neg_etype not in r_types:
            continue

        uncond_preds = {pred for subj, pred, _obj in pos_common_direct if subj == f"?{rname}"}
        preds_by_subtype: dict[str, set[str]] = {}
        invariants_by_subtype: dict[str, set[tuple[str, str]]] = {}
        for subtype in sorted(r_types):
            sub_eps = [
                ep
                for ep in pos_obs
                if ep.before_frame.entity_type_of(ep.roles[rname]) == subtype
            ]
            if not sub_eps:
                continue
            lit_sets = [
                extract_entity_literal_facts(ep.before_frame, ep.roles[rname]) for ep in sub_eps
            ]
            inter = set(lit_sets[0])
            for ls in lit_sets[1:]:
                inter &= ls
            invariants_by_subtype[subtype] = inter
            preds_by_subtype[subtype] = {p for ls in lit_sets for p, _v in ls}

        if neg_etype not in invariants_by_subtype:
            continue
        other_common_preds: set[str] | None = None
        for other_t, other_ps in preds_by_subtype.items():
            if other_t == neg_etype:
                continue
            other_common_preds = (
                set(other_ps) if other_common_preds is None else (other_common_preds & other_ps)
            )
        shared_by_others = other_common_preds or set()
        neg_lits = extract_entity_literal_facts(neg_frame, neg_eid)
        for pred, val in sorted(invariants_by_subtype[neg_etype]):
            if pred in uncond_preds or pred in shared_by_others:
                continue
            if (pred, val) not in neg_lits:
                cg = ConditionalRoleFactGuard(
                    role=rname,
                    when_entity_type=neg_etype,
                    predicate=pred,
                    value=val,
                )
                if cg not in existing_cond:
                    non_attr_candidates.add(("conditional_fact", cg))

    # 4. Missing IndirectRoleFactGuard (1-hop neighbor invariants)
    existing_ind = set(active_card.indirect_fact_guards)
    pos_links_by_key: dict[tuple[str, str], list[tuple[ExtractedEpisodeObservation, str, str]]] = {}
    for pep in pos_obs:
        for link in extract_one_hop_links(pep.before_frame, pep.roles):
            pos_links_by_key.setdefault((link.role, link.via_predicate), []).append(
                (pep, link.neighbor_id, link.neighbor_type)
            )
    for nlink in extract_one_hop_links(neg_frame, neg_roles):
        key = (nlink.role, nlink.via_predicate)
        all_pos_for_key = pos_links_by_key.get(key, [])
        same_type_pos = [
            (pep, z_id) for pep, z_id, z_t in all_pos_for_key if z_t == nlink.neighbor_type
        ]
        if not same_type_pos:
            continue
        pos_lit_sets = [
            extract_entity_literal_facts(pep.before_frame, z_id) for pep, z_id in same_type_pos
        ]
        pos_invs = set(pos_lit_sets[0])
        for ls in pos_lit_sets[1:]:
            pos_invs &= ls

        other_type_preds: set[str] = set()
        for pep, z_id, z_t in all_pos_for_key:
            if z_t != nlink.neighbor_type:
                for p, _v in extract_entity_literal_facts(pep.before_frame, z_id):
                    other_type_preds.add(p)

        cand_invs = {(p, v) for p, v in pos_invs if p not in other_type_preds} or pos_invs
        neg_n_lits = extract_entity_literal_facts(neg_frame, nlink.neighbor_id)
        for pred, val in sorted(cand_invs):
            if (pred, val) not in neg_n_lits:
                ig = IndirectRoleFactGuard(
                    role=nlink.role,
                    via_predicate=nlink.via_predicate,
                    when_neighbor_type=nlink.neighbor_type,
                    predicate=pred,
                    value=val,
                )
                if ig not in existing_ind:
                    non_attr_candidates.add(("indirect_fact", ig))

    # 5. Missing AttributeGuard (with MDL functional-entailment & decorative-attribute filtering)
    existing_attrs = set(active_card.attribute_guards)
    pos_attr_maps = [extract_role_attribute_map(ep.before_frame, ep.roles) for ep in pos_obs]
    for rname in role_vars:
        common_keys: set[str] | None = None
        for amap in pos_attr_maps:
            ks = set(amap.get(rname, {}).keys())
            common_keys = ks if common_keys is None else (common_keys & ks)
        if not common_keys:
            continue
        neg_eid = neg_roles.get(rname)
        if neg_eid is None:
            continue
        nef = neg_frame.get_entity(neg_eid)
        if nef is None or not nef.attributes_observed:
            continue

        r_types = allowed_types.get(rname, frozenset())
        role_lit_preconds = frozenset(
            (pred, obj)
            for subj, pred, obj in pos_common_direct
            if subj == f"?{rname}" and not obj.startswith("?")
        )
        for attr_name in sorted(common_keys):
            pos_vals = {amap[rname][attr_name] for amap in pos_attr_maps}
            if len(pos_vals) != 1:
                continue
            expected_val = next(iter(pos_vals))
            if nef.get_attr(attr_name) == expected_val:
                continue
            # Prune attributes 100% functionally entailed by positive literal preconditions
            if _is_attr_functionally_entailed(
                r_types, attr_name, expected_val, role_lit_preconds, corpus_stats
            ):
                continue
            # If a structural/fact guard already explains this failure, skip unedited decorative attrs
            if non_attr_candidates and not corpus_stats.is_attr_edited_on_types(r_types, attr_name):
                other_varying = corpus_stats.other_varying_pos_role_slots(
                    r_types,
                    attr_name,
                    exclude_action=active_card.action_id,
                    exclude_role=rname,
                )
                if other_varying:
                    continue
            ag = AttributeGuard(role=rname, attribute=attr_name, expected_value=expected_val)
            if ag not in existing_attrs:
                attr_candidates.add(("attribute", ag))

    return non_attr_candidates | attr_candidates


def _merge_guards_into_card(
    base: OperatorCard,
    guards: Sequence[tuple[str, Any]],
    *,
    version: int,
    parent_version: int,
    support_negative: int | None = None,
) -> OperatorCard:
    req_facts = set(base.required_facts)
    cond_guards = set(base.conditional_fact_guards)
    ind_guards = set(base.indirect_fact_guards)
    attr_guards = set(base.attribute_guards)
    ineq_guards = set(base.inequality_guards)

    for kind, payload in guards:
        if kind == "required_fact":
            req_facts.add(payload)
        elif kind == "conditional_fact":
            cond_guards.add(payload)
        elif kind == "indirect_fact":
            ind_guards.add(payload)
        elif kind == "attribute":
            attr_guards.add(payload)
        elif kind == "inequality":
            ineq_guards.add(payload)

    return _clone_card_with_updates(
        base,
        version=version,
        parent_version=parent_version,
        required_facts=tuple(sorted(req_facts)),
        conditional_fact_guards=tuple(
            sorted(
                cond_guards,
                key=lambda g: (g.role, g.when_entity_type, g.predicate, g.value),
            )
        ),
        indirect_fact_guards=tuple(
            sorted(
                ind_guards,
                key=lambda g: (
                    g.role,
                    g.via_predicate,
                    g.when_neighbor_type,
                    g.predicate,
                    g.value,
                ),
            )
        ),
        attribute_guards=tuple(
            sorted(attr_guards, key=lambda g: (g.role, g.attribute, str(g.expected_value)))
        ),
        inequality_guards=tuple(sorted(ineq_guards, key=lambda g: (g.role_a, g.role_b))),
        support_negative=support_negative,
    )


def acquire_guards_from_contrasts(
    action_id: str,
    version_store: OperatorVersionStore,
    pos_obs: Sequence[ExtractedEpisodeObservation],
    contrast_probes: Sequence[tuple[StateGraph, ActionInvocation]],
    corpus_stats: CorpusStatistics,
    env: TransitionEnvironment,
    *,
    min_guard_support: int = 2,
    neg_obs: Sequence[ExtractedEpisodeObservation] = (),
    refuted_facts: frozenset[tuple[str, str, str]] = frozenset(),
    created_at: str = DEFAULT_VERSION_TIMESTAMP,
) -> GuardAcquisitionReport:
    """Acquire missing guards from over-generalizing contrast probes with repeated-evidence gating.

    - Runs every probe against ``env.step`` and compares against the currently active card.
    - Deduplicates over-generalizations by canonical state-action key so repeating an
      identical world cannot inflate evidence count.
    - ONLY promotes a guard when ``>= min_guard_support`` (default ``2``) independent
      contrast failures support it.
    - Any 1-shot single failure (support ``< min_guard_support``) registers a ``shadow``
      version ONLY and NEVER promotes it!
    """
    active_rec = version_store.get_active(action_id)
    if active_rec is None:
        raise VersionStoreError(f"No active promoted OperatorCard for action_id {action_id!r}")
    active_card = active_rec.card
    initial_version = active_card.version

    if not contrast_probes or not pos_obs:
        return GuardAcquisitionReport(
            action_id=action_id,
            initial_version=initial_version,
            active_version_after=initial_version,
            shadow_version_registered=None,
            promoted=False,
            total_probes=len(contrast_probes),
            independent_overgeneralizations=0,
            min_guard_support=min_guard_support,
            acquired_guards=(),
            shadow_only_guards=(),
            contrast_pass_rate_before=1.0,
            contrast_pass_rate_after=1.0,
            reason="no_probes_or_positives",
        )

    probe_outcomes: list[tuple[StateGraph, ActionInvocation, StateGraph, SparseDelta]] = []
    pre_matches = 0
    independent_failures: dict[str, tuple[TypedStateFrame, dict[str, str]]] = {}

    for state, act in contrast_probes:
        if act.action_id != action_id:
            continue
        roles = act.role_dict()
        pred = active_card.apply(state, roles)
        env_after, env_delta = env.step(state, act)
        probe_outcomes.append((state, act, env_after, env_delta))
        if (
            (not pred.abstained)
            and (pred.after_state == env_after)
            and (pred.applied == (not env_delta.is_noop))
        ):
            pre_matches += 1
        if pred.applied and env_delta.is_noop:
            key = _state_action_fingerprint(state, action_id, roles)
            if key not in independent_failures:
                independent_failures[key] = (TypedStateFrame.from_state_graph(state), roles)

    total_checked = len(probe_outcomes)
    pre_rate = pre_matches / float(total_checked) if total_checked > 0 else 1.0

    if not independent_failures:
        return GuardAcquisitionReport(
            action_id=action_id,
            initial_version=initial_version,
            active_version_after=initial_version,
            shadow_version_registered=None,
            promoted=False,
            total_probes=total_checked,
            independent_overgeneralizations=0,
            min_guard_support=min_guard_support,
            acquired_guards=(),
            shadow_only_guards=(),
            contrast_pass_rate_before=pre_rate,
            contrast_pass_rate_after=pre_rate,
            reason="no_overgeneralization_detected",
        )

    direct_sets = [extract_direct_lifted_facts(ep.before_frame, ep.roles) for ep in pos_obs]
    pos_common_direct = frozenset.intersection(*direct_sets) if direct_sets else frozenset()
    allowed_types = active_card.allowed_types_map()

    support_by_guard: dict[tuple[str, Any], set[str]] = {}
    for fail_key, (neg_frame, neg_roles) in sorted(independent_failures.items()):
        cands = _extract_discriminating_guards_for_failure(
            active_card,
            neg_frame,
            neg_roles,
            pos_obs,
            corpus_stats,
            pos_common_direct,
            allowed_types,
            refuted_facts,
        )
        for cand in cands:
            support_by_guard.setdefault(cand, set()).add(fail_key)

    supported = [
        (kind, payload)
        for (kind, payload), keys in sorted(
            support_by_guard.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))
        )
        if len(keys) >= min_guard_support
    ]
    shadow_only = [
        (kind, payload)
        for (kind, payload), keys in sorted(
            support_by_guard.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))
        )
        if 0 < len(keys) < min_guard_support
    ]

    if supported:
        new_ver = version_store.next_version(action_id)
        candidate_card = _merge_guards_into_card(
            active_card,
            supported,
            version=new_ver,
            parent_version=initial_version,
            support_negative=active_card.support_negative + len(independent_failures),
        )
        # Verify 100% preservation on all positive observations
        pos_ok = all(
            candidate_card.apply(
                ep.before_frame.to_observed_state_graph(),
                ep.roles,
                mask=ep.before_frame.observation_mask,
            ).delta
            == ep.sparse_delta
            for ep in pos_obs
        )
        post_matches = sum(
            1
            for state, act, env_after, env_delta in probe_outcomes
            if (
                not (res := candidate_card.apply(state, act.role_dict())).abstained
                and res.after_state == env_after
                and res.applied == (not env_delta.is_noop)
            )
        )
        post_rate = post_matches / float(total_checked) if total_checked > 0 else 1.0
        conf = _recompute_card_confidence(candidate_card, pos_obs, neg_obs)
        candidate_card = _clone_card_with_updates(
            candidate_card,
            version=candidate_card.version,
            parent_version=candidate_card.parent_version,
            confidence=conf,
        )

        ev = {
            "source": "s3_repeated_contrast_guard_acquisition",
            "independent_overgeneralizations": len(independent_failures),
            "min_guard_support": min_guard_support,
            "acquired_guards": [_serialize_guard_candidate(k, p) for k, p in supported],
            "shadow_only_guards_excluded": [
                _serialize_guard_candidate(k, p) for k, p in shadow_only
            ],
            "positive_preservation_verified": pos_ok,
            "contrast_pass_rate_before": round(pre_rate, 6),
            "contrast_pass_rate_after": round(post_rate, 6),
        }
        version_store.register_shadow(
            candidate_card,
            evidence=ev,
            contrast_pass_rate=post_rate,
            created_at=created_at,
            notes="candidate_guard_repair",
        )
        if pos_ok and post_rate > pre_rate:
            version_store.promote(
                action_id,
                candidate_card.version,
                evidence={"promotion_verified": True},
                contrast_pass_rate=post_rate,
                notes="promoted_repeated_evidence_guard_repair",
                created_at=created_at,
            )
            return GuardAcquisitionReport(
                action_id=action_id,
                initial_version=initial_version,
                active_version_after=candidate_card.version,
                shadow_version_registered=candidate_card.version,
                promoted=True,
                total_probes=total_checked,
                independent_overgeneralizations=len(independent_failures),
                min_guard_support=min_guard_support,
                acquired_guards=tuple(_serialize_guard_candidate(k, p) for k, p in supported),
                shadow_only_guards=tuple(_serialize_guard_candidate(k, p) for k, p in shadow_only),
                contrast_pass_rate_before=pre_rate,
                contrast_pass_rate_after=post_rate,
                reason="promoted_with_repeated_evidence",
            )
        return GuardAcquisitionReport(
            action_id=action_id,
            initial_version=initial_version,
            active_version_after=initial_version,
            shadow_version_registered=candidate_card.version,
            promoted=False,
            total_probes=total_checked,
            independent_overgeneralizations=len(independent_failures),
            min_guard_support=min_guard_support,
            acquired_guards=(),
            shadow_only_guards=tuple(_serialize_guard_candidate(k, p) for k, p in supported),
            contrast_pass_rate_before=pre_rate,
            contrast_pass_rate_after=pre_rate,
            reason="shadow_only_failed_positive_or_pass_rate_check",
        )

    if shadow_only:
        new_ver = version_store.next_version(action_id)
        shadow_card = _merge_guards_into_card(
            active_card,
            shadow_only,
            version=new_ver,
            parent_version=initial_version,
            support_negative=active_card.support_negative + len(independent_failures),
        )
        shadow_matches = sum(
            1
            for state, act, env_after, env_delta in probe_outcomes
            if (
                not (res := shadow_card.apply(state, act.role_dict())).abstained
                and res.after_state == env_after
                and res.applied == (not env_delta.is_noop)
            )
        )
        shadow_rate = shadow_matches / float(total_checked) if total_checked > 0 else 1.0
        version_store.register_shadow(
            shadow_card,
            evidence={
                "source": "s3_one_shot_or_under_supported_contrast",
                "independent_overgeneralizations": len(independent_failures),
                "min_guard_support": min_guard_support,
                "promotion_blocked_reason": "insufficient_repeated_evidence",
                "shadow_only_guards": [_serialize_guard_candidate(k, p) for k, p in shadow_only],
            },
            contrast_pass_rate=shadow_rate,
            created_at=created_at,
            notes="shadow_only_insufficient_repeated_evidence",
        )
        return GuardAcquisitionReport(
            action_id=action_id,
            initial_version=initial_version,
            active_version_after=initial_version,
            shadow_version_registered=shadow_card.version,
            promoted=False,
            total_probes=total_checked,
            independent_overgeneralizations=len(independent_failures),
            min_guard_support=min_guard_support,
            acquired_guards=(),
            shadow_only_guards=tuple(_serialize_guard_candidate(k, p) for k, p in shadow_only),
            contrast_pass_rate_before=pre_rate,
            contrast_pass_rate_after=pre_rate,
            reason="shadow_only_insufficient_repeated_evidence",
        )

    return GuardAcquisitionReport(
        action_id=action_id,
        initial_version=initial_version,
        active_version_after=initial_version,
        shadow_version_registered=None,
        promoted=False,
        total_probes=total_checked,
        independent_overgeneralizations=len(independent_failures),
        min_guard_support=min_guard_support,
        acquired_guards=(),
        shadow_only_guards=(),
        contrast_pass_rate_before=pre_rate,
        contrast_pass_rate_after=pre_rate,
        reason="no_discriminating_guard_found",
    )


def generate_domain_agnostic_contrast_probes(
    action_id: str,
    card: OperatorCard,
    pos_obs: Sequence[ExtractedEpisodeObservation],
    corpus_stats: CorpusStatistics,
) -> list[tuple[StateGraph, ActionInvocation]]:
    """Synthesize domain-agnostic single-factor contrast probes from ``pos_obs`` and ``corpus_stats``.

    Covers all 6 structural perturbation families without any domain literals:
    1. Role equality (type-compatible role pairs bound to the same entity)
    2. Role-to-role relation swap (substituting an entity in a direct lifted role link)
    3. Varying entity attributes (flipping each observed attribute on bound roles)
    4. Direct literal role facts (flipping each observed literal fact on bound roles)
    5. Multi-type role subtype invariants (retargeting a role onto each subtype and flipping
       each subtype-specific literal fact one at a time across distinct base worlds)
    6. 1-hop neighbor literal facts (relinking a role onto each observed 1-hop neighbor
       type and flipping each neighbor-specific literal fact one at a time across distinct worlds)
    """
    distinct_worlds = _distinct_positive_worlds(pos_obs)
    if not distinct_worlds:
        return []

    allowed_types = card.allowed_types_map()
    role_vars = list(card.role_variables)
    direct_sets = [extract_direct_lifted_facts(ep.before_frame, ep.roles) for ep in pos_obs]
    pos_common_direct = frozenset.intersection(*direct_sets) if direct_sets else frozenset()

    probes: list[tuple[StateGraph, ActionInvocation]] = []
    seen_keys: set[str] = set()

    def _add_probe(state: StateGraph, roles: Mapping[str, str]) -> None:
        key = _state_action_fingerprint(state, action_id, roles)
        if key not in seen_keys:
            seen_keys.add(key)
            probes.append((state, build_probe_action(action_id, roles)))

    # 1. Role equality probes (?r1 == ?r2 in both binding directions)
    for i in range(len(role_vars)):
        for j in range(i + 1, len(role_vars)):
            r1, r2 = role_vars[i], role_vars[j]
            if not (allowed_types.get(r1, frozenset()) & allowed_types.get(r2, frozenset())):
                continue
            for base_state, ep in distinct_worlds:
                roles_eq_1 = dict(ep.roles)
                roles_eq_1[r2] = roles_eq_1[r1]
                _add_probe(base_state, roles_eq_1)
                roles_eq_2 = dict(ep.roles)
                roles_eq_2[r1] = roles_eq_2[r2]
                _add_probe(base_state, roles_eq_2)

    # 2. Direct role-to-role fact swap probes
    for subj_t, _pred, obj_t in sorted(pos_common_direct):
        if not (subj_t.startswith("?") and obj_t.startswith("?")):
            continue
        r_obj = obj_t[1:]
        r_obj_types = allowed_types.get(r_obj, frozenset())
        for base_state, ep in distinct_worlds:
            cur_eid = ep.roles.get(r_obj)
            for cand_eid, cand_ent in base_state.entities:
                if cand_eid != cur_eid and cand_ent.entity_type in r_types_or_any(r_obj_types, cand_ent.entity_type):
                    if cand_eid in ep.roles.values():
                        continue
                    roles_sw = dict(ep.roles)
                    roles_sw[r_obj] = cand_eid
                    _add_probe(base_state, roles_sw)
                    break

    # 3. Varying attribute probes on bound roles
    for rname in role_vars:
        r_types = allowed_types.get(rname, frozenset())
        for base_state, ep in distinct_worlds:
            eid = ep.roles.get(rname)
            if eid is None:
                continue
            ent = base_state.get_entity(eid)
            if ent is None:
                continue
            for ak, av in ent.attributes:
                domain_vals = corpus_stats.attr_domain_for_types(r_types, ak)
                for alt_v in sorted((v for v in domain_vals if v != av), key=lambda x: str(x)):
                    cf_state = base_state.with_changes(attribute_edits=((eid, ak, alt_v),))
                    _add_probe(cf_state, ep.roles)

    # 4. Direct literal fact probes on bound roles
    for rname in role_vars:
        for base_state, ep in distinct_worlds:
            eid = ep.roles.get(rname)
            if eid is None:
                continue
            etype = ep.before_frame.entity_type_of(eid) or ""
            for pred, val in sorted(extract_entity_literal_facts(ep.before_frame, eid)):
                domain_vals = (
                    corpus_stats.literal_fact_domain_for_type(etype, pred)
                    | corpus_stats.global_literal_fact_domain(pred)
                )
                for alt_v in sorted(v for v in domain_vals if v != val):
                    cf_state = base_state.with_changes(
                        delete_facts=((eid, pred, val),),
                        add_facts=((eid, pred, alt_v),),
                    )
                    _add_probe(cf_state, ep.roles)

    # 5. Multi-type role subtype-targeted probes across distinct base worlds
    for rname in role_vars:
        r_types = allowed_types.get(rname, frozenset())
        if len(r_types) <= 1:
            continue
        preds_by_sub: dict[str, set[str]] = {}
        invs_by_sub: dict[str, set[tuple[str, str]]] = {}
        for sub_t in sorted(r_types):
            sub_eps = [
                ep for ep in pos_obs if ep.before_frame.entity_type_of(ep.roles[rname]) == sub_t
            ]
            if not sub_eps:
                continue
            lsets = [
                extract_entity_literal_facts(ep.before_frame, ep.roles[rname]) for ep in sub_eps
            ]
            inter = set(lsets[0])
            for ls in lsets[1:]:
                inter &= ls
            invs_by_sub[sub_t] = inter
            preds_by_sub[sub_t] = {p for ls in lsets for p, _v in ls}

        for sub_t, sub_invs in sorted(invs_by_sub.items()):
            other_preds: set[str] | None = None
            for ot, ops in preds_by_sub.items():
                if ot == sub_t:
                    continue
                other_preds = set(ops) if other_preds is None else (other_preds & ops)
            shared_others = other_preds or set()
            sub_specific = sorted((p, v) for p, v in sub_invs if p not in shared_others)
            if not sub_specific:
                continue

            for base_state, ep in distinct_worlds:
                old_eid = ep.roles[rname]
                old_ent = base_state.get_entity(old_eid)
                work_state = base_state
                work_roles = dict(ep.roles)

                if old_ent is not None and old_ent.entity_type == sub_t:
                    target_eid = old_eid
                else:
                    # Rebind rname to an existing unbound entity of subtype sub_t in this world
                    bound_other = {v for k, v in work_roles.items() if k != rname}
                    cand_eid = next(
                        (
                            eid
                            for eid, ent in base_state.entities
                            if ent.entity_type == sub_t and eid not in bound_other
                        ),
                        None,
                    )
                    if cand_eid is None:
                        continue
                    target_eid = cand_eid
                    work_roles[rname] = cand_eid
                    # Transfer incoming role-to-role relational facts from old_eid -> cand_eid
                    del_f = []
                    add_f = []
                    for s_f, p_f, o_f in base_state.facts:
                        if o_f == old_eid and s_f in bound_other:
                            del_f.append((s_f, p_f, old_eid))
                            add_f.append((s_f, p_f, cand_eid))
                    # Normalize cand_eid to satisfy all positive subtype invariants first
                    for inv_p, inv_v in sorted(sub_invs):
                        for ex_f in base_state.facts_for(cand_eid, inv_p):
                            del_f.append(ex_f)
                        add_f.append((cand_eid, inv_p, inv_v))
                    work_state = base_state.with_changes(delete_facts=del_f, add_facts=add_f)

                for pred, val in sub_specific:
                    domain_vals = (
                        corpus_stats.literal_fact_domain_for_type(sub_t, pred)
                        | corpus_stats.global_literal_fact_domain(pred)
                    )
                    for alt_v in sorted(v for v in domain_vals if v != val):
                        del_cur = list(work_state.facts_for(target_eid, pred))
                        cf_state = work_state.with_changes(
                            delete_facts=del_cur,
                            add_facts=((target_eid, pred, alt_v),),
                        )
                        _add_probe(cf_state, work_roles)

    # 6. 1-hop neighbor literal fact probes across distinct base worlds
    pos_links_by_key: dict[tuple[str, str], list[tuple[ExtractedEpisodeObservation, str, str]]] = {}
    for pep in pos_obs:
        for link in extract_one_hop_links(pep.before_frame, pep.roles):
            pos_links_by_key.setdefault((link.role, link.via_predicate), []).append(
                (pep, link.neighbor_id, link.neighbor_type)
            )
    for (rname, via_pred), link_entries in sorted(pos_links_by_key.items()):
        neighbor_types = sorted({nt for _p, _nid, nt in link_entries})
        for ntype in neighbor_types:
            same_t = [(pep, nid) for pep, nid, nt in link_entries if nt == ntype]
            lsets = [extract_entity_literal_facts(pep.before_frame, nid) for pep, nid in same_t]
            if not lsets:
                continue
            pos_invs = set(lsets[0])
            for ls in lsets[1:]:
                pos_invs &= ls
            other_preds = {
                p
                for pep, nid, nt in link_entries
                if nt != ntype
                for p, _v in extract_entity_literal_facts(pep.before_frame, nid)
            }
            target_invs = sorted((p, v) for p, v in pos_invs if p not in other_preds)
            if not target_invs:
                continue

            for base_state, ep in distinct_worlds:
                role_eid = ep.roles.get(rname)
                if role_eid is None:
                    continue
                cur_links = base_state.facts_for(role_eid, via_pred)
                bound_vals = set(ep.roles.values())
                work_state = base_state
                neighbor_eid: str | None = None
                for _s, _p, o_id in cur_links:
                    nent = base_state.get_entity(o_id)
                    if nent is not None and nent.entity_type == ntype and o_id not in bound_vals:
                        neighbor_eid = o_id
                        break
                if neighbor_eid is None:
                    cand_n = next(
                        (
                            eid
                            for eid, ent in base_state.entities
                            if ent.entity_type == ntype and eid not in bound_vals
                        ),
                        None,
                    )
                    if cand_n is None:
                        continue
                    neighbor_eid = cand_n
                    del_f = list(cur_links)
                    add_f = [(role_eid, via_pred, cand_n)]
                    for inv_p, inv_v in sorted(pos_invs):
                        del_f.extend(base_state.facts_for(cand_n, inv_p))
                        add_f.append((cand_n, inv_p, inv_v))
                    work_state = base_state.with_changes(delete_facts=del_f, add_facts=add_f)

                for pred, val in target_invs:
                    domain_vals = (
                        corpus_stats.literal_fact_domain_for_type(ntype, pred)
                        | corpus_stats.global_literal_fact_domain(pred)
                    )
                    for alt_v in sorted(v for v in domain_vals if v != val):
                        del_cur = list(work_state.facts_for(neighbor_eid, pred))
                        cf_state = work_state.with_changes(
                            delete_facts=del_cur,
                            add_facts=((neighbor_eid, pred, alt_v),),
                        )
                        _add_probe(cf_state, ep.roles)

    return probes


def r_types_or_any(allowed: frozenset[str], candidate_type: str) -> frozenset[str]:
    return allowed if allowed else frozenset({candidate_type})


def _execute_card_sequence(
    cards: Mapping[str, OperatorCard],
    initial_state: StateGraph,
    actions: Sequence[ActionInvocation],
    mask: ObservationMask | None = None,
) -> tuple[StateGraph | None, tuple[StateGraph | None, ...], bool]:
    cur = initial_state.copy()
    steps: list[StateGraph | None] = []
    for act in actions:
        card = cards.get(act.action_id)
        if card is None:
            return (None, tuple(steps), True)
        res = card.apply(cur, act.role_dict(), mask=mask)
        if res.abstained or res.after_state is None:
            return (None, tuple(steps), True)
        cur = res.after_state
        steps.append(cur)
    return (cur, tuple(steps), False)


def _execute_env_sequence(
    env: TransitionEnvironment,
    initial_state: StateGraph,
    actions: Sequence[ActionInvocation],
    mask: ObservationMask | None = None,
) -> tuple[StateGraph, tuple[StateGraph, ...]]:
    cur = initial_state.copy()
    steps: list[StateGraph] = []
    for act in actions:
        cur, _delta = env.step(cur, act, mask=mask)
        steps.append(cur)
    return (cur, tuple(steps))


def _summarize_probe_evaluations(
    category: str,
    records: Sequence[tuple[StateGraph | None, tuple[StateGraph | None, ...], bool, StateGraph, tuple[StateGraph, ...]]],
    *,
    non_commuting_pairs_detected: int = 0,
    total_pairs: int = 0,
    details: Sequence[dict[str, Any]] = (),
) -> ProbeCategorySummary:
    total = len(records)
    if total == 0:
        return ProbeCategorySummary(
            category=category,
            total_probes=0,
            exact_final_matches=0,
            exact_final_accuracy=0.0,
            total_steps=0,
            intermediate_step_matches=0,
            intermediate_step_accuracy=0.0,
            all_steps_exact_matches=0,
            all_steps_exact_accuracy=0.0,
            non_commuting_pairs_detected=non_commuting_pairs_detected,
            total_pairs=total_pairs,
            passed=False,
            details=tuple(details),
        )

    final_ok = 0
    total_steps = 0
    step_ok = 0
    all_steps_ok = 0
    for pred_final, pred_steps, abstained, gold_final, gold_steps in records:
        if (not abstained) and pred_final == gold_final:
            final_ok += 1
        n_g = len(gold_steps)
        total_steps += n_g
        if (not abstained) and len(pred_steps) == n_g:
            matched_here = sum(1 for ps, gs in zip(pred_steps, gold_steps) if ps == gs)
            step_ok += matched_here
            if matched_here == n_g:
                all_steps_ok += 1
        elif not abstained:
            step_ok += sum(1 for ps, gs in zip(pred_steps, gold_steps) if ps == gs)

    final_acc = final_ok / float(total)
    step_acc = step_ok / float(total_steps) if total_steps > 0 else 0.0
    all_acc = all_steps_ok / float(total)
    return ProbeCategorySummary(
        category=category,
        total_probes=total,
        exact_final_matches=final_ok,
        exact_final_accuracy=final_acc,
        total_steps=total_steps,
        intermediate_step_matches=step_ok,
        intermediate_step_accuracy=step_acc,
        all_steps_exact_matches=all_steps_ok,
        all_steps_exact_accuracy=all_acc,
        non_commuting_pairs_detected=non_commuting_pairs_detected,
        total_pairs=total_pairs,
        passed=(final_ok == total and step_ok == total_steps),
        details=tuple(details[:8]),
    )


def run_role_swap_probes(
    cards: Mapping[str, OperatorCard],
    grouped_pos_obs: Mapping[str, Sequence[ExtractedEpisodeObservation]],
    env: TransitionEnvironment,
) -> ProbeCategorySummary:
    """Domain-agnostic role-swap probes for every operator with type-compatible swappable roles."""
    eval_records = []
    details: list[dict[str, Any]] = []
    for action_id, card in sorted(cards.items()):
        if not card.inequality_guards:
            continue
        worlds = _distinct_positive_worlds(grouped_pos_obs.get(action_id, ()))
        for ig in card.inequality_guards:
            for base_state, ep in worlds:
                fwd = build_probe_action(action_id, ep.roles)
                swapped_roles = dict(ep.roles)
                swapped_roles[ig.role_a], swapped_roles[ig.role_b] = (
                    swapped_roles[ig.role_b],
                    swapped_roles[ig.role_a],
                )
                rev = build_probe_action(action_id, swapped_roles)
                # 1. Forward then swapped reverse (2-step round-trip)
                g_fin, g_steps = _execute_env_sequence(env, base_state, (fwd, rev))
                p_fin, p_steps, abst = _execute_card_sequence(cards, base_state, (fwd, rev))
                eval_records.append((p_fin, p_steps, abst, g_fin, g_steps))
                # 2. Swapped roles directly on initial state (contrast guard)
                g_fin2, g_steps2 = _execute_env_sequence(env, base_state, (rev,))
                p_fin2, p_steps2, abst2 = _execute_card_sequence(cards, base_state, (rev,))
                eval_records.append((p_fin2, p_steps2, abst2, g_fin2, g_steps2))
                if len(details) < 6:
                    details.append(
                        {
                            "action_id": action_id,
                            "swapped_roles": [ig.role_a, ig.role_b],
                            "roundtrip_exact": (not abst and p_fin == g_fin),
                            "wrong_dir_exact": (not abst2 and p_fin2 == g_fin2),
                        }
                    )
    return _summarize_probe_evaluations("role_swap_probes", eval_records, details=details)


def _bind_compatible_action(
    card_b: OperatorCard,
    state: StateGraph,
    seed_roles: Mapping[str, str],
) -> ActionInvocation | None:
    """Domain-agnostically bind ``card_b``'s roles onto entities overlapping ``seed_roles``."""
    allowed_b = card_b.allowed_types_map()
    seed_eids_by_type: dict[str, list[str]] = {}
    for _r, eid in sorted(seed_roles.items()):
        ent = state.get_entity(eid)
        if ent is not None:
            seed_eids_by_type.setdefault(ent.entity_type, []).append(eid)

    all_eids_by_type: dict[str, list[str]] = {}
    for eid, ent in state.entities:
        all_eids_by_type.setdefault(ent.entity_type, []).append(eid)

    # First try exact role-name match when entity types are compatible
    bound: dict[str, str] = {}
    used: set[str] = set()
    for rname in card_b.role_variables:
        req_ts = allowed_b.get(rname, frozenset())
        if rname in seed_roles:
            cand = seed_roles[rname]
            ent = state.get_entity(cand)
            if ent is not None and (not req_ts or ent.entity_type in req_ts):
                bound[rname] = cand
                used.add(cand)

    # Fill remaining roles preferring entities already in seed_roles, then any matching world entity
    for rname in card_b.role_variables:
        if rname in bound:
            continue
        req_ts = allowed_b.get(rname, frozenset())
        chosen: str | None = None
        for et in sorted(req_ts) if req_ts else sorted(all_eids_by_type):
            for cand in seed_eids_by_type.get(et, ()):
                if cand not in used:
                    chosen = cand
                    break
            if chosen is not None:
                break
        if chosen is None:
            for et in sorted(req_ts) if req_ts else sorted(all_eids_by_type):
                for cand in all_eids_by_type.get(et, ()):
                    if cand not in used:
                        chosen = cand
                        break
                if chosen is not None:
                    break
        if chosen is None:
            return None
        bound[rname] = chosen
        used.add(chosen)

    return build_probe_action(card_b.action_id, bound)


def run_commutator_order_probes(
    cards: Mapping[str, OperatorCard],
    grouped_pos_obs: Mapping[str, Sequence[ExtractedEpisodeObservation]],
    env: TransitionEnvironment,
) -> ProbeCategorySummary:
    """Domain-agnostic non-commuting action-order ``(a;b)`` vs ``(b;a)`` commutator probes."""
    eval_records = []
    details: list[dict[str, Any]] = []
    action_ids = sorted(cards.keys())
    total_pairs = 0
    non_commuting_detected = 0

    for i in range(len(action_ids)):
        for j in range(len(action_ids)):
            if i == j:
                continue
            aid_a, aid_b = action_ids[i], action_ids[j]
            card_b = cards[aid_b]
            worlds_a = _distinct_positive_worlds(grouped_pos_obs.get(aid_a, ()))
            for base_state, ep in worlds_a[:2]:
                act_a = build_probe_action(aid_a, ep.roles)
                act_b = _bind_compatible_action(card_b, base_state, ep.roles)
                if act_b is None:
                    continue
                total_pairs += 1
                g_ab, g_ab_steps = _execute_env_sequence(env, base_state, (act_a, act_b))
                g_ba, g_ba_steps = _execute_env_sequence(env, base_state, (act_b, act_a))
                p_ab, p_ab_steps, abst_ab = _execute_card_sequence(cards, base_state, (act_a, act_b))
                p_ba, p_ba_steps, abst_ba = _execute_card_sequence(cards, base_state, (act_b, act_a))
                eval_records.append((p_ab, p_ab_steps, abst_ab, g_ab, g_ab_steps))
                eval_records.append((p_ba, p_ba_steps, abst_ba, g_ba, g_ba_steps))

                env_non_comm = g_ab != g_ba
                pred_non_comm = (not abst_ab) and (not abst_ba) and (p_ab != p_ba)
                if env_non_comm and pred_non_comm and p_ab == g_ab and p_ba == g_ba:
                    non_commuting_detected += 1
                if env_non_comm and len(details) < 8:
                    details.append(
                        {
                            "pair": [aid_a, aid_b],
                            "env_non_commuting": env_non_comm,
                            "pred_non_commuting": pred_non_comm,
                            "ab_exact": (not abst_ab and p_ab == g_ab),
                            "ba_exact": (not abst_ba and p_ba == g_ba),
                        }
                    )

    summary = _summarize_probe_evaluations(
        "commutator_order_probes",
        eval_records,
        non_commuting_pairs_detected=non_commuting_detected,
        total_pairs=total_pairs,
        details=details,
    )
    return summary


def run_repeated_idempotent_probes(
    cards: Mapping[str, OperatorCard],
    grouped_pos_obs: Mapping[str, Sequence[ExtractedEpisodeObservation]],
    env: TransitionEnvironment,
) -> ProbeCategorySummary:
    """Domain-agnostic repeated/idempotent ``(a, a)`` probes across all induced operators."""
    eval_records = []
    details: list[dict[str, Any]] = []
    for action_id in sorted(cards):
        for base_state, ep in _distinct_positive_worlds(grouped_pos_obs.get(action_id, ())):
            act = build_probe_action(action_id, ep.roles)
            g_fin, g_steps = _execute_env_sequence(env, base_state, (act, act))
            p_fin, p_steps, abst = _execute_card_sequence(cards, base_state, (act, act))
            eval_records.append((p_fin, p_steps, abst, g_fin, g_steps))
            if len(details) < 8:
                details.append(
                    {
                        "action_id": action_id,
                        "step1_exact": (not abst and len(p_steps) >= 1 and p_steps[0] == g_steps[0]),
                        "step2_exact": (not abst and len(p_steps) >= 2 and p_steps[1] == g_steps[1]),
                        "second_step_noop_agreement": (
                            not abst
                            and len(p_steps) == 2
                            and ((p_steps[1] == p_steps[0]) == (g_steps[1] == g_steps[0]))
                        ),
                    }
                )
    return _summarize_probe_evaluations("repeated_idempotent_probes", eval_records, details=details)


def verify_depth_stratified_sequences(
    cards: Mapping[str, OperatorCard],
    depth_tasks: Mapping[int, Sequence[tuple[StateGraph, Sequence[ActionInvocation], tuple[StateGraph, ...] | None]]],
    env: TransitionEnvironment | None = None,
) -> dict[str, ProbeCategorySummary]:
    """Verify every-step intermediate postconditions across stratified depths (e.g., 1, 2, 4, 6).

    If ``gold_steps`` is supplied in ``depth_tasks[d]``, compares strictly against those gold
    intermediate states without querying ``env``; otherwise queries ``env`` to compute ground truth.
    """
    out: dict[str, ProbeCategorySummary] = {}
    for depth in sorted(depth_tasks):
        eval_records = []
        for initial_state, actions, precomputed_gold in depth_tasks[depth]:
            if precomputed_gold is not None:
                g_steps = tuple(precomputed_gold)
                g_fin = g_steps[-1] if g_steps else initial_state
            else:
                if env is None:
                    raise ValueError("env is required when precomputed gold_steps are not supplied")
                g_fin, g_steps = _execute_env_sequence(env, initial_state, actions)
            p_fin, p_steps, abst = _execute_card_sequence(cards, initial_state, actions)
            eval_records.append((p_fin, p_steps, abst, g_fin, g_steps))
        out[f"depth_{depth}"] = _summarize_probe_evaluations(f"depth_{depth}", eval_records)
    return out


def build_domain_agnostic_depth_sequences(
    cards: Mapping[str, OperatorCard],
    grouped_pos_obs: Mapping[str, Sequence[ExtractedEpisodeObservation]],
    env: TransitionEnvironment,
    *,
    depths: Sequence[int] = (1, 2, 4, 6),
) -> dict[int, list[tuple[StateGraph, tuple[ActionInvocation, ...], tuple[StateGraph, ...]]]]:
    """Synthesize domain-agnostic depth-stratified action chains using only public states & cards."""
    base_entries: list[tuple[StateGraph, str, dict[str, str]]] = []
    for aid in sorted(cards):
        for sg, ep in _distinct_positive_worlds(grouped_pos_obs.get(aid, ()))[:2]:
            base_entries.append((sg, aid, dict(ep.roles)))

    out: dict[int, list[tuple[StateGraph, tuple[ActionInvocation, ...], tuple[StateGraph, ...]]]] = {
        int(d): [] for d in depths
    }
    action_ids = sorted(cards.keys())
    if not base_entries or not action_ids:
        return out

    for depth in depths:
        d = int(depth)
        for idx, (base_state, start_aid, start_roles) in enumerate(base_entries):
            cur_env = base_state.copy()
            seq: list[ActionInvocation] = []
            gold_steps: list[StateGraph] = []
            cur_roles = dict(start_roles)
            cur_aid = start_aid

            for step_i in range(d):
                chosen_act: ActionInvocation | None = None
                if step_i == 0:
                    chosen_act = build_probe_action(cur_aid, cur_roles)
                else:
                    # Prefer a valid effect-bearing next action (e.g. swapped role or compatible action)
                    candidates: list[ActionInvocation] = []
                    prev_card = cards.get(cur_aid)
                    if prev_card is not None:
                        for ig in prev_card.inequality_guards:
                            sw = dict(cur_roles)
                            sw[ig.role_a], sw[ig.role_b] = sw[ig.role_b], sw[ig.role_a]
                            candidates.append(build_probe_action(cur_aid, sw))
                    for alt_aid in action_ids:
                        bound_alt = _bind_compatible_action(cards[alt_aid], cur_env, cur_roles)
                        if bound_alt is not None:
                            candidates.append(bound_alt)
                    # Pick first candidate that applies in cur_env (or fallback to first candidate)
                    for cand in candidates:
                        res_card = cards[cand.action_id].apply(cur_env, cand.role_dict())
                        if res_card.applied:
                            chosen_act = cand
                            break
                    if chosen_act is None and candidates:
                        chosen_act = candidates[(idx + step_i) % len(candidates)]
                    if chosen_act is None:
                        chosen_act = build_probe_action(cur_aid, cur_roles)

                cur_env, _delta = env.step(cur_env, chosen_act)
                seq.append(chosen_act)
                gold_steps.append(cur_env)
                cur_aid = chosen_act.action_id
                cur_roles = chosen_act.role_dict()

            out[d].append((base_state, tuple(seq), tuple(gold_steps)))
    return out


def run_contrast_audit(
    cards: Mapping[str, OperatorCard],
    grouped_pos_obs: Mapping[str, Sequence[ExtractedEpisodeObservation]],
    env: TransitionEnvironment,
    *,
    depth_tasks: Mapping[int, Sequence[tuple[StateGraph, Sequence[ActionInvocation], tuple[StateGraph, ...] | None]]] | None = None,
    ablation_reports: Sequence[AblationClauseReport] = (),
    guard_acquisition_reports: Sequence[GuardAcquisitionReport] = (),
) -> ContrastAuditReport:
    """Run all Stage S3 controlled contrast audits and return a JSON-serializable report."""
    role_swaps = run_role_swap_probes(cards, grouped_pos_obs, env)
    commutators = run_commutator_order_probes(cards, grouped_pos_obs, env)
    repeated = run_repeated_idempotent_probes(cards, grouped_pos_obs, env)

    if depth_tasks is None:
        depth_tasks = build_domain_agnostic_depth_sequences(
            cards, grouped_pos_obs, env, depths=(1, 2, 4, 6)
        )
    depth_summaries = verify_depth_stratified_sequences(cards, depth_tasks, env=env)

    all_passed = (
        role_swaps.passed
        and commutators.passed
        and commutators.non_commuting_pairs_detected >= 1
        and repeated.passed
        and all(ds.passed for ds in depth_summaries.values())
    )
    return ContrastAuditReport(
        role_swap_probes=role_swaps,
        commutator_order_probes=commutators,
        repeated_idempotent_probes=repeated,
        depth_stratified_verification=depth_summaries,
        ablation_reports=tuple(ablation_reports),
        guard_acquisition_reports=tuple(guard_acquisition_reports),
        all_checks_passed=all_passed,
    )


class ContrastRefinedITLLearner(ITLOperatorLearner):
    """Stage S3 ITL Learner with Single-Factor Precondition Ablation & OperatorVersionStore.

    - Runs S2 ITL induction from S1's append-only ``EpisodeLedger``.
    - Registers induced ``v1`` cards in ``OperatorVersionStore`` and promotes them.
    - Runs Stage S3 single-factor precondition ablation probes against the injected
      black-box ``TransitionEnvironment`` (refuting incidental invariants across
      ``>= min_ablation_support`` distinct base worlds) and promotes revised ``v2`` cards
      while retiring ``v1`` without deletion.
    - Supports repeated-evidence guard acquisition (``acquire_guards``) and safe version
      ``rollback_operator`` before ``freeze()``.
    - After ``freeze()``: forbids ALL further environment queries and ALL version-store
      mutations; ``predict()`` executes strictly from promoted ``OperatorCard``s with
      ZERO test-time episode reads and ZERO test-time environment queries.
    """

    def __init__(
        self,
        seed: int = 0,
        *,
        env: TransitionEnvironment | None = None,
        min_ablation_support: int = 2,
        min_guard_support: int = 2,
        probe_attribute_guards: bool = False,
        use_negative_contrasts: bool = True,
        raw_utf8_parse_mode: bool = False,
        taxonomy: TypeTaxonomy | None = DEFAULT_WORLD_TYPE_TAXONOMY,
        abstain_on_unknown_guard: bool = True,
    ) -> None:
        super().__init__(
            seed=seed,
            use_negative_contrasts=use_negative_contrasts,
            raw_utf8_parse_mode=raw_utf8_parse_mode,
            taxonomy=taxonomy,
            abstain_on_unknown_guard=abstain_on_unknown_guard,
        )
        self._env = env
        self.min_ablation_support = int(min_ablation_support)
        self.min_guard_support = int(min_guard_support)
        self.probe_attribute_guards = bool(probe_attribute_guards)
        self._version_store = OperatorVersionStore()
        self._corpus_stats = CorpusStatistics()
        self._grouped_pos_obs: dict[str, list[ExtractedEpisodeObservation]] = {}
        self._grouped_neg_obs: dict[str, list[ExtractedEpisodeObservation]] = {}
        self._ablation_reports: list[AblationClauseReport] = []
        self._guard_reports: list[GuardAcquisitionReport] = []
        self._refuted_facts_by_action: dict[str, set[tuple[str, str, str]]] = {}
        self._fit_env_queries: int = 0
        self._post_freeze_env_queries: int = 0
        self.name = "itl_s3_contrast_refined"
        self.category = "s3_contrast_refined_learner"

    @property
    def version_store(self) -> OperatorVersionStore:
        return self._version_store

    @property
    def ablation_reports(self) -> tuple[AblationClauseReport, ...]:
        return tuple(self._ablation_reports)

    @property
    def guard_acquisition_reports(self) -> tuple[GuardAcquisitionReport, ...]:
        return tuple(self._guard_reports)

    @property
    def fit_env_queries(self) -> int:
        return self._fit_env_queries

    @property
    def post_freeze_env_queries(self) -> int:
        return self._post_freeze_env_queries

    def _guarded_env(self) -> TransitionEnvironment:
        self._check_not_frozen()
        if self._env is None:
            raise RuntimeError("No TransitionEnvironment injected into ContrastRefinedITLLearner")
        outer = self

        class _CountingFrozenCheckedEnv:
            def step(
                self,
                state: StateGraph,
                action: ActionInvocation,
                mask: ObservationMask | None = None,
            ) -> tuple[StateGraph, SparseDelta]:
                if outer.is_frozen:
                    outer._post_freeze_env_queries += 1
                    raise RuntimeError(
                        "ContrastRefinedITLLearner is frozen; environment queries are forbidden"
                    )
                outer._fit_env_queries += 1
                assert outer._env is not None
                return outer._env.step(state, action, mask=mask)

        return _CountingFrozenCheckedEnv()

    def freeze(self) -> None:
        super().freeze()
        self._version_store.freeze()

    def _sync_active_operators(self) -> None:
        self._operators = self._version_store.active_cards()

    def _register_initial_and_refine(
        self,
        grouped: dict[str, list[ExtractedEpisodeObservation]],
        initial_cards: dict[str, OperatorCard],
    ) -> None:
        self._version_store = OperatorVersionStore()
        self._corpus_stats = collect_corpus_statistics(grouped)
        self._grouped_pos_obs = {}
        self._grouped_neg_obs = {}
        self._ablation_reports = []
        self._guard_reports = []
        self._refuted_facts_by_action = {}

        for aid, obs_list in sorted(grouped.items()):
            pos, neg = partition_positive_negative(obs_list)
            self._grouped_pos_obs[aid] = pos
            self._grouped_neg_obs[aid] = neg

        for aid, card in sorted(initial_cards.items()):
            self._version_store.register_shadow(
                card,
                evidence={
                    "source": "s2_passive_induction",
                    "support_positive": card.support_positive,
                    "support_negative": card.support_negative,
                },
                contrast_pass_rate=card.confidence,
                notes="initial_s2_induced_card",
            )
            self._version_store.promote(
                aid,
                card.version,
                notes="initial_s2_promoted",
            )

        if self._env is not None:
            guarded_env = self._guarded_env()
            for aid in sorted(initial_cards):
                active_rec = self._version_store.get_active(aid)
                if active_rec is None:
                    continue
                revised, reports = refine_operator_by_single_factor_ablation(
                    active_rec.card,
                    self._grouped_pos_obs.get(aid, ()),
                    self._grouped_neg_obs.get(aid, ()),
                    self._corpus_stats,
                    guarded_env,
                    min_support=self.min_ablation_support,
                    probe_attribute_guards=self.probe_attribute_guards,
                    next_version=self._version_store.next_version(aid),
                )
                self._ablation_reports.extend(reports)
                refuted_facts_here = {
                    (r.clause[0], r.clause[1], r.clause[2])
                    for r in reports
                    if r.refuted and r.clause_kind == "required_fact" and len(r.clause) == 3
                }
                refuted_all_here = sorted(r.clause for r in reports if r.refuted)
                if refuted_facts_here:
                    self._refuted_facts_by_action[aid] = refuted_facts_here
                if revised is not None:
                    self._version_store.register_shadow(
                        revised,
                        evidence={
                            "source": "s3_single_factor_ablation",
                            "refuted_clauses": [list(c) for c in refuted_all_here],
                            "min_support": self.min_ablation_support,
                        },
                        contrast_pass_rate=revised.confidence,
                        notes="single_factor_ablation_refined",
                    )
                    self._version_store.promote(
                        aid,
                        revised.version,
                        notes="promoted_after_single_factor_ablation",
                    )

        self._sync_active_operators()

    def fit_from_ledger(
        self,
        ledger: EpisodeLedger,
        *,
        split_id: str | None = None,
        seed: int | None = None,
    ) -> dict[str, OperatorCard]:
        self._check_not_frozen()
        initial_cards = super().fit_from_ledger(ledger, split_id=split_id, seed=seed)
        grouped = extract_observations_from_ledger(
            ledger, split_id=split_id, seed=seed, caller_tag=f"{self.name}:extract"
        )
        self._register_initial_and_refine(grouped, initial_cards)
        return self.operators

    def fit(self, episodes: Sequence[TrainingEpisode]) -> None:
        self._check_not_frozen()
        if self._ledger is not None:
            self._ledger.close()
        self._ledger = EpisodeLedger(":memory:")
        for ep in episodes:
            self._ledger.ingest_training_episode(ep, use_typed_action=True)
        self.fit_from_ledger(self._ledger, split_id=None)
        self._ledger.reset_audit_log()

    def acquire_guards(
        self,
        action_id: str,
        contrast_probes: Sequence[tuple[StateGraph, ActionInvocation]],
        *,
        min_guard_support: int | None = None,
    ) -> GuardAcquisitionReport:
        """Run repeated-evidence guard acquisition on ``action_id`` (forbidden after freeze)."""
        guarded_env = self._guarded_env()
        rep = acquire_guards_from_contrasts(
            action_id,
            self._version_store,
            self._grouped_pos_obs.get(action_id, ()),
            contrast_probes,
            self._corpus_stats,
            guarded_env,
            min_guard_support=(
                self.min_guard_support if min_guard_support is None else int(min_guard_support)
            ),
            neg_obs=self._grouped_neg_obs.get(action_id, ()),
            refuted_facts=frozenset(self._refuted_facts_by_action.get(action_id, ())),
        )
        self._guard_reports.append(rep)
        self._sync_active_operators()
        return rep

    def rollback_operator(
        self,
        action_id: str,
        target_version: int,
        *,
        reason: str = "",
    ) -> OperatorCard:
        """Roll back an active operator to ``target_version`` without deleting any version."""
        self._check_not_frozen()
        rec = self._version_store.rollback(action_id, target_version, reason=reason)
        self._sync_active_operators()
        return rec.card


class UnderconstrainedContrastRepairLearner(ContrastRefinedITLLearner):
    """Stage S3 experimental learner proving repeated-evidence guard acquisition from stripped cards.

    1. Induces baseline cards, then strips selective guards (``strip_operator_guards``) to
       produce intentionally under-constrained ``v1`` candidates.
    2. Generates domain-agnostic single-factor contrast probes from training base worlds.
    3. Repairs each operator via ``acquire_guards_from_contrasts(..., min_guard_support=2)``,
       promoting repaired ``v2`` cards while preserving ``v1`` as ``retired``.
    """

    def __init__(
        self,
        seed: int = 0,
        *,
        env: TransitionEnvironment,
        min_guard_support: int = 2,
        taxonomy: TypeTaxonomy | None = DEFAULT_WORLD_TYPE_TAXONOMY,
    ) -> None:
        super().__init__(
            seed=seed,
            env=env,
            min_ablation_support=2,
            min_guard_support=min_guard_support,
            taxonomy=taxonomy,
        )
        self.name = "itl_s3_underconstrained_repaired"
        self.category = "s3_guard_repair_learner"
        self._pre_repair_cards: dict[str, OperatorCard] = {}

    @property
    def pre_repair_cards(self) -> dict[str, OperatorCard]:
        return dict(self._pre_repair_cards)

    def _register_initial_and_refine(
        self,
        grouped: dict[str, list[ExtractedEpisodeObservation]],
        initial_cards: dict[str, OperatorCard],
    ) -> None:
        self._version_store = OperatorVersionStore()
        self._corpus_stats = collect_corpus_statistics(grouped)
        self._grouped_pos_obs = {}
        self._grouped_neg_obs = {}
        self._ablation_reports = []
        self._guard_reports = []
        self._refuted_facts_by_action = {}
        self._pre_repair_cards = {}

        for aid, obs_list in sorted(grouped.items()):
            pos, neg = partition_positive_negative(obs_list)
            self._grouped_pos_obs[aid] = pos
            self._grouped_neg_obs[aid] = neg

        guarded_env = self._guarded_env()

        for aid, full_card in sorted(initial_cards.items()):
            stripped = strip_operator_guards(full_card, version=1)
            self._pre_repair_cards[aid] = stripped
            self._version_store.register_shadow(
                stripped,
                evidence={"source": "s3_intentionally_stripped_underconstrained_v1"},
                contrast_pass_rate=0.0,
                notes="stripped_underconstrained_initial",
            )
            self._version_store.promote(
                aid,
                stripped.version,
                notes="stripped_initial_promoted",
            )

            probes = generate_domain_agnostic_contrast_probes(
                aid,
                stripped,
                self._grouped_pos_obs.get(aid, ()),
                self._corpus_stats,
            )
            rep = acquire_guards_from_contrasts(
                aid,
                self._version_store,
                self._grouped_pos_obs.get(aid, ()),
                probes,
                self._corpus_stats,
                guarded_env,
                min_guard_support=self.min_guard_support,
                neg_obs=self._grouped_neg_obs.get(aid, ()),
            )
            self._guard_reports.append(rep)

        self._sync_active_operators()
