"""Stage S5 RDC Minimum Description Length (MDL) Consolidation Objective & Judge.

Implements the exact PDF p. 8 consolidation objective:
    J = L(library) + L(examples | library) + beta * verified_failures + gamma * collateral_changes

STRICT ISOLATION (Stage S5):
- Contains ZERO hardcoded domain action, predicate, attribute, or entity-type literals.
- NEVER imports ``hidden_actions``, ``oracle``, or ``generators``.
- Operates strictly on public ``StateGraph``, ``ActionInvocation``, ``SparseDelta``,
  and executable ``OperatorCard`` libraries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from research.cpu_learning.itl.operator_card import OperatorCard
from research.cpu_learning.simulator.state_graph import (
    ActionInvocation,
    ObservationMask,
    SparseDelta,
    StateGraph,
    TrainingEpisode,
    canonical_json_bytes,
    compressed_bit_length,
    compute_sparse_delta,
)

RDC_MDL_SCHEMA_VERSION: str = "s5-rdc-mdl-v1"
DEFAULT_BETA: float = 10_000.0
DEFAULT_GAMMA: float = 20_000.0

_MISSING_ATTR = object()


@dataclass(frozen=True)
class MDLExample:
    """Public observed single-step or multi-step transition example for MDL scoring."""

    example_id: str
    before_state: StateGraph
    actions: tuple[ActionInvocation, ...]
    observed_after_state: StateGraph
    mask: ObservationMask = ObservationMask(fully_observed=True)

    @classmethod
    def from_training_episode(cls, ep: TrainingEpisode) -> MDLExample:
        return cls(
            example_id=ep.episode_id,
            before_state=ep.before_state,
            actions=(ep.action,),
            observed_after_state=ep.after_state,
            mask=ep.observation_mask,
        )


@dataclass(frozen=True)
class MDLScoreReport:
    """Exact breakdown of the RDC MDL consolidation objective J."""

    j_score: float
    library_bits: int
    residual_bits: int
    verified_failures: int
    collateral_changes: int
    total_examples: int
    exact_matches: int
    exact_match_rate: float
    beta: float = DEFAULT_BETA
    gamma: float = DEFAULT_GAMMA
    schema_version: str = RDC_MDL_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "j_score": round(self.j_score, 4),
            "library_bits": self.library_bits,
            "residual_bits": self.residual_bits,
            "verified_failures": self.verified_failures,
            "collateral_changes": self.collateral_changes,
            "total_examples": self.total_examples,
            "exact_matches": self.exact_matches,
            "exact_match_rate": round(self.exact_match_rate, 6),
            "beta": self.beta,
            "gamma": self.gamma,
        }


def count_collateral_changes(
    initial_state: StateGraph,
    observed_after: StateGraph,
    predicted_after: StateGraph | None,
    *,
    abstained: bool = False,
) -> int:
    """Count unrelated facts, attributes, entity types, and hallucinated entities altered incorrectly."""
    if abstained or predicted_after is None:
        return 0

    preserved_facts = initial_state.facts & observed_after.facts
    init_ents = initial_state.entity_map()
    obs_ents = observed_after.entity_map()
    pred_ents = predicted_after.entity_map()

    obs_changed_fact_keys = {(f[0], f[1]) for f in (initial_state.facts ^ observed_after.facts)}
    obs_changed_attr_keys: set[tuple[str, str]] = set()
    obs_changed_types: set[str] = set()

    for eid in set(init_ents) | set(obs_ents):
        i_ent = init_ents.get(eid)
        o_ent = obs_ents.get(eid)
        if i_ent is None or o_ent is None or i_ent.entity_type != o_ent.entity_type:
            obs_changed_types.add(eid)
        i_attrs = i_ent.attr_dict() if i_ent is not None else {}
        o_attrs = o_ent.attr_dict() if o_ent is not None else {}
        for ak in set(i_attrs) | set(o_attrs):
            if i_attrs.get(ak, _MISSING_ATTR) != o_attrs.get(ak, _MISSING_ATTR):
                obs_changed_attr_keys.add((eid, ak))

    deleted_unrelated = len(preserved_facts - predicted_after.facts)
    spurious_facts = predicted_after.facts - observed_after.facts
    spurious_unrelated = sum(
        1 for f in spurious_facts if (f[0], f[1]) not in obs_changed_fact_keys
    )

    entity_attr_col = 0
    for eid, o_ent in obs_ents.items():
        p_ent = pred_ents.get(eid)
        if p_ent is None:
            entity_attr_col += 1
            continue
        if eid not in obs_changed_types and p_ent.entity_type != o_ent.entity_type:
            entity_attr_col += 1
        o_attrs = o_ent.attr_dict()
        p_attrs = p_ent.attr_dict()
        for ak in set(o_attrs) | set(p_attrs):
            if (eid, ak) in obs_changed_attr_keys:
                continue
            if o_attrs.get(ak, _MISSING_ATTR) != p_attrs.get(ak, _MISSING_ATTR):
                entity_attr_col += 1

    hallucinated_entities = len(set(pred_ents) - set(obs_ents))
    return deleted_unrelated + spurious_unrelated + entity_attr_col + hallucinated_entities


def compute_library_compressed_bits(
    library: Mapping[str, OperatorCard],
    *,
    alias_map: Mapping[str, str] | None = None,
    concept_payload: Mapping[str, Any] | None = None,
) -> int:
    """Compute exact zlib-compressed bit length ``L(library)`` of executable cards & abstractions."""
    if concept_payload is not None:
        return compressed_bit_length(canonical_json_bytes(concept_payload))
    if not library and not alias_map:
        return 0
    payload: dict[str, Any] = {
        "aliases": {k: v for k, v in sorted((alias_map or {}).items())},
        "cards": {aid: card.to_executable_dict() for aid, card in sorted(library.items())},
    }
    return compressed_bit_length(canonical_json_bytes(payload))


def execute_library_sequence(
    library: Mapping[str, OperatorCard],
    initial_state: StateGraph,
    actions: Sequence[ActionInvocation],
    *,
    alias_map: Mapping[str, str] | None = None,
    mask: ObservationMask | None = None,
) -> tuple[StateGraph | None, bool, tuple[StateGraph, ...]]:
    """Execute ``actions`` sequentially through ``library`` (with optional alias redirection)."""
    cur = initial_state.copy()
    intermediates: list[StateGraph] = []
    aliases = alias_map or {}

    for act in actions:
        resolved_aid = aliases.get(act.action_id, act.action_id)
        card = library.get(resolved_aid)
        if card is None:
            return (None, True, tuple(intermediates))
        res = card.apply(cur, act.role_dict(), mask=mask)
        if res.abstained or res.after_state is None:
            return (None, True, tuple(intermediates))
        cur = res.after_state
        intermediates.append(cur)
    return (cur, False, tuple(intermediates))


def evaluate_library_mdl(
    library: Mapping[str, OperatorCard],
    examples: Sequence[MDLExample],
    *,
    alias_map: Mapping[str, str] | None = None,
    concept_payload: Mapping[str, Any] | None = None,
    library_bits_override: int | None = None,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> MDLScoreReport:
    """Evaluate exact RDC consolidation score J = L(lib) + L(ex|lib) + beta*fail + gamma*col."""
    lib_bits = (
        int(library_bits_override)
        if library_bits_override is not None
        else compute_library_compressed_bits(
            library, alias_map=alias_map, concept_payload=concept_payload
        )
    )

    residuals: list[dict[str, Any]] = []
    verified_failures = 0
    total_collateral = 0
    exact_matches = 0

    for ex in examples:
        pred_after, abstained, _inters = execute_library_sequence(
            library, ex.before_state, ex.actions, alias_map=alias_map, mask=ex.mask
        )
        if (not abstained) and (pred_after == ex.observed_after_state):
            exact_matches += 1
            continue

        verified_failures += 1
        base_for_res = ex.before_state if (abstained or pred_after is None) else pred_after
        res_delta: SparseDelta = compute_sparse_delta(
            base_for_res, ex.observed_after_state, mask=ex.mask
        )
        residuals.append({"id": ex.example_id, "residual": res_delta.to_dict()})
        total_collateral += count_collateral_changes(
            ex.before_state,
            ex.observed_after_state,
            pred_after,
            abstained=abstained,
        )

    residual_bits = (
        0 if not residuals else compressed_bit_length(canonical_json_bytes(residuals))
    )
    j_score = (
        float(lib_bits)
        + float(residual_bits)
        + float(beta) * float(verified_failures)
        + float(gamma) * float(total_collateral)
    )
    total = len(examples)
    match_rate = exact_matches / float(total) if total > 0 else 1.0
    return MDLScoreReport(
        j_score=j_score,
        library_bits=lib_bits,
        residual_bits=residual_bits,
        verified_failures=verified_failures,
        collateral_changes=total_collateral,
        total_examples=total,
        exact_matches=exact_matches,
        exact_match_rate=match_rate,
        beta=float(beta),
        gamma=float(gamma),
    )
