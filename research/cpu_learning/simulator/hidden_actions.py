"""EVALUATOR-ONLY hidden transition rules for the deterministic relational simulator.

STRICT ISOLATION RULE (PDF pp. 10, 15, 16):
- This module defines the ground-truth hidden transition semantics for the four
  benchmark actions: ``move``, ``give``, ``open``, and ``lock``.
- Candidate learners (S1-S10) MUST NEVER import this module or receive its edit
  scripts, precondition clauses, or gold transition functions.
- Only the generator (to produce ground-truth transitions) and the evaluator/ceiling
  oracle may import this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .state_graph import (
    ActionInvocation,
    ObservationMask,
    SparseDelta,
    StateGraph,
    compute_sparse_delta,
)

EVALUATOR_ONLY: bool = True

HIDDEN_ACTION_NAMES: tuple[str, ...] = ("move", "give", "open", "lock")


@dataclass(frozen=True)
class HiddenTransitionResult:
    """Ground-truth single-step transition output (evaluator-only)."""

    after_state: StateGraph
    delta: SparseDelta
    applied: bool
    reason: str


@dataclass(frozen=True)
class HiddenOperatorSchema:
    """Abstract gold preconditions and effects for S2+ operator-recovery scoring.

    Never exposed to learners; used exclusively by ``metrics.score_operator_recovery``.
    """

    action_id: str
    role_variables: tuple[str, ...]
    required_facts: frozenset[tuple[str, str, str]]
    required_types: frozenset[tuple[str, str]]
    required_attr_guards: frozenset[tuple[str, str, str]]
    add_facts: frozenset[tuple[str, str, str]]
    delete_facts: frozenset[tuple[str, str, str]]
    attribute_edits: frozenset[tuple[str, str, str]]


HIDDEN_OPERATOR_SCHEMAS: Mapping[str, HiddenOperatorSchema] = {
    "move": HiddenOperatorSchema(
        action_id="move",
        role_variables=("actor", "object", "source", "destination"),
        required_facts=frozenset(
            {
                ("?object", "owned_by", "?actor"),
                ("?object", "located_on", "?source"),
                ("?destination", "accessible", "true"),
                ("?source_if_container", "is_open", "true"),
                ("?source_if_container", "is_locked", "false"),
                ("?destination_if_container", "is_open", "true"),
                ("?destination_if_container", "is_locked", "false"),
            }
        ),
        required_types=frozenset(
            {
                ("?actor", "agent"),
                ("?object", "object"),
                ("?source", "location"),
                ("?destination", "location"),
            }
        ),
        required_attr_guards=frozenset(
            {
                ("?object", "portable", "true"),
                ("?source", "neq", "?destination"),
            }
        ),
        add_facts=frozenset({("?object", "located_on", "?destination")}),
        delete_facts=frozenset({("?object", "located_on", "?source")}),
        attribute_edits=frozenset(),
    ),
    "give": HiddenOperatorSchema(
        action_id="give",
        role_variables=("actor", "object", "recipient"),
        required_facts=frozenset(
            {
                ("?object", "owned_by", "?actor"),
                ("?object_container_if_any", "is_open", "true"),
                ("?object_container_if_any", "is_locked", "false"),
            }
        ),
        required_types=frozenset(
            {
                ("?actor", "agent"),
                ("?recipient", "agent"),
                ("?object", "object"),
            }
        ),
        required_attr_guards=frozenset(
            {
                ("?object", "portable", "true"),
                ("?actor", "neq", "?recipient"),
            }
        ),
        add_facts=frozenset({("?object", "owned_by", "?recipient")}),
        delete_facts=frozenset({("?object", "owned_by", "?actor")}),
        attribute_edits=frozenset(),
    ),
    "open": HiddenOperatorSchema(
        action_id="open",
        role_variables=("actor", "target"),
        required_facts=frozenset(
            {
                ("?target", "accessible", "true"),
                ("?target", "is_locked", "false"),
                ("?target", "is_open", "false"),
            }
        ),
        required_types=frozenset(
            {
                ("?actor", "agent"),
                ("?target", "container"),
            }
        ),
        required_attr_guards=frozenset(),
        add_facts=frozenset({("?target", "is_open", "true")}),
        delete_facts=frozenset({("?target", "is_open", "false")}),
        attribute_edits=frozenset({("?target", "status", "open")}),
    ),
    "lock": HiddenOperatorSchema(
        action_id="lock",
        role_variables=("actor", "target"),
        required_facts=frozenset(
            {
                ("?target", "accessible", "true"),
                ("?target", "is_open", "false"),
                ("?target", "is_locked", "false"),
            }
        ),
        required_types=frozenset(
            {
                ("?actor", "agent"),
                ("?target", "container"),
            }
        ),
        required_attr_guards=frozenset(),
        add_facts=frozenset({("?target", "is_locked", "true")}),
        delete_facts=frozenset({("?target", "is_locked", "false")}),
        attribute_edits=frozenset({("?target", "status", "locked")}),
    ),
}


def _noop(state: StateGraph, reason: str, mask: ObservationMask | None) -> HiddenTransitionResult:
    after = state.copy()
    return HiddenTransitionResult(
        after_state=after,
        delta=compute_sparse_delta(state, after, mask=mask),
        applied=False,
        reason=reason,
    )


def _check_container_passable(state: StateGraph, loc_id: str, prefix: str) -> str | None:
    ent = state.get_entity(loc_id)
    if ent is None or ent.entity_type != "container":
        return None
    if not state.has_fact(loc_id, "is_locked", "false"):
        return f"blocked:{prefix}_container_locked"
    if not state.has_fact(loc_id, "is_open", "true"):
        return f"blocked:{prefix}_container_closed"
    return None


def _apply_move(
    state: StateGraph, roles: Mapping[str, str], mask: ObservationMask | None
) -> HiddenTransitionResult:
    actor = roles.get("actor", "")
    obj = roles.get("object", "")
    src = roles.get("source", "")
    dst = roles.get("destination", "")

    actor_ent = state.get_entity(actor)
    obj_ent = state.get_entity(obj)
    src_ent = state.get_entity(src)
    dst_ent = state.get_entity(dst)

    if actor_ent is None or actor_ent.entity_type != "agent":
        return _noop(state, "blocked:invalid_actor", mask)
    if obj_ent is None or obj_ent.entity_type != "object":
        return _noop(state, "blocked:invalid_object", mask)
    if obj_ent.get("portable") is not True:
        return _noop(state, "blocked:object_not_portable", mask)
    if src_ent is None or src_ent.entity_type not in ("surface", "container"):
        return _noop(state, "blocked:invalid_source", mask)
    if dst_ent is None or dst_ent.entity_type not in ("surface", "container"):
        return _noop(state, "blocked:invalid_destination", mask)
    if src == dst:
        return _noop(state, "blocked:same_source_and_destination", mask)
    if not state.has_fact(obj, "owned_by", actor):
        return _noop(state, "blocked:actor_not_owner", mask)
    if not state.has_fact(obj, "located_on", src):
        return _noop(state, "blocked:object_not_at_source", mask)
    if not state.has_fact(dst, "accessible", "true"):
        return _noop(state, "blocked:destination_inaccessible", mask)

    src_block = _check_container_passable(state, src, "source")
    if src_block is not None:
        return _noop(state, src_block, mask)
    dst_block = _check_container_passable(state, dst, "destination")
    if dst_block is not None:
        return _noop(state, dst_block, mask)

    after = state.with_changes(
        delete_facts=((obj, "located_on", src),),
        add_facts=((obj, "located_on", dst),),
    )
    return HiddenTransitionResult(
        after_state=after,
        delta=compute_sparse_delta(state, after, mask=mask),
        applied=True,
        reason="applied",
    )


def _apply_give(
    state: StateGraph, roles: Mapping[str, str], mask: ObservationMask | None
) -> HiddenTransitionResult:
    actor = roles.get("actor", "")
    obj = roles.get("object", "")
    recipient = roles.get("recipient", "")

    actor_ent = state.get_entity(actor)
    obj_ent = state.get_entity(obj)
    rec_ent = state.get_entity(recipient)

    if actor_ent is None or actor_ent.entity_type != "agent":
        return _noop(state, "blocked:invalid_actor", mask)
    if rec_ent is None or rec_ent.entity_type != "agent":
        return _noop(state, "blocked:invalid_recipient", mask)
    if actor == recipient:
        return _noop(state, "blocked:self_transfer", mask)
    if obj_ent is None or obj_ent.entity_type != "object":
        return _noop(state, "blocked:invalid_object", mask)
    if obj_ent.get("portable") is not True:
        return _noop(state, "blocked:object_not_portable", mask)
    if not state.has_fact(obj, "owned_by", actor):
        return _noop(state, "blocked:actor_not_owner", mask)

    for _subj, _pred, loc in state.facts_for(obj, "located_on"):
        loc_block = _check_container_passable(state, loc, "object_location")
        if loc_block is not None:
            return _noop(state, loc_block, mask)

    after = state.with_changes(
        delete_facts=((obj, "owned_by", actor),),
        add_facts=((obj, "owned_by", recipient),),
    )
    return HiddenTransitionResult(
        after_state=after,
        delta=compute_sparse_delta(state, after, mask=mask),
        applied=True,
        reason="applied",
    )


def _apply_open(
    state: StateGraph, roles: Mapping[str, str], mask: ObservationMask | None
) -> HiddenTransitionResult:
    actor = roles.get("actor", "")
    target = roles.get("target", "")

    actor_ent = state.get_entity(actor)
    target_ent = state.get_entity(target)

    if actor_ent is None or actor_ent.entity_type != "agent":
        return _noop(state, "blocked:invalid_actor", mask)
    if target_ent is None or target_ent.entity_type != "container":
        return _noop(state, "blocked:target_not_container", mask)
    if not state.has_fact(target, "accessible", "true"):
        return _noop(state, "blocked:target_inaccessible", mask)
    if not state.has_fact(target, "is_locked", "false"):
        return _noop(state, "blocked:target_locked", mask)
    if not state.has_fact(target, "is_open", "false"):
        return _noop(state, "blocked:already_open", mask)

    after = state.with_changes(
        delete_facts=((target, "is_open", "false"),),
        add_facts=((target, "is_open", "true"),),
        attribute_edits=((target, "status", "open"),),
    )
    return HiddenTransitionResult(
        after_state=after,
        delta=compute_sparse_delta(state, after, mask=mask),
        applied=True,
        reason="applied",
    )


def _apply_lock(
    state: StateGraph, roles: Mapping[str, str], mask: ObservationMask | None
) -> HiddenTransitionResult:
    actor = roles.get("actor", "")
    target = roles.get("target", "")

    actor_ent = state.get_entity(actor)
    target_ent = state.get_entity(target)

    if actor_ent is None or actor_ent.entity_type != "agent":
        return _noop(state, "blocked:invalid_actor", mask)
    if target_ent is None or target_ent.entity_type != "container":
        return _noop(state, "blocked:target_not_container", mask)
    if not state.has_fact(target, "accessible", "true"):
        return _noop(state, "blocked:target_inaccessible", mask)
    if not state.has_fact(target, "is_open", "false"):
        return _noop(state, "blocked:target_must_be_closed", mask)
    if not state.has_fact(target, "is_locked", "false"):
        return _noop(state, "blocked:already_locked", mask)

    after = state.with_changes(
        delete_facts=((target, "is_locked", "false"),),
        add_facts=((target, "is_locked", "true"),),
        attribute_edits=((target, "status", "locked"),),
    )
    return HiddenTransitionResult(
        after_state=after,
        delta=compute_sparse_delta(state, after, mask=mask),
        applied=True,
        reason="applied",
    )


_DISPATCH = {
    "move": _apply_move,
    "give": _apply_give,
    "open": _apply_open,
    "lock": _apply_lock,
}


def apply_hidden_action(
    state: StateGraph,
    action: ActionInvocation,
    mask: ObservationMask | None = None,
) -> HiddenTransitionResult:
    """Apply one hidden action deterministically (evaluator-only)."""
    handler = _DISPATCH.get(action.action_id)
    if handler is None:
        return _noop(state, f"blocked:unknown_action:{action.action_id}", mask)
    return handler(state, action.role_dict(), mask)


def apply_hidden_sequence(
    initial_state: StateGraph,
    actions: Sequence[ActionInvocation],
    mask: ObservationMask | None = None,
) -> tuple[
    StateGraph,
    tuple[StateGraph, ...],
    tuple[SparseDelta, ...],
    tuple[bool, ...],
    tuple[str, ...],
]:
    """Execute a deterministic sequence of hidden actions (evaluator-only)."""
    cur = initial_state.copy()
    intermediates: list[StateGraph] = []
    deltas: list[SparseDelta] = []
    applied_flags: list[bool] = []
    reasons: list[str] = []

    for act in actions:
        res = apply_hidden_action(cur, act, mask=mask)
        cur = res.after_state
        intermediates.append(cur)
        deltas.append(res.delta)
        applied_flags.append(res.applied)
        reasons.append(res.reason)

    return (
        cur,
        tuple(intermediates),
        tuple(deltas),
        tuple(applied_flags),
        tuple(reasons),
    )