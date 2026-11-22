"""Action-name-agnostic relational lifting & anti-unification utilities for Stage S2 ITL.

STRICT ISOLATION:
- Operates strictly on public ``TypedStateFrame`` and ``SparseDelta`` observations.
- NEVER imports ``hidden_actions``, ``oracle``, or ``generators``.
- Contains ZERO hardcoded action names, predicate names, or entity-type names.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from research.cpu_learning.frames.state_frame import TypedStateFrame
from research.cpu_learning.itl.operator_card import LiftedAttributeEdit
from research.cpu_learning.simulator.state_graph import Fact, SparseDelta


@dataclass(frozen=True)
class LiftedDelta:
    """Role-variable lifted representation of a concrete SparseDelta."""

    add_facts: frozenset[tuple[str, str, str]]
    delete_facts: frozenset[tuple[str, str, str]]
    attribute_edits: frozenset[LiftedAttributeEdit]

    @property
    def is_empty(self) -> bool:
        return not self.add_facts and not self.delete_facts and not self.attribute_edits


@dataclass(frozen=True)
class OneHopLink:
    """Directed 1-hop relational edge from a bound role entity to an unbound world entity."""

    role: str
    via_predicate: str
    neighbor_id: str
    neighbor_type: str


def build_entity_to_roles_map(roles: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    """Invert role->entity_id into entity_id->sorted(role_names)."""
    inv: dict[str, list[str]] = {}
    for rname, eid in sorted(roles.items()):
        inv.setdefault(str(eid), []).append(str(rname))
    return {eid: tuple(rnames) for eid, rnames in inv.items()}


def world_entity_id_set(frame: TypedStateFrame) -> frozenset[str]:
    """All concrete entity IDs present in ``frame``."""
    return frozenset(eid for eid, _ef in frame.entities)


def lift_fact_for_roles(
    fact: Fact,
    eid_to_roles: Mapping[str, tuple[str, ...]],
    world_entities: frozenset[str],
) -> tuple[tuple[str, str, str], ...]:
    """Lift a concrete fact ``(s, p, o)`` into role-variable tuple(s) when representable.

    - Subject ``s`` must be bound to at least one role variable.
    - Object ``o`` lifts to ``?role`` if bound to a role, stays literal if ``o`` is NOT
      a concrete entity ID in ``world_entities``, and is skipped if ``o`` is an unbound
      concrete world entity (handled separately by 1-hop link extraction).
    """
    subj, pred, obj = fact
    s_roles = eid_to_roles.get(subj)
    if not s_roles:
        return ()

    o_roles = eid_to_roles.get(obj)
    if o_roles:
        o_terms = tuple(f"?{r}" for r in o_roles)
    elif obj not in world_entities:
        o_terms = (obj,)
    else:
        return ()

    out: list[tuple[str, str, str]] = []
    for sr in s_roles:
        for ot in o_terms:
            out.append((f"?{sr}", pred, ot))
    return tuple(out)


def lift_sparse_delta(
    delta: SparseDelta,
    roles: Mapping[str, str],
    before_frame: TypedStateFrame,
    after_frame: TypedStateFrame | None = None,
) -> LiftedDelta:
    """Lift concrete added/deleted facts and attribute edits into role-variable space."""
    eid_to_roles = build_entity_to_roles_map(roles)
    world_ents = world_entity_id_set(before_frame)
    if after_frame is not None:
        world_ents = world_ents | world_entity_id_set(after_frame)

    lifted_adds: set[tuple[str, str, str]] = set()
    for fact in delta.added_facts:
        lifted_adds.update(lift_fact_for_roles(fact, eid_to_roles, world_ents))

    lifted_dels: set[tuple[str, str, str]] = set()
    for fact in delta.deleted_facts:
        lifted_dels.update(lift_fact_for_roles(fact, eid_to_roles, world_ents))

    lifted_edits: set[LiftedAttributeEdit] = set()
    for eid, attr_name, _old_v, new_v in delta.attribute_edits:
        e_roles = eid_to_roles.get(eid)
        if not e_roles:
            continue
        for rname in e_roles:
            lifted_edits.add(LiftedAttributeEdit(role=rname, attribute=attr_name, new_value=new_v))

    return LiftedDelta(
        add_facts=frozenset(lifted_adds),
        delete_facts=frozenset(lifted_dels),
        attribute_edits=frozenset(lifted_edits),
    )


def intersect_lifted_deltas(lifted_deltas: Sequence[LiftedDelta]) -> LiftedDelta:
    """Anti-unify lifted deltas across positive demonstrations by exact intersection."""
    if not lifted_deltas:
        return LiftedDelta(frozenset(), frozenset(), frozenset())
    adds = set(lifted_deltas[0].add_facts)
    dels = set(lifted_deltas[0].delete_facts)
    edits = set(lifted_deltas[0].attribute_edits)
    for ld in lifted_deltas[1:]:
        adds &= ld.add_facts
        dels &= ld.delete_facts
        edits &= ld.attribute_edits
    return LiftedDelta(
        add_facts=frozenset(adds),
        delete_facts=frozenset(dels),
        attribute_edits=frozenset(edits),
    )


def extract_direct_lifted_facts(
    frame: TypedStateFrame,
    roles: Mapping[str, str],
) -> frozenset[tuple[str, str, str]]:
    """Extract all observed facts in ``frame`` liftable directly onto ``roles`` or literals."""
    eid_to_roles = build_entity_to_roles_map(roles)
    world_ents = world_entity_id_set(frame)
    out: set[tuple[str, str, str]] = set()
    for fact in frame.observed_facts:
        out.update(lift_fact_for_roles(fact, eid_to_roles, world_ents))
    return frozenset(out)


def extract_entity_literal_facts(
    frame: TypedStateFrame,
    entity_id: str,
) -> frozenset[tuple[str, str]]:
    """Return all observed ``(predicate, literal_value)`` facts on ``entity_id``."""
    world_ents = world_entity_id_set(frame)
    out: set[tuple[str, str]] = set()
    for _s, pred, obj in frame.facts_for(entity_id):
        if obj not in world_ents:
            out.add((pred, obj))
    return frozenset(out)


def extract_one_hop_links(
    frame: TypedStateFrame,
    roles: Mapping[str, str],
) -> tuple[OneHopLink, ...]:
    """Extract all 1-hop directed links ``(role, pred, neighbor_id, neighbor_type)``
    where ``neighbor_id`` is a concrete observed entity NOT already bound in ``roles``."""
    bound_eids = set(roles.values())
    ent_map = frame.entity_map()
    links: list[OneHopLink] = []
    for rname, eid in sorted(roles.items()):
        for _s, pred, obj in frame.facts_for(eid):
            if obj in bound_eids:
                continue
            nent = ent_map.get(obj)
            if nent is None or not nent.attributes_observed:
                continue
            links.append(
                OneHopLink(
                    role=rname,
                    via_predicate=pred,
                    neighbor_id=obj,
                    neighbor_type=nent.entity_type,
                )
            )
    return tuple(links)


def extract_role_attribute_map(
    frame: TypedStateFrame,
    roles: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Map each bound role to its observed attribute dictionary in ``frame``."""
    out: dict[str, dict[str, Any]] = {}
    for rname, eid in sorted(roles.items()):
        ef = frame.get_entity(eid)
        if ef is not None and ef.attributes_observed:
            out[rname] = ef.attr_dict()
        else:
            out[rname] = {}
    return out
