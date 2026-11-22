"""Versioned, typed, executable OperatorCard representation for Stage S2 ITL.

STRICT ISOLATION (PDF pp. 6-7, 10, 15-17):
- Contains ZERO hardcoded action rules, ZERO domain-specific transition handlers, and
  NEVER imports ``hidden_actions``, ``oracle``, or ``generators``.
- Represents an induced relational transition schema with typed role constraints,
  direct/conditional/1-hop relational preconditions, attribute and inequality guards,
  sparse fact/attribute effects, preservation invariants, provenance hashes, and
  exact MDL bit accounting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from research.cpu_learning.frames.state_frame import FactEpistemicStatus, TypedStateFrame
from research.cpu_learning.simulator.state_graph import (
    Fact,
    ObservationMask,
    SparseDelta,
    StateGraph,
    canonical_json,
    canonical_json_bytes,
    compressed_bit_length,
    compute_sparse_delta,
    sha256_hex,
)

OPERATOR_CARD_SCHEMA_VERSION: str = "s2-operator-card-v1"


def canonical_value_str(value: Any) -> str:
    """Serialize attribute/literal values canonically ('true'/'false' for bool)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


@dataclass(frozen=True)
class TypeTaxonomy:
    """Optional public L2 entity-type supertype registry (pure ontology, zero action rules)."""

    supertypes: tuple[tuple[str, frozenset[str]], ...] = ()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Sequence[str] | frozenset[str] | set[str]]) -> TypeTaxonomy:
        norm = tuple(
            sorted((str(super_t), frozenset(str(s) for s in subs)) for super_t, subs in mapping.items())
        )
        return cls(supertypes=norm)

    def canonical_type_label(self, allowed_types: frozenset[str]) -> str:
        """Return singleton type, matching supertype name, or deterministic sorted union."""
        if len(allowed_types) == 1:
            return next(iter(allowed_types))
        for super_name, subtypes in self.supertypes:
            if allowed_types == subtypes:
                return super_name
        return "|".join(sorted(allowed_types))


# Public L2 world-type supertype hierarchy shared by relational frames (NOT an action rule).
DEFAULT_WORLD_TYPE_TAXONOMY = TypeTaxonomy.from_mapping(
    {"location": frozenset({"surface", "container"})}
)


@dataclass(frozen=True)
class ConditionalRoleFactGuard:
    """Fact precondition that applies to a role only when its entity has ``when_entity_type``."""

    role: str
    when_entity_type: str
    predicate: str
    value: str

    def to_schema_clause(self) -> tuple[str, str, str]:
        return (f"?{self.role}_if_{self.when_entity_type}", self.predicate, self.value)

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "when_entity_type": self.when_entity_type,
            "predicate": self.predicate,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConditionalRoleFactGuard:
        return cls(
            role=str(data["role"]),
            when_entity_type=str(data["when_entity_type"]),
            predicate=str(data["predicate"]),
            value=str(data["value"]),
        )


@dataclass(frozen=True)
class IndirectRoleFactGuard:
    """1-hop relational guard: if ``?role`` links via ``via_predicate`` to an entity of
    ``when_neighbor_type``, that neighbor must satisfy ``(neighbor, predicate, value)``."""

    role: str
    via_predicate: str
    when_neighbor_type: str
    predicate: str
    value: str

    def to_schema_clause(self) -> tuple[str, str, str]:
        return (f"?{self.role}_{self.when_neighbor_type}_if_any", self.predicate, self.value)

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "via_predicate": self.via_predicate,
            "when_neighbor_type": self.when_neighbor_type,
            "predicate": self.predicate,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> IndirectRoleFactGuard:
        return cls(
            role=str(data["role"]),
            via_predicate=str(data["via_predicate"]),
            when_neighbor_type=str(data["when_neighbor_type"]),
            predicate=str(data["predicate"]),
            value=str(data["value"]),
        )


@dataclass(frozen=True)
class AttributeGuard:
    """Required entity attribute value on a bound role."""

    role: str
    attribute: str
    expected_value: Any

    def to_schema_clause(self) -> tuple[str, str, str]:
        return (f"?{self.role}", self.attribute, canonical_value_str(self.expected_value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "attribute": self.attribute,
            "expected_value": self.expected_value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AttributeGuard:
        return cls(
            role=str(data["role"]),
            attribute=str(data["attribute"]),
            expected_value=data["expected_value"],
        )


@dataclass(frozen=True)
class RoleInequalityGuard:
    """Required non-identity constraint between two type-compatible roles (?role_a != ?role_b)."""

    role_a: str
    role_b: str

    def to_schema_clause(self) -> tuple[str, str, str]:
        return (f"?{self.role_a}", "neq", f"?{self.role_b}")

    def to_dict(self) -> dict[str, str]:
        return {"role_a": self.role_a, "role_b": self.role_b}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RoleInequalityGuard:
        return cls(role_a=str(data["role_a"]), role_b=str(data["role_b"]))


@dataclass(frozen=True)
class LiftedAttributeEdit:
    """Lifted entity attribute update emitted when an OperatorCard fires."""

    role: str
    attribute: str
    new_value: Any

    def to_schema_clause(self) -> tuple[str, str, str]:
        return (f"?{self.role}", self.attribute, canonical_value_str(self.new_value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "attribute": self.attribute,
            "new_value": self.new_value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LiftedAttributeEdit:
        return cls(
            role=str(data["role"]),
            attribute=str(data["attribute"]),
            new_value=data["new_value"],
        )


@dataclass(frozen=True)
class OperatorApplicationResult:
    """Deterministic result of executing an OperatorCard on a StateGraph."""

    after_state: StateGraph | None
    delta: SparseDelta | None
    applied: bool
    abstained: bool
    reason: str


def _resolve_var(term: str, roles: Mapping[str, str]) -> str:
    if term.startswith("?"):
        rname = term[1:]
        if rname not in roles:
            raise KeyError(f"Unbound role variable {term!r}")
        return roles[rname]
    return term


@dataclass(frozen=True)
class OperatorCard:
    """Immutable, versioned ITL OperatorCard (PDF p. 7) induced from public EpisodeLedger records."""

    card_id: str
    action_id: str
    role_variables: tuple[str, ...]
    required_types: tuple[tuple[str, frozenset[str]], ...]
    required_facts: tuple[tuple[str, str, str], ...]
    conditional_fact_guards: tuple[ConditionalRoleFactGuard, ...]
    indirect_fact_guards: tuple[IndirectRoleFactGuard, ...]
    attribute_guards: tuple[AttributeGuard, ...]
    inequality_guards: tuple[RoleInequalityGuard, ...]
    add_facts: tuple[tuple[str, str, str], ...]
    delete_facts: tuple[tuple[str, str, str], ...]
    attribute_edits: tuple[LiftedAttributeEdit, ...]
    preserved_invariants: tuple[str, ...] = (
        "preserve_unmentioned_facts",
        "preserve_unmentioned_attributes",
        "preserve_unrelated_entities",
        "preserve_unobserved_under_mask",
    )
    scope: str = "general_relational"
    confidence: float = 1.0
    support_positive: int = 0
    support_negative: int = 0
    version: int = 1
    parent_version: int | None = None
    provenance_episode_ids: tuple[str, ...] = ()
    provenance_record_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.card_id or not self.action_id:
            raise ValueError("card_id and action_id must be non-empty strings")
        object.__setattr__(self, "role_variables", tuple(sorted(str(r) for r in self.role_variables)))
        norm_types = tuple(
            sorted((str(r), frozenset(str(t) for t in ts)) for r, ts in self.required_types)
        )
        object.__setattr__(self, "required_types", norm_types)
        object.__setattr__(
            self,
            "required_facts",
            tuple(sorted({(str(s), str(p), str(o)) for s, p, o in self.required_facts})),
        )
        object.__setattr__(
            self,
            "conditional_fact_guards",
            tuple(
                sorted(
                    set(self.conditional_fact_guards),
                    key=lambda g: (g.role, g.when_entity_type, g.predicate, g.value),
                )
            ),
        )
        object.__setattr__(
            self,
            "indirect_fact_guards",
            tuple(
                sorted(
                    set(self.indirect_fact_guards),
                    key=lambda g: (g.role, g.via_predicate, g.when_neighbor_type, g.predicate, g.value),
                )
            ),
        )
        object.__setattr__(
            self,
            "attribute_guards",
            tuple(
                sorted(
                    set(self.attribute_guards),
                    key=lambda g: (g.role, g.attribute, canonical_value_str(g.expected_value)),
                )
            ),
        )
        object.__setattr__(
            self,
            "inequality_guards",
            tuple(sorted(set(self.inequality_guards), key=lambda g: (g.role_a, g.role_b))),
        )
        object.__setattr__(
            self,
            "add_facts",
            tuple(sorted({(str(s), str(p), str(o)) for s, p, o in self.add_facts})),
        )
        object.__setattr__(
            self,
            "delete_facts",
            tuple(sorted({(str(s), str(p), str(o)) for s, p, o in self.delete_facts})),
        )
        object.__setattr__(
            self,
            "attribute_edits",
            tuple(
                sorted(
                    set(self.attribute_edits),
                    key=lambda e: (e.role, e.attribute, canonical_value_str(e.new_value)),
                )
            ),
        )
        object.__setattr__(self, "preserved_invariants", tuple(sorted(set(self.preserved_invariants))))
        object.__setattr__(self, "provenance_episode_ids", tuple(sorted(set(self.provenance_episode_ids))))
        object.__setattr__(self, "provenance_record_hashes", tuple(sorted(set(self.provenance_record_hashes))))
        object.__setattr__(self, "confidence", round(float(self.confidence), 6))

    def allowed_types_map(self) -> dict[str, frozenset[str]]:
        return dict(self.required_types)

    def check_preconditions(
        self,
        state: StateGraph,
        roles: Mapping[str, str],
        mask: ObservationMask | None = None,
    ) -> tuple[bool, str, bool]:
        """Verify all induced preconditions on ``state``.

        Returns ``(satisfied, reason, epistemic_unknown)``.
        When ``mask`` hides a required entity, attribute, or fact, ``epistemic_unknown=True``
        so callers can honestly abstain under partial observability.
        """
        obs = mask or ObservationMask(fully_observed=True)
        frame = TypedStateFrame.from_state_graph(state, mask=obs) if not obs.fully_observed else None

        # 1. Required roles present and bound to existing entities with matching types
        for rname, allowed in self.required_types:
            if rname not in roles:
                return (False, f"blocked:missing_role:{rname}", False)
            eid = roles[rname]
            if not obs.is_entity_observed(eid):
                return (False, f"unknown:occluded_role_entity:{rname}", True)
            ent = state.get_entity(eid)
            if ent is None:
                return (False, f"blocked:unknown_entity:{rname}", False)
            if allowed and ent.entity_type not in allowed:
                return (False, f"blocked:type_mismatch:{rname}:{ent.entity_type}", False)

        # Ensure all role_variables exist even if untyped
        for rname in self.role_variables:
            if rname not in roles:
                return (False, f"blocked:missing_role:{rname}", False)

        # 2. Inequality guards
        for ig in self.inequality_guards:
            if roles.get(ig.role_a) == roles.get(ig.role_b):
                return (False, f"blocked:role_equality:{ig.role_a}=={ig.role_b}", False)

        # 3. Attribute guards
        for ag in self.attribute_guards:
            eid = roles[ag.role]
            if not obs.is_entity_observed(eid):
                return (False, f"unknown:occluded_attr:{ag.role}.{ag.attribute}", True)
            ent = state.get_entity(eid)
            if ent is None:
                return (False, f"blocked:unknown_entity:{ag.role}", False)
            if ent.get(ag.attribute) != ag.expected_value:
                return (False, f"blocked:attr_guard:{ag.role}.{ag.attribute}", False)

        # 4. Direct fact preconditions
        for s_t, pred, o_t in self.required_facts:
            cs = _resolve_var(s_t, roles)
            co = _resolve_var(o_t, roles)
            fact: Fact = (cs, pred, co)
            if frame is not None:
                status = frame.query_fact_status(fact)
                if status == FactEpistemicStatus.UNOBSERVED:
                    return (False, f"unknown:occluded_fact:{pred}", True)
                if status != FactEpistemicStatus.OBSERVED_PRESENT:
                    return (False, f"blocked:missing_fact:{s_t}.{pred}={o_t}", False)
            elif not state.has_fact(cs, pred, co):
                return (False, f"blocked:missing_fact:{s_t}.{pred}={o_t}", False)

        # 5. Type-conditional role fact guards
        for cg in self.conditional_fact_guards:
            eid = roles[cg.role]
            if not obs.is_entity_observed(eid):
                return (False, f"unknown:occluded_cond_entity:{cg.role}", True)
            ent = state.get_entity(eid)
            if ent is None or ent.entity_type != cg.when_entity_type:
                continue
            fact = (eid, cg.predicate, cg.value)
            if frame is not None:
                status = frame.query_fact_status(fact)
                if status == FactEpistemicStatus.UNOBSERVED:
                    return (False, f"unknown:occluded_cond_fact:{cg.role}.{cg.predicate}", True)
                if status != FactEpistemicStatus.OBSERVED_PRESENT:
                    return (
                        False,
                        f"blocked:cond_guard:{cg.role}_if_{cg.when_entity_type}.{cg.predicate}",
                        False,
                    )
            elif not state.has_fact(eid, cg.predicate, cg.value):
                return (
                    False,
                    f"blocked:cond_guard:{cg.role}_if_{cg.when_entity_type}.{cg.predicate}",
                    False,
                )

        # 6. Indirect 1-hop relational guards
        for ind in self.indirect_fact_guards:
            eid = roles[ind.role]
            if not obs.fully_observed and not obs.is_fact_observed((eid, ind.via_predicate, "")):
                return (False, f"unknown:occluded_link:{ind.role}.{ind.via_predicate}", True)
            for _s, _p, neighbor_id in state.facts_for(eid, ind.via_predicate):
                if not obs.is_entity_observed(neighbor_id):
                    return (False, f"unknown:occluded_neighbor:{neighbor_id}", True)
                nent = state.get_entity(neighbor_id)
                if nent is None or nent.entity_type != ind.when_neighbor_type:
                    continue
                nfact: Fact = (neighbor_id, ind.predicate, ind.value)
                if frame is not None:
                    status = frame.query_fact_status(nfact)
                    if status == FactEpistemicStatus.UNOBSERVED:
                        return (False, f"unknown:occluded_indirect_fact:{ind.predicate}", True)
                    if status != FactEpistemicStatus.OBSERVED_PRESENT:
                        return (
                            False,
                            f"blocked:indirect_guard:{ind.role}.{ind.via_predicate}.{ind.predicate}",
                            False,
                        )
                elif not state.has_fact(neighbor_id, ind.predicate, ind.value):
                    return (
                        False,
                        f"blocked:indirect_guard:{ind.role}.{ind.via_predicate}.{ind.predicate}",
                        False,
                    )

        return (True, "applied", False)

    def apply(
        self,
        state: StateGraph,
        roles: Mapping[str, str],
        mask: ObservationMask | None = None,
        *,
        abstain_on_unknown: bool = True,
    ) -> OperatorApplicationResult:
        """Execute this induced OperatorCard on ``state`` preserving all unrelated facts/attributes."""
        ok, reason, unknown = self.check_preconditions(state, roles, mask=mask)
        if unknown and abstain_on_unknown:
            return OperatorApplicationResult(
                after_state=None,
                delta=None,
                applied=False,
                abstained=True,
                reason=reason,
            )
        if not ok:
            after = state.copy()
            return OperatorApplicationResult(
                after_state=after,
                delta=SparseDelta(preserved_fact_count=len(state.facts)),
                applied=False,
                abstained=False,
                reason=reason,
            )

        del_concrete = tuple(
            (_resolve_var(s, roles), p, _resolve_var(o, roles)) for s, p, o in self.delete_facts
        )
        add_concrete = tuple(
            (_resolve_var(s, roles), p, _resolve_var(o, roles)) for s, p, o in self.add_facts
        )
        edit_concrete = tuple(
            (roles[e.role], e.attribute, e.new_value) for e in self.attribute_edits
        )
        after = state.with_changes(
            delete_facts=del_concrete,
            add_facts=add_concrete,
            attribute_edits=edit_concrete,
        )
        delta = compute_sparse_delta(state, after, mask=mask)
        return OperatorApplicationResult(
            after_state=after,
            delta=delta,
            applied=True,
            abstained=False,
            reason="applied",
        )

    def to_schema_dict(self, taxonomy: TypeTaxonomy | None = None) -> dict[str, Any]:
        """Export induced operator schema in the evaluator-compatible canonical clause format."""
        tax = taxonomy if taxonomy is not None else TypeTaxonomy()
        req_facts = set(self.required_facts)
        for cg in self.conditional_fact_guards:
            req_facts.add(cg.to_schema_clause())
        for ind in self.indirect_fact_guards:
            req_facts.add(ind.to_schema_clause())

        req_types = {
            (f"?{rname}", tax.canonical_type_label(allowed))
            for rname, allowed in self.required_types
            if allowed
        }
        req_attr_guards = {ag.to_schema_clause() for ag in self.attribute_guards}
        for ig in self.inequality_guards:
            req_attr_guards.add(ig.to_schema_clause())

        return {
            "required_facts": sorted(req_facts),
            "required_types": sorted(req_types),
            "required_attr_guards": sorted(req_attr_guards),
            "add_facts": sorted(self.add_facts),
            "delete_facts": sorted(self.delete_facts),
            "attribute_edits": sorted(e.to_schema_clause() for e in self.attribute_edits),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OPERATOR_CARD_SCHEMA_VERSION,
            "card_id": self.card_id,
            "action_id": self.action_id,
            "role_variables": list(self.role_variables),
            "required_types": {r: sorted(ts) for r, ts in self.required_types},
            "required_facts": [list(f) for f in self.required_facts],
            "conditional_fact_guards": [g.to_dict() for g in self.conditional_fact_guards],
            "indirect_fact_guards": [g.to_dict() for g in self.indirect_fact_guards],
            "attribute_guards": [g.to_dict() for g in self.attribute_guards],
            "inequality_guards": [g.to_dict() for g in self.inequality_guards],
            "add_facts": [list(f) for f in self.add_facts],
            "delete_facts": [list(f) for f in self.delete_facts],
            "attribute_edits": [e.to_dict() for e in self.attribute_edits],
            "preserved_invariants": list(self.preserved_invariants),
            "scope": self.scope,
            "confidence": self.confidence,
            "support_counts": {
                "positive": self.support_positive,
                "negative": self.support_negative,
            },
            "version": self.version,
            "parent_version": self.parent_version,
            "provenance": {
                "episode_ids": list(self.provenance_episode_ids),
                "record_hashes": list(self.provenance_record_hashes),
            },
        }

    def to_executable_dict(self) -> dict[str, Any]:
        """Minimal executable representation (excluding verbose provenance logs) for MDL measurement."""
        return {
            "action_id": self.action_id,
            "role_variables": list(self.role_variables),
            "required_types": {r: sorted(ts) for r, ts in self.required_types},
            "required_facts": [list(f) for f in self.required_facts],
            "conditional_fact_guards": [g.to_dict() for g in self.conditional_fact_guards],
            "indirect_fact_guards": [g.to_dict() for g in self.indirect_fact_guards],
            "attribute_guards": [g.to_dict() for g in self.attribute_guards],
            "inequality_guards": [g.to_dict() for g in self.inequality_guards],
            "add_facts": [list(f) for f in self.add_facts],
            "delete_facts": [list(f) for f in self.delete_facts],
            "attribute_edits": [e.to_dict() for e in self.attribute_edits],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OperatorCard:
        supp = data.get("support_counts", {})
        prov = data.get("provenance", {})
        return cls(
            card_id=str(data["card_id"]),
            action_id=str(data["action_id"]),
            role_variables=tuple(str(r) for r in data.get("role_variables", ())),
            required_types=tuple(
                (str(r), frozenset(str(t) for t in ts))
                for r, ts in data.get("required_types", {}).items()
            ),
            required_facts=tuple(
                (str(f[0]), str(f[1]), str(f[2])) for f in data.get("required_facts", ())
            ),
            conditional_fact_guards=tuple(
                ConditionalRoleFactGuard.from_dict(g)
                for g in data.get("conditional_fact_guards", ())
            ),
            indirect_fact_guards=tuple(
                IndirectRoleFactGuard.from_dict(g) for g in data.get("indirect_fact_guards", ())
            ),
            attribute_guards=tuple(
                AttributeGuard.from_dict(g) for g in data.get("attribute_guards", ())
            ),
            inequality_guards=tuple(
                RoleInequalityGuard.from_dict(g) for g in data.get("inequality_guards", ())
            ),
            add_facts=tuple((str(f[0]), str(f[1]), str(f[2])) for f in data.get("add_facts", ())),
            delete_facts=tuple(
                (str(f[0]), str(f[1]), str(f[2])) for f in data.get("delete_facts", ())
            ),
            attribute_edits=tuple(
                LiftedAttributeEdit.from_dict(e) for e in data.get("attribute_edits", ())
            ),
            preserved_invariants=tuple(str(x) for x in data.get("preserved_invariants", ())),
            scope=str(data.get("scope", "general_relational")),
            confidence=float(data.get("confidence", 1.0)),
            support_positive=int(supp.get("positive", 0)),
            support_negative=int(supp.get("negative", 0)),
            version=int(data.get("version", 1)),
            parent_version=data.get("parent_version"),
            provenance_episode_ids=tuple(str(x) for x in prov.get("episode_ids", ())),
            provenance_record_hashes=tuple(str(x) for x in prov.get("record_hashes", ())),
        )

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def fingerprint(self) -> str:
        return sha256_hex(canonical_json_bytes(self.to_executable_dict()))

    def executable_bit_length_raw(self) -> int:
        return len(canonical_json_bytes(self.to_executable_dict())) * 8

    def executable_bit_length_compressed(self) -> int:
        return compressed_bit_length(canonical_json_bytes(self.to_executable_dict()))
