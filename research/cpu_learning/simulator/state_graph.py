"""Public relational state-graph, sparse-delta, and episode record data model.

Strictly public representation types only: contains NO hidden action rules,
NO edit scripts, and NO gold transition functions.
"""
from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

Fact = tuple[str, str, str]
AttributeEdit = tuple[str, str, Any, Any]  # (entity_id, attr_name, old_val, new_val)


def canonical_json(value: Any) -> str:
    """Deterministic UTF-8 JSON serialization with sorted keys and compact separators."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_json_bytes(value: Any) -> bytes:
    """UTF-8 encoded canonical JSON bytes."""
    return canonical_json(value).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Lowercase hex SHA-256 digest."""
    return hashlib.sha256(data).hexdigest()


def compressed_bit_length(data: bytes, level: int = 9) -> int:
    """Deterministic zlib-compressed size in bits for MDL / memory-growth audits."""
    if not data:
        return 0
    return len(zlib.compress(data, level=level)) * 8


@dataclass(frozen=True)
class Entity:
    """Typed world entity with immutable sorted attribute key/value pairs."""

    entity_id: str
    entity_type: str
    attributes: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.entity_id or not isinstance(self.entity_id, str):
            raise ValueError("entity_id must be a non-empty string")
        if not self.entity_type or not isinstance(self.entity_type, str):
            raise ValueError("entity_type must be a non-empty string")
        sorted_attrs = tuple(sorted((str(k), v) for k, v in self.attributes))
        object.__setattr__(self, "attributes", sorted_attrs)

    def attr_dict(self) -> dict[str, Any]:
        return dict(self.attributes)

    def get(self, key: str, default: Any = None) -> Any:
        for k, v in self.attributes:
            if k == key:
                return v
        return default

    def with_attr(self, key: str, value: Any) -> Entity:
        updated = dict(self.attributes)
        updated[str(key)] = value
        return Entity(
            entity_id=self.entity_id,
            entity_type=self.entity_type,
            attributes=tuple(sorted(updated.items())),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.entity_type}
        for k, v in self.attributes:
            out[k] = v
        return out

    @classmethod
    def from_dict(cls, entity_id: str, data: Mapping[str, Any]) -> Entity:
        etype = str(data["type"])
        attrs = tuple(sorted((str(k), v) for k, v in data.items() if k != "type"))
        return cls(entity_id=entity_id, entity_type=etype, attributes=attrs)


@dataclass(frozen=True)
class ObservationMask:
    """Specifies which entities and predicates are observable in a state.

    Unmentioned or unobserved facts are preserved by default; absence under a
    partial mask is NOT evidence that a fact became false.
    """

    observed_entities: frozenset[str] = frozenset()
    observed_predicates: frozenset[str] = frozenset()
    fully_observed: bool = True

    def is_entity_observed(self, entity_id: str) -> bool:
        if self.fully_observed:
            return True
        return entity_id in self.observed_entities

    def is_fact_observed(self, fact: Fact) -> bool:
        if self.fully_observed:
            return True
        subj, pred, _ = fact
        ent_ok = (not self.observed_entities) or (subj in self.observed_entities)
        pred_ok = (not self.observed_predicates) or (pred in self.observed_predicates)
        return ent_ok and pred_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "fully_observed": self.fully_observed,
            "observed_entities": sorted(self.observed_entities),
            "observed_predicates": sorted(self.observed_predicates),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ObservationMask:
        return cls(
            observed_entities=frozenset(data.get("observed_entities", ())),
            observed_predicates=frozenset(data.get("observed_predicates", ())),
            fully_observed=bool(data.get("fully_observed", True)),
        )


@dataclass(frozen=True)
class StateGraph:
    """Typed relational world state: entities with attributes + directed facts."""

    entities: tuple[tuple[str, Entity], ...]
    facts: frozenset[Fact]

    def __post_init__(self) -> None:
        sorted_entities = tuple(sorted(self.entities, key=lambda pair: pair[0]))
        object.__setattr__(self, "entities", sorted_entities)
        normalized_facts = frozenset((str(s), str(p), str(o)) for s, p, o in self.facts)
        object.__setattr__(self, "facts", normalized_facts)

    @classmethod
    def create(
        cls,
        entities: Mapping[str, Entity] | Iterable[Entity],
        facts: Iterable[Fact],
    ) -> StateGraph:
        if isinstance(entities, Mapping):
            ent_pairs = tuple((eid, ent) for eid, ent in entities.items())
        else:
            ent_pairs = tuple((ent.entity_id, ent) for ent in entities)
        return cls(entities=ent_pairs, facts=frozenset(facts))

    def copy(self) -> StateGraph:
        return StateGraph(entities=self.entities, facts=self.facts)

    def entity_map(self) -> dict[str, Entity]:
        return dict(self.entities)

    def get_entity(self, entity_id: str) -> Entity | None:
        for eid, ent in self.entities:
            if eid == entity_id:
                return ent
        return None

    def has_fact(self, subj: str, pred: str, obj: str) -> bool:
        return (subj, pred, obj) in self.facts

    def facts_for(self, subj: str, pred: str | None = None) -> tuple[Fact, ...]:
        matched = [
            f
            for f in self.facts
            if f[0] == subj and (pred is None or f[1] == pred)
        ]
        return tuple(sorted(matched))

    def with_changes(
        self,
        *,
        add_facts: Iterable[Fact] = (),
        delete_facts: Iterable[Fact] = (),
        attribute_edits: Iterable[tuple[str, str, Any]] = (),
    ) -> StateGraph:
        """Return a new StateGraph applying explicit fact additions/deletions and attribute edits."""
        new_facts = (set(self.facts) - set(delete_facts)) | set(add_facts)
        ent_dict = self.entity_map()
        for eid, attr_name, new_val in attribute_edits:
            if eid in ent_dict:
                ent_dict[eid] = ent_dict[eid].with_attr(attr_name, new_val)
        return StateGraph.create(ent_dict, new_facts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": {eid: ent.to_dict() for eid, ent in self.entities},
            "facts": [list(f) for f in sorted(self.facts)],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StateGraph:
        ent_map = {
            str(eid): Entity.from_dict(str(eid), edata)
            for eid, edata in data.get("entities", {}).items()
        }
        facts = [
            (str(f[0]), str(f[1]), str(f[2]))
            for f in data.get("facts", ())
        ]
        return cls.create(ent_map, facts)

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def fingerprint(self) -> str:
        return sha256_hex(self.canonical_json().encode("utf-8"))

    def bit_length_raw(self) -> int:
        return len(self.canonical_json().encode("utf-8")) * 8

    def bit_length_compressed(self) -> int:
        return compressed_bit_length(self.canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class SparseDelta:
    """Exact difference between before_state and after_state under an observation mask."""

    added_facts: tuple[Fact, ...] = ()
    deleted_facts: tuple[Fact, ...] = ()
    attribute_edits: tuple[AttributeEdit, ...] = ()
    preserved_fact_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "added_facts", tuple(sorted(self.added_facts)))
        object.__setattr__(self, "deleted_facts", tuple(sorted(self.deleted_facts)))
        object.__setattr__(
            self,
            "attribute_edits",
            tuple(sorted(self.attribute_edits, key=lambda e: (e[0], e[1]))),
        )

    @property
    def is_noop(self) -> bool:
        return (
            not self.added_facts
            and not self.deleted_facts
            and not self.attribute_edits
        )

    def apply_to(self, state: StateGraph) -> StateGraph:
        """Apply this concrete delta directly to a StateGraph."""
        attr_updates = [(eid, attr, new_val) for eid, attr, _old, new_val in self.attribute_edits]
        return state.with_changes(
            add_facts=self.added_facts,
            delete_facts=self.deleted_facts,
            attribute_edits=attr_updates,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "added_facts": [list(f) for f in self.added_facts],
            "deleted_facts": [list(f) for f in self.deleted_facts],
            "attribute_edits": [
                {
                    "entity_id": eid,
                    "attribute": attr,
                    "old": old_val,
                    "new": new_val,
                }
                for eid, attr, old_val, new_val in self.attribute_edits
            ],
            "preserved_fact_count": self.preserved_fact_count,
            "is_noop": self.is_noop,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SparseDelta:
        added = tuple((str(f[0]), str(f[1]), str(f[2])) for f in data.get("added_facts", ()))
        deleted = tuple((str(f[0]), str(f[1]), str(f[2])) for f in data.get("deleted_facts", ()))
        edits = tuple(
            (str(e["entity_id"]), str(e["attribute"]), e["old"], e["new"])
            for e in data.get("attribute_edits", ())
        )
        return cls(
            added_facts=added,
            deleted_facts=deleted,
            attribute_edits=edits,
            preserved_fact_count=int(data.get("preserved_fact_count", 0)),
        )


def compute_sparse_delta(
    before: StateGraph,
    after: StateGraph,
    mask: ObservationMask | None = None,
) -> SparseDelta:
    """Compute facts added/deleted, attribute changes, and preserved fact count."""
    obs = mask or ObservationMask(fully_observed=True)
    before_facts = {f for f in before.facts if obs.is_fact_observed(f)}
    after_facts = {f for f in after.facts if obs.is_fact_observed(f)}

    added = tuple(sorted(after_facts - before_facts))
    deleted = tuple(sorted(before_facts - after_facts))
    preserved = len(before_facts & after_facts)

    before_ents = before.entity_map()
    after_ents = after.entity_map()
    attr_edits: list[AttributeEdit] = []
    for eid in sorted(set(before_ents) | set(after_ents)):
        if not obs.is_entity_observed(eid):
            continue
        b_attrs = before_ents[eid].attr_dict() if eid in before_ents else {}
        a_attrs = after_ents[eid].attr_dict() if eid in after_ents else {}
        for attr_name in sorted(set(b_attrs) | set(a_attrs)):
            old_v = b_attrs.get(attr_name)
            new_v = a_attrs.get(attr_name)
            if old_v != new_v:
                attr_edits.append((eid, attr_name, old_v, new_v))

    return SparseDelta(
        added_facts=added,
        deleted_facts=deleted,
        attribute_edits=tuple(attr_edits),
        preserved_fact_count=preserved,
    )


@dataclass(frozen=True)
class ActionInvocation:
    """Typed action call paired with raw UTF-8 instruction bytes and exact byte spans."""

    action_id: str
    roles: tuple[tuple[str, str], ...]
    raw_instruction_bytes: bytes
    byte_spans: tuple[tuple[str, int, int], ...]

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must be non-empty")
        object.__setattr__(
            self,
            "roles",
            tuple(sorted((str(k), str(v)) for k, v in self.roles)),
        )
        raw = bytes(self.raw_instruction_bytes)
        object.__setattr__(self, "raw_instruction_bytes", raw)
        spans = tuple(sorted((str(k), int(s), int(e)) for k, s, e in self.byte_spans))
        for key, start, end in spans:
            if not (0 <= start <= end <= len(raw)):
                raise ValueError(
                    f"Invalid byte span for {key!r}: ({start}, {end}) vs len={len(raw)}"
                )
        object.__setattr__(self, "byte_spans", spans)

    def role_dict(self) -> dict[str, str]:
        return dict(self.roles)

    def instruction_text(self) -> str:
        return self.raw_instruction_bytes.decode("utf-8")

    def span_dict(self) -> dict[str, tuple[int, int]]:
        return {k: (s, e) for k, s, e in self.byte_spans}

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "roles": dict(self.roles),
            "raw_instruction_utf8": self.instruction_text(),
            "raw_instruction_byte_length": len(self.raw_instruction_bytes),
            "raw_instruction_sha256": sha256_hex(self.raw_instruction_bytes),
            "byte_spans": {k: [s, e] for k, s, e in self.byte_spans},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ActionInvocation:
        raw = str(data["raw_instruction_utf8"]).encode("utf-8")
        roles = tuple((str(k), str(v)) for k, v in data.get("roles", {}).items())
        spans = tuple(
            (str(k), int(v[0]), int(v[1]))
            for k, v in data.get("byte_spans", {}).items()
        )
        return cls(
            action_id=str(data["action_id"]),
            roles=roles,
            raw_instruction_bytes=raw,
            byte_spans=spans,
        )

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())


@dataclass(frozen=True)
class TrainingEpisode:
    """Auditable before/action/after training demonstration (NO hidden rule fields)."""

    episode_id: str
    before_state: StateGraph
    action: ActionInvocation
    observation_mask: ObservationMask
    after_state: StateGraph
    sparse_delta: SparseDelta
    context: tuple[tuple[str, Any], ...]
    observation_source: str
    split_id: str
    generator_id: str
    generator_version: str
    seed: int
    timestamp: str

    def state_action_key(self) -> str:
        """Canonical exact-match key used to audit test-time episode-store retrieval."""
        payload = {
            "before_state_fp": self.before_state.fingerprint(),
            "action_id": self.action.action_id,
            "roles": dict(self.action.roles),
        }
        return sha256_hex(canonical_json_bytes(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.episode_id,
            "raw_instruction_utf8": self.action.instruction_text(),
            "raw_instruction_byte_length": len(self.action.raw_instruction_bytes),
            "byte_spans": {k: [s, e] for k, s, e in self.action.byte_spans},
            "before_state": self.before_state.to_dict(),
            "action": self.action.to_dict(),
            "observation_mask": self.observation_mask.to_dict(),
            "after_state": self.after_state.to_dict(),
            "sparse_delta": self.sparse_delta.to_dict(),
            "context": dict(self.context),
            "observation_source": self.observation_source,
            "split_id": self.split_id,
            "generator_id": self.generator_id,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class EvaluationTaskPublic:
    """Public held-out task input passed to learners/baselines (zero gold labels)."""

    task_id: str
    split_id: str
    generator_id: str
    generator_version: str
    seed: int
    depth: int
    initial_state: StateGraph
    actions: tuple[ActionInvocation, ...]
    observation_mask: ObservationMask
    metadata: tuple[tuple[str, Any], ...] = ()

    def step_state_action_key(self, state: StateGraph, step_idx: int) -> str:
        act = self.actions[step_idx]
        payload = {
            "before_state_fp": state.fingerprint(),
            "action_id": act.action_id,
            "roles": dict(act.roles),
        }
        return sha256_hex(canonical_json_bytes(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "split_id": self.split_id,
            "generator_id": self.generator_id,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "depth": self.depth,
            "initial_state": self.initial_state.to_dict(),
            "actions": [a.to_dict() for a in self.actions],
            "observation_mask": self.observation_mask.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EvaluationTaskHiddenLabel:
    """Evaluator-only ground truth for an evaluation task. Never exposed to learners."""

    task_id: str
    gold_final_state: StateGraph
    gold_intermediate_states: tuple[StateGraph, ...]
    gold_step_deltas: tuple[SparseDelta, ...]
    gold_step_applied: tuple[bool, ...]
    gold_block_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "gold_final_state": self.gold_final_state.to_dict(),
            "gold_intermediate_states": [s.to_dict() for s in self.gold_intermediate_states],
            "gold_step_deltas": [d.to_dict() for d in self.gold_step_deltas],
            "gold_step_applied": list(self.gold_step_applied),
            "gold_block_reasons": list(self.gold_block_reasons),
        }


@dataclass(frozen=True)
class EvaluationCase:
    """Evaluator-only pairing of public task and hidden ground-truth label."""

    public: EvaluationTaskPublic
    hidden: EvaluationTaskHiddenLabel