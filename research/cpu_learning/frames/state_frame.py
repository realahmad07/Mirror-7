"""Learner-facing L2 Typed State Frames, Partial ObservationMask Ingestion & Candidate Role Frames.

STRICT ISOLATION (PDF pp. 4-6, 10, 15):
- Receives ONLY public ``StateGraph`` / ``ObservationMask`` observations and raw UTF-8 bytes.
- NEVER imports ``hidden_actions``, ``oracle``, or ``generators``.
- Enforces the epistemic preservation invariant: unmentioned/occluded facts under a partial
  ``ObservationMask`` are ``UNOBSERVED``, NEVER ``false`` and NEVER emitted as causal deletions.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from research.cpu_learning.simulator.state_graph import (
    AttributeEdit,
    Entity,
    Fact,
    ObservationMask,
    SparseDelta,
    StateGraph,
    canonical_json,
    canonical_json_bytes,
    compressed_bit_length,
    sha256_hex,
)

_FORBIDDEN_KEY_PREFIXES: tuple[str, ...] = ("gold_", "hidden_")
_FORBIDDEN_EXACT_KEYS: frozenset[str] = frozenset(
    {
        "hidden_actions",
        "hidden_operator_schemas",
        "expected_transition_kind",
        "gold_final_state",
        "gold_intermediate_states",
        "gold_step_deltas",
        "gold_step_applied",
        "gold_block_reasons",
    }
)


def assert_no_hidden_fields(payload: Any, *, path: str = "root") -> None:
    """Defense-in-depth recursive check rejecting evaluator-only gold/hidden keys."""
    if isinstance(payload, Mapping):
        for raw_key, val in payload.items():
            key = str(raw_key)
            key_lower = key.casefold()
            if key_lower in _FORBIDDEN_EXACT_KEYS or any(
                key_lower.startswith(pref) for pref in _FORBIDDEN_KEY_PREFIXES
            ):
                raise ValueError(
                    f"Learner-facing ingestion rejected forbidden hidden/gold key {key!r} at {path}"
                )
            assert_no_hidden_fields(val, path=f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for idx, item in enumerate(payload):
            assert_no_hidden_fields(item, path=f"{path}[{idx}]")


class FactEpistemicStatus(str, Enum):
    """Three-valued epistemic status distinguishing unobserved facts from absent/false facts."""

    OBSERVED_PRESENT = "observed_present"
    OBSERVED_ABSENT = "observed_absent"
    UNOBSERVED = "unobserved"


@dataclass(frozen=True)
class TypedEntityFrame:
    """Learner-facing typed view of a single entity under an ObservationMask."""

    entity_id: str
    entity_type: str
    observed_attributes: tuple[tuple[str, Any], ...] = ()
    attributes_observed: bool = True

    def __post_init__(self) -> None:
        if not self.entity_id or not isinstance(self.entity_id, str):
            raise ValueError("entity_id must be a non-empty string")
        if not self.entity_type or not isinstance(self.entity_type, str):
            raise ValueError("entity_type must be a non-empty string")
        seen: set[str] = set()
        normalized: list[tuple[str, Any]] = []
        for k, v in self.observed_attributes:
            sk = str(k)
            if sk == "type":
                raise ValueError("attribute key 'type' is reserved for entity_type")
            if sk in seen:
                raise ValueError(f"Duplicate attribute key {sk!r} on TypedEntityFrame {self.entity_id!r}")
            seen.add(sk)
            normalized.append((sk, v))
        object.__setattr__(self, "observed_attributes", tuple(sorted(normalized, key=lambda p: p[0])))
        object.__setattr__(self, "attributes_observed", bool(self.attributes_observed))

    def attr_dict(self) -> dict[str, Any]:
        return dict(self.observed_attributes)

    def get_attr(self, attr_name: str, default: Any = None) -> Any:
        for k, v in self.observed_attributes:
            if k == attr_name:
                return v
        return default

    def to_entity(self) -> Entity:
        return Entity(
            entity_id=self.entity_id,
            entity_type=self.entity_type,
            attributes=self.observed_attributes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "observed_attributes": dict(self.observed_attributes),
            "attributes_observed": self.attributes_observed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TypedEntityFrame:
        attrs = tuple(sorted((str(k), v) for k, v in data.get("observed_attributes", {}).items()))
        return cls(
            entity_id=str(data["entity_id"]),
            entity_type=str(data["entity_type"]),
            observed_attributes=attrs,
            attributes_observed=bool(data.get("attributes_observed", True)),
        )


@dataclass(frozen=True)
class TypedStateFrame:
    """Learner-facing L2 typed relational state frame projected strictly through ObservationMask.

    Any entity or fact outside ``observation_mask`` is redacted at ingestion time so
    occluded world details cannot leak into the learner, while epistemic queries
    explicitly return ``FactEpistemicStatus.UNOBSERVED`` rather than false.
    """

    entities: tuple[tuple[str, TypedEntityFrame], ...]
    observed_facts: frozenset[Fact]
    observation_mask: ObservationMask
    occluded_entity_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        seen_ids: set[str] = set()
        norm_ents: list[tuple[str, TypedEntityFrame]] = []
        for eid, frame in self.entities:
            seid = str(eid)
            if seid != frame.entity_id:
                raise ValueError(f"Entity key {seid!r} mismatches frame.entity_id {frame.entity_id!r}")
            if seid in seen_ids:
                raise ValueError(f"Duplicate entity_id {seid!r} in TypedStateFrame")
            seen_ids.add(seid)
            norm_ents.append((seid, frame))
        object.__setattr__(self, "entities", tuple(sorted(norm_ents, key=lambda p: p[0])))
        norm_facts = frozenset((str(s), str(p), str(o)) for s, p, o in self.observed_facts)
        for fact in norm_facts:
            if not self.observation_mask.is_fact_observed(fact):
                raise ValueError(
                    f"TypedStateFrame cannot store unobserved fact {fact!r} under mask {self.observation_mask}"
                )
        object.__setattr__(self, "observed_facts", norm_facts)
        object.__setattr__(self, "occluded_entity_ids", frozenset(str(e) for e in self.occluded_entity_ids))

    @classmethod
    def from_state_graph(
        cls,
        state: StateGraph,
        mask: ObservationMask | None = None,
    ) -> TypedStateFrame:
        """Project a public StateGraph through ObservationMask, strictly redacting occluded data."""
        obs = mask or ObservationMask(fully_observed=True)
        obs_facts = frozenset(f for f in state.facts if obs.is_fact_observed(f))

        referenced_in_facts: set[str] = set()
        for subj, _pred, obj in obs_facts:
            referenced_in_facts.add(subj)
            if obj not in ("true", "false"):
                referenced_in_facts.add(obj)

        entity_frames: dict[str, TypedEntityFrame] = {}
        occluded_ids: set[str] = set()
        state_ent_map = state.entity_map()

        for eid, ent in state_ent_map.items():
            if obs.is_entity_observed(eid):
                entity_frames[eid] = TypedEntityFrame(
                    entity_id=eid,
                    entity_type=ent.entity_type,
                    observed_attributes=ent.attributes,
                    attributes_observed=True,
                )
            elif eid in referenced_in_facts:
                entity_frames[eid] = TypedEntityFrame(
                    entity_id=eid,
                    entity_type="unknown",
                    observed_attributes=(),
                    attributes_observed=False,
                )
                occluded_ids.add(eid)

        for ref_id in sorted(referenced_in_facts):
            if ref_id not in entity_frames and not obs.is_entity_observed(ref_id):
                entity_frames[ref_id] = TypedEntityFrame(
                    entity_id=ref_id,
                    entity_type="unknown",
                    observed_attributes=(),
                    attributes_observed=False,
                )
                occluded_ids.add(ref_id)

        return cls(
            entities=tuple((eid, ef) for eid, ef in entity_frames.items()),
            observed_facts=obs_facts,
            observation_mask=obs,
            occluded_entity_ids=frozenset(occluded_ids),
        )

    def entity_map(self) -> dict[str, TypedEntityFrame]:
        return dict(self.entities)

    def get_entity(self, entity_id: str) -> TypedEntityFrame | None:
        for eid, ef in self.entities:
            if eid == entity_id:
                return ef
        return None

    def entity_type_of(self, entity_id: str) -> str | None:
        ef = self.get_entity(entity_id)
        if ef is None:
            return None
        return ef.entity_type

    def resolve_entity_id(self, surface_token: str) -> str | None:
        """Resolve an entity token against observed entities (exact match, then case-insensitive)."""
        for eid, _ef in self.entities:
            if eid == surface_token:
                return eid
        lowered = surface_token.casefold()
        matches = [eid for eid, _ef in self.entities if eid.casefold() == lowered]
        if len(matches) == 1:
            return matches[0]
        return None

    def entities_by_type(self, entity_type: str) -> tuple[str, ...]:
        return tuple(
            eid
            for eid, ef in self.entities
            if ef.attributes_observed and ef.entity_type == entity_type
        )

    def facts_for(self, subj: str, pred: str | None = None) -> tuple[Fact, ...]:
        return tuple(
            sorted(
                f
                for f in self.observed_facts
                if f[0] == subj and (pred is None or f[1] == pred)
            )
        )

    def query_fact_status(self, fact: Fact) -> FactEpistemicStatus:
        """Return OBSERVED_PRESENT, OBSERVED_ABSENT, or UNOBSERVED (never conflating unobserved with false)."""
        norm = (str(fact[0]), str(fact[1]), str(fact[2]))
        if not self.observation_mask.is_fact_observed(norm):
            return FactEpistemicStatus.UNOBSERVED
        if norm in self.observed_facts:
            return FactEpistemicStatus.OBSERVED_PRESENT
        return FactEpistemicStatus.OBSERVED_ABSENT

    def query_attribute_status(self, entity_id: str, attr_name: str) -> tuple[FactEpistemicStatus, Any]:
        """Return (epistemic_status, value) for an entity attribute under observation_mask."""
        if not self.observation_mask.is_entity_observed(entity_id):
            return (FactEpistemicStatus.UNOBSERVED, None)
        ef = self.get_entity(entity_id)
        if ef is None or not ef.attributes_observed:
            return (FactEpistemicStatus.UNOBSERVED, None)
        attrs = ef.attr_dict()
        if attr_name in attrs:
            return (FactEpistemicStatus.OBSERVED_PRESENT, attrs[attr_name])
        return (FactEpistemicStatus.OBSERVED_ABSENT, None)

    def to_observed_state_graph(self) -> StateGraph:
        """Convert the observed projection to a canonical StateGraph."""
        return StateGraph.create(
            entities={eid: ef.to_entity() for eid, ef in self.entities},
            facts=self.observed_facts,
        )

    def apply_delta_preserving_unobserved(self, delta: SparseDelta) -> TypedStateFrame:
        """Apply a SparseDelta strictly within this frame's observed scope.

        Unobserved facts and occluded entities are NEVER deleted or overwritten.
        """
        new_facts = set(self.observed_facts)
        for fact in delta.deleted_facts:
            if self.observation_mask.is_fact_observed(fact):
                new_facts.discard(fact)
        for fact in delta.added_facts:
            if self.observation_mask.is_fact_observed(fact):
                new_facts.add(fact)

        ent_dict = self.entity_map()
        for eid, attr_name, _old_val, new_val in delta.attribute_edits:
            if not self.observation_mask.is_entity_observed(eid):
                continue
            ef = ent_dict.get(eid)
            if ef is None or not ef.attributes_observed:
                continue
            attrs = ef.attr_dict()
            if new_val is None:
                attrs.pop(attr_name, None)
            else:
                attrs[attr_name] = new_val
            ent_dict[eid] = TypedEntityFrame(
                entity_id=ef.entity_id,
                entity_type=ef.entity_type,
                observed_attributes=tuple(sorted(attrs.items())),
                attributes_observed=True,
            )

        return TypedStateFrame(
            entities=tuple(ent_dict.items()),
            observed_facts=frozenset(new_facts),
            observation_mask=self.observation_mask,
            occluded_entity_ids=self.occluded_entity_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": {eid: ef.to_dict() for eid, ef in self.entities},
            "observed_facts": [list(f) for f in sorted(self.observed_facts)],
            "observation_mask": self.observation_mask.to_dict(),
            "occluded_entity_ids": sorted(self.occluded_entity_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TypedStateFrame:
        assert_no_hidden_fields(data, path="TypedStateFrame")
        ent_pairs = tuple(
            (str(eid), TypedEntityFrame.from_dict(edata))
            for eid, edata in data.get("entities", {}).items()
        )
        facts = frozenset(
            (str(f[0]), str(f[1]), str(f[2]))
            for f in data.get("observed_facts", ())
        )
        mask = ObservationMask.from_dict(data["observation_mask"])
        occluded = frozenset(str(e) for e in data.get("occluded_entity_ids", ()))
        return cls(
            entities=ent_pairs,
            observed_facts=facts,
            observation_mask=mask,
            occluded_entity_ids=occluded,
        )

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def fingerprint(self) -> str:
        return sha256_hex(self.canonical_json().encode("utf-8"))

    def bit_length_raw(self) -> int:
        return len(self.canonical_json().encode("utf-8")) * 8

    def bit_length_compressed(self) -> int:
        return compressed_bit_length(self.canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class MaskedDeltaReport:
    """Canonical SparseDelta paired with explicit partial-observability preservation accounting.

    Guarantees that under partial or asymmetric before/after ``ObservationMask``s:
    - Facts observed before but occluded after are recorded in ``occluded_in_after_facts``
      and NEVER emitted in ``sparse_delta.deleted_facts``.
    - Facts occluded before and revealed after are recorded in ``newly_revealed_facts``
      and NEVER emitted in ``sparse_delta.added_facts``.
    - Entities occluded on either side NEVER produce synthetic attribute edits or deletions.
    """

    sparse_delta: SparseDelta
    preserved_facts: tuple[Fact, ...]
    occluded_in_after_facts: tuple[Fact, ...]
    newly_revealed_facts: tuple[Fact, ...]
    occluded_in_after_entities: tuple[str, ...]
    newly_revealed_entities: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sparse_delta": self.sparse_delta.to_dict(),
            "preserved_facts": [list(f) for f in self.preserved_facts],
            "occluded_in_after_facts": [list(f) for f in self.occluded_in_after_facts],
            "newly_revealed_facts": [list(f) for f in self.newly_revealed_facts],
            "occluded_in_after_entities": list(self.occluded_in_after_entities),
            "newly_revealed_entities": list(self.newly_revealed_entities),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MaskedDeltaReport:
        return cls(
            sparse_delta=SparseDelta.from_dict(data["sparse_delta"]),
            preserved_facts=tuple((str(f[0]), str(f[1]), str(f[2])) for f in data.get("preserved_facts", ())),
            occluded_in_after_facts=tuple(
                (str(f[0]), str(f[1]), str(f[2])) for f in data.get("occluded_in_after_facts", ())
            ),
            newly_revealed_facts=tuple(
                (str(f[0]), str(f[1]), str(f[2])) for f in data.get("newly_revealed_facts", ())
            ),
            occluded_in_after_entities=tuple(str(e) for e in data.get("occluded_in_after_entities", ())),
            newly_revealed_entities=tuple(str(e) for e in data.get("newly_revealed_entities", ())),
        )


def extract_masked_sparse_delta(
    before: TypedStateFrame,
    after: TypedStateFrame,
) -> MaskedDeltaReport:
    """Compute canonical SparseDelta strictly over mutually observed facts & entities.

    Enforces the PDF p. 6 preservation invariant: absence of a fact outside the
    observation mask is NEVER treated as evidence that the fact became false.
    """
    b_mask = before.observation_mask
    a_mask = after.observation_mask

    added: list[Fact] = []
    deleted: list[Fact] = []
    preserved: list[Fact] = []
    occluded_in_after_facts: list[Fact] = []
    newly_revealed_facts: list[Fact] = []

    for fact in sorted(before.observed_facts):
        if not a_mask.is_fact_observed(fact):
            occluded_in_after_facts.append(fact)
        elif fact in after.observed_facts:
            preserved.append(fact)
        else:
            deleted.append(fact)

    for fact in sorted(after.observed_facts):
        if not b_mask.is_fact_observed(fact):
            newly_revealed_facts.append(fact)
        elif fact not in before.observed_facts:
            added.append(fact)

    b_ents = before.entity_map()
    a_ents = after.entity_map()
    attr_edits: list[AttributeEdit] = []
    occluded_in_after_entities: list[str] = []
    newly_revealed_entities: list[str] = []

    for eid in sorted(set(b_ents) | set(a_ents)):
        b_obs = b_mask.is_entity_observed(eid) and (eid in b_ents) and b_ents[eid].attributes_observed
        a_obs = a_mask.is_entity_observed(eid) and (eid in a_ents) and a_ents[eid].attributes_observed
        if b_obs and a_obs:
            b_attrs = b_ents[eid].attr_dict()
            a_attrs = a_ents[eid].attr_dict()
            for attr_name in sorted(set(b_attrs) | set(a_attrs)):
                old_v = b_attrs.get(attr_name)
                new_v = a_attrs.get(attr_name)
                if old_v != new_v:
                    attr_edits.append((eid, attr_name, old_v, new_v))
        elif b_obs and not a_mask.is_entity_observed(eid):
            occluded_in_after_entities.append(eid)
        elif a_obs and not b_mask.is_entity_observed(eid):
            newly_revealed_entities.append(eid)

    delta = SparseDelta(
        added_facts=tuple(added),
        deleted_facts=tuple(deleted),
        attribute_edits=tuple(attr_edits),
        preserved_fact_count=len(preserved),
    )
    return MaskedDeltaReport(
        sparse_delta=delta,
        preserved_facts=tuple(preserved),
        occluded_in_after_facts=tuple(occluded_in_after_facts),
        newly_revealed_facts=tuple(newly_revealed_facts),
        occluded_in_after_entities=tuple(occluded_in_after_entities),
        newly_revealed_entities=tuple(newly_revealed_entities),
    )


def verify_unobserved_preservation(
    full_before: StateGraph,
    delta: SparseDelta,
    mask: ObservationMask,
) -> bool:
    """Verify that applying a masked SparseDelta leaves 100% of unobserved facts & attrs intact."""
    applied = delta.apply_to(full_before)
    unobs_before = {f for f in full_before.facts if not mask.is_fact_observed(f)}
    unobs_after = {f for f in applied.facts if not mask.is_fact_observed(f)}
    if unobs_before != unobs_after:
        return False

    b_ents = full_before.entity_map()
    a_ents = applied.entity_map()
    for eid in set(b_ents) | set(a_ents):
        if mask.is_entity_observed(eid):
            continue
        if b_ents.get(eid) != a_ents.get(eid):
            return False
    return True


@dataclass(frozen=True)
class CandidateActionFrame:
    """A single candidate typed action role-frame extracted from raw UTF-8 instruction bytes."""

    action_id: str
    roles: tuple[tuple[str, str], ...]
    byte_spans: tuple[tuple[str, int, int], ...]
    role_types: tuple[tuple[str, str], ...] = ()
    confidence: float = 1.0
    parse_rule: str = "typed_frame"
    ambiguity_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.action_id or not isinstance(self.action_id, str):
            raise ValueError("action_id must be a non-empty string")
        seen_r: set[str] = set()
        norm_roles: list[tuple[str, str]] = []
        for k, v in self.roles:
            sk = str(k)
            if sk in seen_r:
                raise ValueError(f"Duplicate role key {sk!r} in CandidateActionFrame")
            seen_r.add(sk)
            norm_roles.append((sk, str(v)))
        object.__setattr__(self, "roles", tuple(sorted(norm_roles)))

        seen_s: set[str] = set()
        norm_spans: list[tuple[str, int, int]] = []
        for k, s, e in self.byte_spans:
            sk = str(k)
            si, ei = int(s), int(e)
            if sk in seen_s:
                raise ValueError(f"Duplicate byte span key {sk!r} in CandidateActionFrame")
            if si < 0 or ei < si:
                raise ValueError(f"Invalid byte span ({si}, {ei}) for {sk!r}")
            seen_s.add(sk)
            norm_spans.append((sk, si, ei))
        object.__setattr__(self, "byte_spans", tuple(sorted(norm_spans)))

        norm_types = tuple(sorted((str(k), str(v)) for k, v in self.role_types))
        object.__setattr__(self, "role_types", norm_types)
        conf = float(self.confidence)
        if not (0.0 <= conf <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {conf}")
        object.__setattr__(self, "confidence", round(conf, 6))
        object.__setattr__(self, "ambiguity_reasons", tuple(sorted({str(r) for r in self.ambiguity_reasons})))

    def role_dict(self) -> dict[str, str]:
        return dict(self.roles)

    def span_dict(self) -> dict[str, tuple[int, int]]:
        return {k: (s, e) for k, s, e in self.byte_spans}

    def role_type_dict(self) -> dict[str, str]:
        return dict(self.role_types)

    def validate_spans_against(self, raw_bytes: bytes) -> bool:
        """Verify every span is within bounds and decodes cleanly as valid UTF-8."""
        n = len(raw_bytes)
        for _key, s, e in self.byte_spans:
            if not (0 <= s <= e <= n):
                return False
            try:
                raw_bytes[s:e].decode("utf-8")
            except UnicodeDecodeError:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "roles": dict(self.roles),
            "byte_spans": {k: [s, e] for k, s, e in self.byte_spans},
            "role_types": dict(self.role_types),
            "confidence": self.confidence,
            "parse_rule": self.parse_rule,
            "ambiguity_reasons": list(self.ambiguity_reasons),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateActionFrame:
        return cls(
            action_id=str(data["action_id"]),
            roles=tuple((str(k), str(v)) for k, v in data.get("roles", {}).items()),
            byte_spans=tuple(
                (str(k), int(v[0]), int(v[1]))
                for k, v in data.get("byte_spans", {}).items()
            ),
            role_types=tuple((str(k), str(v)) for k, v in data.get("role_types", {}).items()),
            confidence=float(data.get("confidence", 1.0)),
            parse_rule=str(data.get("parse_rule", "typed_frame")),
            ambiguity_reasons=tuple(str(r) for r in data.get("ambiguity_reasons", ())),
        )


@dataclass(frozen=True)
class FrameParseResult:
    """Ambiguity-preserving parser output over raw UTF-8 instruction bytes."""

    raw_instruction_bytes: bytes
    parser_name: str
    parser_version: str
    candidate_frames: tuple[CandidateActionFrame, ...]
    is_ambiguous: bool
    abstained: bool
    abstention_reason: str = ""

    def __post_init__(self) -> None:
        raw = bytes(self.raw_instruction_bytes)
        object.__setattr__(self, "raw_instruction_bytes", raw)
        cands = tuple(self.candidate_frames)
        for frame in cands:
            if not frame.validate_spans_against(raw):
                raise ValueError(
                    f"CandidateActionFrame byte_spans {frame.byte_spans} invalid for raw bytes len={len(raw)}"
                )
        object.__setattr__(self, "candidate_frames", cands)
        if not cands and not self.abstained:
            raise ValueError("FrameParseResult with zero candidates must set abstained=True")
        if len(cands) > 1 and not self.is_ambiguous:
            object.__setattr__(self, "is_ambiguous", True)

    @property
    def top_frame(self) -> CandidateActionFrame | None:
        if not self.candidate_frames:
            return None
        return self.candidate_frames[0]

    @property
    def unambiguous_frame(self) -> CandidateActionFrame | None:
        """Return the single selected frame ONLY when parse is non-empty and unambiguous."""
        if self.abstained or self.is_ambiguous or len(self.candidate_frames) != 1:
            return None
        return self.candidate_frames[0]

    def to_dict(self) -> dict[str, Any]:
        try:
            text_preview = self.raw_instruction_bytes.decode("utf-8")
            valid_utf8 = True
        except UnicodeDecodeError:
            text_preview = self.raw_instruction_bytes.decode("utf-8", errors="replace")
            valid_utf8 = False
        return {
            "raw_instruction_utf8": text_preview,
            "raw_instruction_valid_utf8": valid_utf8,
            "raw_instruction_byte_length": len(self.raw_instruction_bytes),
            "raw_instruction_sha256": sha256_hex(self.raw_instruction_bytes),
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "candidate_frames": [c.to_dict() for c in self.candidate_frames],
            "is_ambiguous": self.is_ambiguous,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, raw_bytes: bytes | None = None) -> FrameParseResult:
        raw = raw_bytes if raw_bytes is not None else str(data["raw_instruction_utf8"]).encode("utf-8")
        cands = tuple(CandidateActionFrame.from_dict(c) for c in data.get("candidate_frames", ()))
        return cls(
            raw_instruction_bytes=raw,
            parser_name=str(data["parser_name"]),
            parser_version=str(data["parser_version"]),
            candidate_frames=cands,
            is_ambiguous=bool(data.get("is_ambiguous", len(cands) > 1)),
            abstained=bool(data.get("abstained", len(cands) == 0)),
            abstention_reason=str(data.get("abstention_reason", "")),
        )


def compute_canonical_state_action_key(
    before_state_fp: str,
    action_id: str,
    roles: Mapping[str, str],
) -> str:
    """Canonical SHA-256 exact-match key matching TrainingEpisode.state_action_key()."""
    payload = {
        "before_state_fp": str(before_state_fp),
        "action_id": str(action_id),
        "roles": dict(sorted((str(k), str(v)) for k, v in roles.items())),
    }
    return sha256_hex(canonical_json_bytes(payload))
