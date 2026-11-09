"""Universal, domain-neutral knowledge representation for Mirror 7.

This module defines data contracts only. It deliberately does not interpret tasks,
plan work, execute operations, compose capabilities, or learn from outcomes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .blueprint_contracts import Blueprint, BlueprintKind, VerificationEvidence


class RepresentationValidationError(ValueError):
    """Raised when a representation violates the universal schema."""


class RepresentationConflictError(RepresentationValidationError):
    """Raised when an existing identifier is reused for different content."""


class RepresentationKind(str, Enum):
    ENTITY = "entity"
    CONCEPT = "concept"
    FACT = "fact"
    RELATION = "relation"
    RULE = "rule"
    CONSTRAINT = "constraint"
    FORMULA = "formula"
    PROCEDURE = "procedure"
    OPERATION = "operation"
    STRUCTURE = "structure"
    OBSERVATION = "observation"
    MEASUREMENT = "measurement"
    EVIDENCE = "evidence"
    GOAL = "goal"
    ARTIFACT = "artifact"


class EpistemicStatus(str, Enum):
    UNKNOWN = "unknown"
    ASSUMED = "assumed"
    ASSERTED = "asserted"
    OBSERVED = "observed"
    DERIVED = "derived"
    DISPUTED = "disputed"


class UncertaintyKind(str, Enum):
    UNKNOWN = "unknown"
    EPISTEMIC = "epistemic"
    ALEATORIC = "aleatoric"
    MEASUREMENT = "measurement"


def _freeze_json(value: Any, path: str = "value") -> Any:
    """Validate and recursively freeze JSON-compatible content."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RepresentationValidationError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RepresentationValidationError(f"{path} mapping keys must be strings")
            frozen[key] = _freeze_json(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise RepresentationValidationError(
        f"{path} must contain only JSON-compatible values, got {type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Return deterministic, JSON-safe serialization."""
    try:
        return json.dumps(
            _thaw_json(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RepresentationValidationError(f"value is not canonical JSON: {exc}") from exc


def stable_representation_id(prefix: str, payload: Any) -> str:
    """Create a deterministic identifier from canonical content."""
    tag = str(prefix).strip().casefold()
    if not tag:
        raise RepresentationValidationError("identifier prefix must not be empty")
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{tag}:{digest[:24]}"


@dataclass(frozen=True)
class RepresentationProvenance:
    source: str = "unknown"
    source_ref: str = ""
    license_spdx: str = "NOASSERTION"
    content_sha256: str = ""
    captured_at: str = ""
    parent_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise RepresentationValidationError("provenance.source must not be empty")
        if self.content_sha256 and (
            len(self.content_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.content_sha256)
        ):
            raise RepresentationValidationError(
                "provenance.content_sha256 must be a lowercase SHA-256 digest"
            )
        parents = tuple(str(value).strip() for value in self.parent_ids)
        if any(not value for value in parents):
            raise RepresentationValidationError("provenance parent identifiers must not be empty")
        if len(set(parents)) != len(parents):
            raise RepresentationValidationError("provenance parent identifiers must be unique")
        object.__setattr__(self, "parent_ids", parents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_ref": self.source_ref,
            "license_spdx": self.license_spdx,
            "content_sha256": self.content_sha256,
            "captured_at": self.captured_at,
            "parent_ids": list(self.parent_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RepresentationProvenance":
        data = dict(value)
        data["parent_ids"] = tuple(data.get("parent_ids", ()))
        return cls(**data)


@dataclass(frozen=True)
class RepresentationUncertainty:
    kind: UncertaintyKind = UncertaintyKind.UNKNOWN
    confidence: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    unit: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", UncertaintyKind(self.kind))
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not math.isfinite(float(self.confidence)):
                raise RepresentationValidationError("uncertainty confidence must be finite")
            confidence = float(self.confidence)
            if not 0.0 <= confidence <= 1.0:
                raise RepresentationValidationError("uncertainty confidence must be between 0 and 1")
            object.__setattr__(self, "confidence", confidence)
        if (self.lower_bound is None) != (self.upper_bound is None):
            raise RepresentationValidationError("uncertainty interval requires both bounds")
        if self.lower_bound is not None:
            if any(
                isinstance(value, bool) or not math.isfinite(float(value))
                for value in (self.lower_bound, self.upper_bound)
            ):
                raise RepresentationValidationError("uncertainty bounds must be finite numbers")
            lower, upper = float(self.lower_bound), float(self.upper_bound)
            if lower > upper:
                raise RepresentationValidationError("uncertainty lower_bound exceeds upper_bound")
            object.__setattr__(self, "lower_bound", lower)
            object.__setattr__(self, "upper_bound", upper)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "confidence": self.confidence,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "unit": self.unit,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RepresentationUncertainty":
        return cls(**dict(value))


@dataclass(frozen=True)
class RepresentationNode:
    node_id: str
    kind: RepresentationKind
    domain: str
    label: str
    content: Any = None
    value_type: str = "any"
    unit: str = ""
    dimensions: Mapping[str, int] = field(default_factory=dict)
    status: EpistemicStatus = EpistemicStatus.UNKNOWN
    attributes: Mapping[str, Any] = field(default_factory=dict)
    provenance: RepresentationProvenance = field(default_factory=RepresentationProvenance)
    verification: VerificationEvidence = field(default_factory=VerificationEvidence)
    uncertainty: RepresentationUncertainty = field(default_factory=RepresentationUncertainty)
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", RepresentationKind(self.kind))
        object.__setattr__(self, "status", EpistemicStatus(self.status))
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise RepresentationValidationError("node_id must not be empty")
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise RepresentationValidationError("domain must not be empty")
        if not isinstance(self.label, str) or not self.label.strip():
            raise RepresentationValidationError("label must not be empty")
        if not isinstance(self.value_type, str) or not self.value_type.strip():
            raise RepresentationValidationError("value_type must not be empty")
        if self.schema_version != 1:
            raise RepresentationValidationError(
                f"unsupported representation schema version: {self.schema_version}"
            )
        if not isinstance(self.provenance, RepresentationProvenance):
            raise RepresentationValidationError("provenance must be RepresentationProvenance")
        if not isinstance(self.verification, VerificationEvidence):
            raise RepresentationValidationError("verification must be VerificationEvidence")
        if not isinstance(self.uncertainty, RepresentationUncertainty):
            raise RepresentationValidationError("uncertainty must be RepresentationUncertainty")
        if not isinstance(self.dimensions, Mapping):
            raise RepresentationValidationError("dimensions must be a mapping")
        if not isinstance(self.attributes, Mapping):
            raise RepresentationValidationError("attributes must be a mapping")

        normalized_dimensions: dict[str, int] = {}
        for name, exponent in self.dimensions.items():
            if not isinstance(name, str) or not name.strip():
                raise RepresentationValidationError("dimension names must be non-empty strings")
            if isinstance(exponent, bool) or not isinstance(exponent, int):
                raise RepresentationValidationError("dimension exponents must be integers")
            key = name.strip().casefold()
            if key in normalized_dimensions:
                raise RepresentationValidationError(f"duplicate normalized dimension: {key}")
            normalized_dimensions[key] = exponent

        object.__setattr__(self, "node_id", self.node_id.strip())
        object.__setattr__(self, "domain", self.domain.strip().casefold())
        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(self, "value_type", self.value_type.strip().casefold())
        object.__setattr__(self, "unit", str(self.unit).strip())
        object.__setattr__(self, "dimensions", _freeze_json(normalized_dimensions, "dimensions"))
        object.__setattr__(self, "content", _freeze_json(self.content, "content"))
        object.__setattr__(self, "attributes", _freeze_json(self.attributes, "attributes"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "domain": self.domain,
            "label": self.label,
            "content": _thaw_json(self.content),
            "value_type": self.value_type,
            "unit": self.unit,
            "dimensions": _thaw_json(self.dimensions),
            "status": self.status.value,
            "attributes": _thaw_json(self.attributes),
            "provenance": self.provenance.to_dict(),
            "verification": {
                **asdict(self.verification),
                "level": self.verification.level.value,
            },
            "uncertainty": self.uncertainty.to_dict(),
            "schema_version": self.schema_version,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RepresentationNode":
        data = dict(value)
        provenance = data.get("provenance", {})
        data["provenance"] = (
            provenance if isinstance(provenance, RepresentationProvenance)
            else RepresentationProvenance.from_dict(provenance)
        )
        verification = data.get("verification", {})
        data["verification"] = (
            verification if isinstance(verification, VerificationEvidence)
            else VerificationEvidence(**dict(verification))
        )
        uncertainty = data.get("uncertainty", {})
        data["uncertainty"] = (
            uncertainty if isinstance(uncertainty, RepresentationUncertainty)
            else RepresentationUncertainty.from_dict(uncertainty)
        )
        return cls(**data)

    @classmethod
    def from_blueprint(cls, blueprint: Blueprint) -> "RepresentationNode":
        """Adapt a blueprint without upgrading its epistemic or evidence status."""
        if not isinstance(blueprint, Blueprint):
            raise TypeError("blueprint must be a Blueprint")
        kind_map = {
            BlueprintKind.FACT: RepresentationKind.FACT,
            BlueprintKind.PROCEDURE: RepresentationKind.PROCEDURE,
            BlueprintKind.FORMULA: RepresentationKind.FORMULA,
            BlueprintKind.POLICY: RepresentationKind.RULE,
            BlueprintKind.HEURISTIC: RepresentationKind.RULE,
            BlueprintKind.CONVERSATION: RepresentationKind.ARTIFACT,
        }
        payload = {
            "prompt_pattern": blueprint.prompt_pattern,
            "response_template": blueprint.response_template,
            "slots": list(blueprint.slots),
            "derived_expr": blueprint.derived_expr,
        }
        attributes = {
            "intent": blueprint.intent,
            "preconditions": list(blueprint.preconditions),
            "postconditions": list(blueprint.postconditions),
            "dependencies": list(blueprint.dependencies),
            "legacy_trust_tier": blueprint.legacy_trust_tier,
        }
        source = blueprint.provenance
        return cls(
            node_id=f"blueprint:{blueprint.blueprint_id}",
            kind=kind_map[blueprint.kind],
            domain=blueprint.domain,
            label=blueprint.intent,
            content=payload,
            value_type="record",
            status=EpistemicStatus.ASSERTED,
            attributes=attributes,
            provenance=RepresentationProvenance(
                source=source.source,
                source_ref=source.source_ref,
                license_spdx=source.license_spdx,
                content_sha256=source.content_sha256,
                captured_at=source.captured_at,
                parent_ids=blueprint.dependencies,
            ),
            verification=blueprint.verification,
        )


@dataclass(frozen=True)
class RepresentationRelation:
    relation_id: str
    source_id: str
    predicate: str
    target_id: str
    domain: str = "general"
    directed: bool = True
    status: EpistemicStatus = EpistemicStatus.UNKNOWN
    attributes: Mapping[str, Any] = field(default_factory=dict)
    provenance: RepresentationProvenance = field(default_factory=RepresentationProvenance)
    verification: VerificationEvidence = field(default_factory=VerificationEvidence)
    uncertainty: RepresentationUncertainty = field(default_factory=RepresentationUncertainty)
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", EpistemicStatus(self.status))
        for name, value in (
            ("relation_id", self.relation_id),
            ("source_id", self.source_id),
            ("predicate", self.predicate),
            ("target_id", self.target_id),
            ("domain", self.domain),
        ):
            if not isinstance(value, str) or not value.strip():
                raise RepresentationValidationError(f"{name} must not be empty")
        if self.schema_version != 1:
            raise RepresentationValidationError(
                f"unsupported relation schema version: {self.schema_version}"
            )
        if not isinstance(self.directed, bool):
            raise RepresentationValidationError("directed must be a boolean")
        if not isinstance(self.provenance, RepresentationProvenance):
            raise RepresentationValidationError("provenance must be RepresentationProvenance")
        if not isinstance(self.verification, VerificationEvidence):
            raise RepresentationValidationError("verification must be VerificationEvidence")
        if not isinstance(self.uncertainty, RepresentationUncertainty):
            raise RepresentationValidationError("uncertainty must be RepresentationUncertainty")
        if not isinstance(self.attributes, Mapping):
            raise RepresentationValidationError("attributes must be a mapping")
        object.__setattr__(self, "relation_id", self.relation_id.strip())
        object.__setattr__(self, "source_id", self.source_id.strip())
        object.__setattr__(self, "target_id", self.target_id.strip())
        object.__setattr__(self, "predicate", self.predicate.strip().casefold())
        object.__setattr__(self, "domain", self.domain.strip().casefold())
        object.__setattr__(self, "attributes", _freeze_json(self.attributes, "attributes"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "source_id": self.source_id,
            "predicate": self.predicate,
            "target_id": self.target_id,
            "domain": self.domain,
            "directed": self.directed,
            "status": self.status.value,
            "attributes": _thaw_json(self.attributes),
            "provenance": self.provenance.to_dict(),
            "verification": {
                **asdict(self.verification),
                "level": self.verification.level.value,
            },
            "uncertainty": self.uncertainty.to_dict(),
            "schema_version": self.schema_version,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RepresentationRelation":
        data = dict(value)
        provenance = data.get("provenance", {})
        data["provenance"] = (
            provenance if isinstance(provenance, RepresentationProvenance)
            else RepresentationProvenance.from_dict(provenance)
        )
        verification = data.get("verification", {})
        data["verification"] = (
            verification if isinstance(verification, VerificationEvidence)
            else VerificationEvidence(**dict(verification))
        )
        uncertainty = data.get("uncertainty", {})
        data["uncertainty"] = (
            uncertainty if isinstance(uncertainty, RepresentationUncertainty)
            else RepresentationUncertainty.from_dict(uncertainty)
        )
        return cls(**data)


class RepresentationGraph:
    """Deterministic graph of domain-neutral nodes and typed relations."""

    def __init__(
        self,
        nodes: Iterable[RepresentationNode] = (),
        relations: Iterable[RepresentationRelation] = (),
    ) -> None:
        self._nodes: dict[str, RepresentationNode] = {}
        self._relations: dict[str, RepresentationRelation] = {}
        for node in nodes:
            self.add_node(node)
        for relation in relations:
            self.add_relation(relation)

    @property
    def nodes(self) -> tuple[RepresentationNode, ...]:
        return tuple(self._nodes[key] for key in sorted(self._nodes))

    @property
    def relations(self) -> tuple[RepresentationRelation, ...]:
        return tuple(self._relations[key] for key in sorted(self._relations))

    def add_node(self, node: RepresentationNode) -> RepresentationNode:
        if not isinstance(node, RepresentationNode):
            raise TypeError("node must be a RepresentationNode")
        previous = self._nodes.get(node.node_id)
        if previous is not None and previous != node:
            raise RepresentationConflictError(f"conflicting representation node: {node.node_id}")
        self._nodes[node.node_id] = node
        return node

    def add_relation(self, relation: RepresentationRelation) -> RepresentationRelation:
        if not isinstance(relation, RepresentationRelation):
            raise TypeError("relation must be a RepresentationRelation")
        if relation.source_id not in self._nodes or relation.target_id not in self._nodes:
            raise RepresentationValidationError(
                f"relation endpoints must exist in graph: {relation.source_id} -> {relation.target_id}"
            )
        previous = self._relations.get(relation.relation_id)
        if previous is not None and previous != relation:
            raise RepresentationConflictError(
                f"conflicting representation relation: {relation.relation_id}"
            )
        self._relations[relation.relation_id] = relation
        return relation

    def get_node(self, node_id: str) -> RepresentationNode | None:
        return self._nodes.get(str(node_id))

    def get_relation(self, relation_id: str) -> RepresentationRelation | None:
        return self._relations.get(str(relation_id))

    def connected(
        self,
        node_id: str,
        *,
        predicate: str | None = None,
        direction: str = "both",
    ) -> tuple[RepresentationRelation, ...]:
        if direction not in {"incoming", "outgoing", "both"}:
            raise ValueError("direction must be incoming, outgoing, or both")
        if node_id not in self._nodes:
            raise KeyError(f"unknown representation node: {node_id}")
        normalized = predicate.casefold().strip() if predicate else None
        matches = []
        for relation in self.relations:
            incident = (
                direction == "both" and (
                    relation.source_id == node_id or relation.target_id == node_id
                )
            ) or (direction == "outgoing" and relation.source_id == node_id) or (
                direction == "incoming" and relation.target_id == node_id
            )
            if incident and (normalized is None or relation.predicate == normalized):
                matches.append(relation)
        return tuple(matches)

    def validate(self) -> None:
        for relation in self._relations.values():
            if relation.source_id not in self._nodes or relation.target_id not in self._nodes:
                raise RepresentationValidationError(
                    f"orphan relation endpoints: {relation.relation_id}"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": 1,
            "nodes": [node.to_dict() for node in self.nodes],
            "relations": [relation.to_dict() for relation in self.relations],
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RepresentationGraph":
        if int(value.get("schema_version", 1)) != 1:
            raise RepresentationValidationError("unsupported graph schema version")
        graph = cls(nodes=(
            RepresentationNode.from_dict(item) for item in value.get("nodes", ())
        ))
        for item in value.get("relations", ()):
            graph.add_relation(RepresentationRelation.from_dict(item))
        return graph
