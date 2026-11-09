"""Universal capability contracts and deterministic registry (Phase 4).

Capabilities describe reusable operations independently of any one subject domain.
Registration is declarative: it never executes a capability or upgrades its evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Iterable, Mapping

from .blueprint_contracts import Blueprint, EvidenceLevel, VerificationEvidence
from .task_understanding import TaskIntent
from .universal_representation import (
    RepresentationProvenance,
    _freeze_json,
    _thaw_json,
    canonical_json,
)


class CapabilityValidationError(ValueError):
    """Raised when a capability or port contract is malformed."""


class CapabilityConflictError(CapabilityValidationError):
    """Raised when a capability identifier is reused for different content."""


def _unique_strings(values: Iterable[Any], field_name: str) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values)
    if any(not item for item in result):
        raise CapabilityValidationError(f"{field_name} cannot contain empty values")
    if len(set(result)) != len(result):
        raise CapabilityValidationError(f"{field_name} values must be unique")
    return result


@dataclass(frozen=True)
class PortContract:
    """Typed input or output boundary for a reusable capability."""
    name: str
    value_type: str = "any"
    required: bool = True
    unit: str = ""
    dimensions: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise CapabilityValidationError("port name must not be empty")
        if not isinstance(self.value_type, str) or not self.value_type.strip():
            raise CapabilityValidationError("port value_type must not be empty")
        if not isinstance(self.required, bool):
            raise CapabilityValidationError("port required must be a boolean")
        if not isinstance(self.dimensions, Mapping):
            raise CapabilityValidationError("port dimensions must be a mapping")
        normalized: dict[str, int] = {}
        for key, exponent in self.dimensions.items():
            name = str(key).strip().casefold()
            if not name or isinstance(exponent, bool) or not isinstance(exponent, int):
                raise CapabilityValidationError("dimension names and integer exponents are required")
            if name in normalized:
                raise CapabilityValidationError(f"duplicate normalized dimension: {name}")
            normalized[name] = exponent
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "value_type", self.value_type.strip().casefold())
        object.__setattr__(self, "unit", str(self.unit).strip())
        object.__setattr__(self, "dimensions", _freeze_json(normalized, "port.dimensions"))

    def accepts_value(self, value: Any) -> bool:
        """Check runtime values without Python's bool-is-int ambiguity."""
        kind = self.value_type
        if kind in {"any", "json"}:
            return True
        if kind in {"string", "text"}:
            return isinstance(value, str)
        if kind in {"boolean", "bool"}:
            return isinstance(value, bool)
        if kind in {"integer", "int"}:
            return isinstance(value, int) and not isinstance(value, bool)
        if kind in {"number", "numeric"}:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if kind in {"float", "decimal"}:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if kind in {"object", "record", "mapping"}:
            return isinstance(value, Mapping)
        if kind in {"array", "list", "sequence"}:
            return isinstance(value, (list, tuple))
        return False

    def accepts_output(self, producer: "PortContract") -> bool:
        """Return whether a producer port can safely feed this consumer port."""
        if not isinstance(producer, PortContract):
            raise TypeError("producer must be a PortContract")
        if self.name != producer.name:
            return False
        if self.unit and self.unit != producer.unit:
            return False
        if self.dimensions and dict(self.dimensions) != dict(producer.dimensions):
            return False
        consumer_type, producer_type = self.value_type, producer.value_type
        if consumer_type in {"any", "json"}:
            return True
        if producer_type in {"any", "json"}:
            return False
        compatible = {
            "number": {"number", "numeric", "float", "decimal", "integer", "int"},
            "numeric": {"number", "numeric", "float", "decimal", "integer", "int"},
            "float": {"number", "numeric", "float", "decimal", "integer", "int"},
            "decimal": {"number", "numeric", "float", "decimal", "integer", "int"},
            "integer": {"integer", "int"},
            "int": {"integer", "int"},
            "string": {"string", "text"},
            "text": {"string", "text"},
            "boolean": {"boolean", "bool"},
            "bool": {"boolean", "bool"},
            "object": {"object", "record", "mapping"},
            "record": {"object", "record", "mapping"},
            "mapping": {"object", "record", "mapping"},
            "array": {"array", "list", "sequence"},
            "list": {"array", "list", "sequence"},
            "sequence": {"array", "list", "sequence"},
        }
        return producer_type in compatible.get(consumer_type, {consumer_type})

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value_type": self.value_type,
            "required": self.required,
            "unit": self.unit,
            "dimensions": _thaw_json(self.dimensions),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PortContract":
        return cls(**dict(value))


_EXECUTABLE_EVIDENCE = {
    EvidenceLevel.EXECUTION_TESTED,
    EvidenceLevel.FORMALLY_VERIFIED,
    EvidenceLevel.INDEPENDENTLY_REPRODUCED,
}


@dataclass(frozen=True)
class CapabilitySpec:
    """A reusable operation with explicit typed boundaries and evidence."""
    capability_id: str
    name: str
    description: str
    domains: tuple[str, ...]
    intents: tuple[str, ...]
    inputs: tuple[PortContract, ...]
    outputs: tuple[PortContract, ...]
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    provenance: RepresentationProvenance = field(default_factory=RepresentationProvenance)
    verification: VerificationEvidence = field(default_factory=VerificationEvidence)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.capability_id, str) or not self.capability_id.strip():
            raise CapabilityValidationError("capability_id must not be empty")
        if not isinstance(self.name, str) or not self.name.strip():
            raise CapabilityValidationError("capability name must not be empty")
        if not isinstance(self.description, str) or not self.description.strip():
            raise CapabilityValidationError("capability description must not be empty")
        if self.schema_version != 1:
            raise CapabilityValidationError(f"unsupported capability schema: {self.schema_version}")
        domains = tuple(sorted(set(x.casefold() for x in _unique_strings(self.domains, "domains"))))
        intents = tuple(sorted(set(x.casefold() for x in _unique_strings(self.intents, "intents"))))
        if not domains:
            raise CapabilityValidationError("at least one supported domain is required")
        if not intents:
            raise CapabilityValidationError("at least one supported intent is required")
        inputs = tuple(self.inputs)
        outputs = tuple(self.outputs)
        if any(not isinstance(item, PortContract) for item in inputs + outputs):
            raise CapabilityValidationError("inputs and outputs must contain PortContract values")
        for label, ports in (("input", inputs), ("output", outputs)):
            names = [port.name for port in ports]
            if len(set(names)) != len(names):
                raise CapabilityValidationError(f"{label} port names must be unique")
        if not isinstance(self.provenance, RepresentationProvenance):
            raise CapabilityValidationError("provenance must be RepresentationProvenance")
        if not isinstance(self.verification, VerificationEvidence):
            raise CapabilityValidationError("verification must be VerificationEvidence")
        object.__setattr__(self, "capability_id", self.capability_id.strip())
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "domains", domains)
        object.__setattr__(self, "intents", intents)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "preconditions", _unique_strings(self.preconditions, "preconditions"))
        object.__setattr__(self, "postconditions", _unique_strings(self.postconditions, "postconditions"))
        object.__setattr__(self, "dependencies", _unique_strings(self.dependencies, "dependencies"))
        if self.capability_id in self.dependencies:
            raise CapabilityValidationError("a capability cannot depend on itself")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def executable(self) -> bool:
        """Only operation-level test/formal/reproduction evidence opens execution eligibility."""
        evidence = self.verification
        return (
            evidence.level in _EXECUTABLE_EVIDENCE
            and bool(evidence.checks_passed)
            and not evidence.checks_failed
        )

    @property
    def evidence_blocker(self) -> str:
        if self.verification.checks_failed:
            return "capability has failed verification checks"
        if self.verification.level not in _EXECUTABLE_EVIDENCE:
            return f"evidence level {self.verification.level.value} is not execution-eligible"
        if not self.verification.checks_passed:
            return "capability has no named passing verification checks"
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "name": self.name,
            "description": self.description,
            "domains": list(self.domains),
            "intents": list(self.intents),
            "inputs": [item.to_dict() for item in self.inputs],
            "outputs": [item.to_dict() for item in self.outputs],
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "dependencies": list(self.dependencies),
            "provenance": self.provenance.to_dict(),
            "verification": {
                "level": self.verification.level.value,
                "checks_passed": list(self.verification.checks_passed),
                "checks_failed": list(self.verification.checks_failed),
                "verifier": self.verification.verifier,
                "verified_at": self.verification.verified_at,
            },
            "schema_version": self.schema_version,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CapabilitySpec":
        data = dict(value)
        data["inputs"] = tuple(
            item if isinstance(item, PortContract) else PortContract.from_dict(item)
            for item in data.get("inputs", ())
        )
        data["outputs"] = tuple(
            item if isinstance(item, PortContract) else PortContract.from_dict(item)
            for item in data.get("outputs", ())
        )
        for name in ("domains", "intents", "preconditions", "postconditions", "dependencies"):
            data[name] = tuple(data.get(name, ()))
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
        return cls(**data)

    @classmethod
    def from_blueprint(cls, blueprint: Blueprint) -> "CapabilitySpec":
        """Adapt a blueprint conservatively; a legacy blueprint remains unverified."""
        if not isinstance(blueprint, Blueprint):
            raise TypeError("blueprint must be a Blueprint")
        return cls(
            capability_id=f"blueprint:{blueprint.blueprint_id}",
            name=blueprint.intent,
            description=f"Blueprint adapter for {blueprint.kind.value}: {blueprint.intent}",
            domains=(blueprint.domain,),
            intents=(blueprint.intent,),
            inputs=tuple(PortContract(name=slot, value_type="any") for slot in blueprint.slots),
            outputs=(PortContract(name="result", value_type="any"),),
            preconditions=blueprint.preconditions,
            postconditions=blueprint.postconditions,
            dependencies=tuple(f"blueprint:{item}" for item in blueprint.dependencies),
            provenance=RepresentationProvenance(
                source=blueprint.provenance.source,
                source_ref=blueprint.provenance.source_ref,
                license_spdx=blueprint.provenance.license_spdx,
                content_sha256=blueprint.provenance.content_sha256,
                captured_at=blueprint.provenance.captured_at,
                parent_ids=blueprint.dependencies,
            ),
            verification=blueprint.verification,
        )


class CapabilityRegistry:
    """In-memory deterministic registry; persistence is an explicit vault operation."""
    def __init__(self, capabilities: Iterable[CapabilitySpec] = ()) -> None:
        self._items: dict[str, CapabilitySpec] = {}
        for capability in capabilities:
            self.add(capability)

    def add(self, capability: CapabilitySpec) -> str:
        if not isinstance(capability, CapabilitySpec):
            raise TypeError("capability must be a CapabilitySpec")
        previous = self._items.get(capability.capability_id)
        if previous is not None and previous.fingerprint != capability.fingerprint:
            raise CapabilityConflictError(
                f"capability identifier has conflicting content: {capability.capability_id}"
            )
        self._items[capability.capability_id] = capability
        return capability.capability_id

    def get(self, capability_id: str) -> CapabilitySpec | None:
        return self._items.get(str(capability_id))

    def list(self, *, domain: str | None = None) -> tuple[CapabilitySpec, ...]:
        values = self._items.values()
        if domain is not None:
            key = str(domain).strip().casefold()
            values = (item for item in values if key in item.domains or "general" in item.domains or "*" in item.domains)
        return tuple(sorted(values, key=lambda item: item.capability_id))

    def __len__(self) -> int:
        return len(self._items)
