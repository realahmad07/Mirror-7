"""Domain-neutral representations for Mirror's compositional intelligence.

Knowledge is represented separately from proof. Tasks, capabilities, plans,
execution traces, and verification reports share stable typed contracts so new
domains can plug into the same reasoning engine without replacing its core.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import re
from typing import Any, Mapping


class RepresentationKind(str, Enum):
    ENTITY = "entity"
    FACT = "fact"
    RELATION = "relation"
    RULE = "rule"
    CONSTRAINT = "constraint"
    FORMULA = "formula"
    PROCEDURE = "procedure"
    CAPABILITY = "capability"
    OBSERVATION = "observation"
    EVIDENCE = "evidence"
    GOAL = "goal"
    ARTIFACT = "artifact"


class VerificationLevel(str, Enum):
    UNVERIFIED = "unverified"
    CURATED = "curated"
    EXECUTION_TESTED = "execution_tested"
    FORMALLY_VERIFIED = "formally_verified"
    INDEPENDENTLY_REPRODUCED = "independently_reproduced"


_TRUST_RANK = {
    VerificationLevel.UNVERIFIED: 0,
    VerificationLevel.CURATED: 1,
    VerificationLevel.EXECUTION_TESTED: 2,
    VerificationLevel.FORMALLY_VERIFIED: 3,
    VerificationLevel.INDEPENDENTLY_REPRODUCED: 4,
}


def canonical_json(value: Any) -> str:
    """Stable JSON representation used for hashes, ids, and persisted contracts."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_id(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def normalize_type_name(name: str) -> str:
    value = re.sub(r"[^a-z0-9_]+", "_", str(name).casefold()).strip("_")
    aliases = {
        "int": "integer",
        "float": "real",
        "number": "number",
        "numeric": "number",
        "str": "string",
        "bool": "boolean",
        "list": "sequence",
        "array": "sequence",
        "dict": "mapping",
        "object": "any",
        "text": "string",
    }
    return aliases.get(value, value or "any")


def type_compatible(actual: str, expected: str) -> bool:
    """Structural type compatibility for the shared value vocabulary."""
    actual_name = normalize_type_name(actual)
    expected_name = normalize_type_name(expected)
    if expected_name in {"any", "value"} or actual_name == expected_name:
        return True
    if expected_name == "number" and actual_name in {"integer", "real"}:
        return True
    if expected_name == "real" and actual_name == "integer":
        return True
    return False


def value_matches_type(value: Any, expected: str) -> bool:
    kind = normalize_type_name(expected)
    if kind in {"any", "value"}:
        return True
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind in {"real", "number"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    if kind in {"sequence", "list"}:
        return isinstance(value, (list, tuple))
    if kind in {"mapping", "record"}:
        return isinstance(value, Mapping)
    if kind == "bytes":
        return isinstance(value, bytes)
    return False


@dataclass(frozen=True)
class ProvenanceRecord:
    source: str = "unknown"
    source_ref: str = ""
    license_spdx: str = "NOASSERTION"
    content_sha256: str = ""
    captured_at: str = ""
    parent_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("provenance.source must not be empty")
        if self.content_sha256 and not re.fullmatch(r"[0-9a-f]{64}", self.content_sha256):
            raise ValueError("provenance.content_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "parent_ids", tuple(str(x) for x in self.parent_ids))


@dataclass(frozen=True)
class EvidenceRecord:
    level: VerificationLevel = VerificationLevel.UNVERIFIED
    verifier: str = ""
    checks_passed: tuple[str, ...] = ()
    checks_failed: tuple[str, ...] = ()
    held_out_passed: bool = False
    adversarial_passed: bool = False
    seed_count: int = 0
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", VerificationLevel(self.level))
        object.__setattr__(self, "checks_passed", tuple(str(x) for x in self.checks_passed))
        object.__setattr__(self, "checks_failed", tuple(str(x) for x in self.checks_failed))
        if self.seed_count < 0:
            raise ValueError("seed_count cannot be negative")
        if self.checks_failed and self.level != VerificationLevel.UNVERIFIED:
            raise ValueError("failed checks cannot carry a verified evidence level")
        if self.level in {
            VerificationLevel.EXECUTION_TESTED,
            VerificationLevel.FORMALLY_VERIFIED,
            VerificationLevel.INDEPENDENTLY_REPRODUCED,
        } and not self.checks_passed:
            raise ValueError(f"{self.level.value} evidence requires named passing checks")
        if self.level in {
            VerificationLevel.FORMALLY_VERIFIED,
            VerificationLevel.INDEPENDENTLY_REPRODUCED,
        } and not self.verifier.strip():
            raise ValueError(f"{self.level.value} evidence requires a named verifier")

    @property
    def trusted_for_execution(self) -> bool:
        return (
            not self.checks_failed
            and _TRUST_RANK[self.level] >= _TRUST_RANK[VerificationLevel.EXECUTION_TESTED]
        )

    @property
    def independently_reproduced(self) -> bool:
        return (
            self.level == VerificationLevel.INDEPENDENTLY_REPRODUCED
            and self.held_out_passed
            and self.adversarial_passed
            and self.seed_count >= 3
            and not self.checks_failed
        )


@dataclass(frozen=True)
class RepresentationNode:
    node_id: str
    kind: RepresentationKind
    domain: str
    label: str
    content: Any
    attributes: Mapping[str, Any] = field(default_factory=dict)
    relations: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    provenance: ProvenanceRecord = field(default_factory=ProvenanceRecord)
    evidence: EvidenceRecord = field(default_factory=EvidenceRecord)
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", RepresentationKind(self.kind))
        object.__setattr__(self, "relations", tuple(str(x) for x in self.relations))
        object.__setattr__(self, "dependencies", tuple(str(x) for x in self.dependencies))
        if not self.node_id.strip() or not self.domain.strip() or not self.label.strip():
            raise ValueError("node_id, domain, and label must be non-empty")
        if self.schema_version != 1:
            raise ValueError("unsupported representation schema version")
        if not isinstance(self.provenance, ProvenanceRecord):
            raise TypeError("provenance must be ProvenanceRecord")
        if not isinstance(self.evidence, EvidenceRecord):
            raise TypeError("evidence must be EvidenceRecord")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "domain": self.domain,
            "label": self.label,
            "content": self.content,
            "attributes": dict(self.attributes),
            "relations": list(self.relations),
            "dependencies": list(self.dependencies),
            "provenance": asdict(self.provenance),
            "evidence": {
                **asdict(self.evidence),
                "level": self.evidence.level.value,
            },
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class SubtaskSpec:
    subtask_id: str
    goal: str
    domain: str = "general"
    capability_tags: tuple[str, ...] = ()
    capability_id: str = ""
    input_bindings: Mapping[str, str] = field(default_factory=dict)
    output_schema: Mapping[str, str] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    acceptance_criteria: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "capability_tags", tuple(str(x) for x in self.capability_tags))
        object.__setattr__(self, "dependencies", tuple(str(x) for x in self.dependencies))
        object.__setattr__(
            self,
            "acceptance_criteria",
            tuple(dict(x) for x in self.acceptance_criteria),
        )
        if not self.subtask_id.strip() or not self.goal.strip():
            raise ValueError("subtask_id and goal must be non-empty")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    request: str
    goal: str
    domain: str = "general"
    inputs: Mapping[str, Any] = field(default_factory=dict)
    input_schema: Mapping[str, str] = field(default_factory=dict)
    required_outputs: Mapping[str, str] = field(default_factory=dict)
    required_capability_tags: tuple[str, ...] = ()
    subtasks: tuple[SubtaskSpec, ...] = ()
    constraints: tuple[Mapping[str, Any], ...] = ()
    acceptance_criteria: tuple[Mapping[str, Any], ...] = ()
    context: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", self.domain.strip().casefold() or "general")
        object.__setattr__(self, "required_capability_tags", tuple(str(x) for x in self.required_capability_tags))
        object.__setattr__(self, "subtasks", tuple(self.subtasks))
        object.__setattr__(self, "constraints", tuple(dict(x) for x in self.constraints))
        object.__setattr__(self, "acceptance_criteria", tuple(dict(x) for x in self.acceptance_criteria))
        if not self.request.strip() or not self.goal.strip() or not self.task_id.strip():
            raise ValueError("task_id, request, and goal must be non-empty")
        if self.schema_version != 1:
            raise ValueError("unsupported task schema version")
        if len({x.subtask_id for x in self.subtasks}) != len(self.subtasks):
            raise ValueError("subtask ids must be unique")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskSpec":
        payload = dict(data)
        payload["subtasks"] = tuple(
            item if isinstance(item, SubtaskSpec) else SubtaskSpec(**item)
            for item in payload.get("subtasks", ())
        )
        for key in ("required_capability_tags", "constraints", "acceptance_criteria"):
            if key in payload:
                payload[key] = tuple(payload[key])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "request": self.request,
            "goal": self.goal,
            "domain": self.domain,
            "inputs": dict(self.inputs),
            "input_schema": dict(self.input_schema),
            "required_outputs": dict(self.required_outputs),
            "required_capability_tags": list(self.required_capability_tags),
            "subtasks": [
                {
                    **asdict(item),
                    "capability_tags": list(item.capability_tags),
                    "dependencies": list(item.dependencies),
                    "acceptance_criteria": list(item.acceptance_criteria),
                }
                for item in self.subtasks
            ],
            "constraints": list(self.constraints),
            "acceptance_criteria": list(self.acceptance_criteria),
            "context": dict(self.context),
            "schema_version": self.schema_version,
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CapabilitySpec:
    capability_id: str
    name: str
    domains: tuple[str, ...]
    input_schema: Mapping[str, str]
    output_schema: Mapping[str, str]
    tags: tuple[str, ...] = ()
    preconditions: tuple[Mapping[str, Any], ...] = ()
    postconditions: tuple[Mapping[str, Any], ...] = ()
    dependencies: tuple[str, ...] = ()
    executor_key: str = ""
    evidence: EvidenceRecord = field(default_factory=EvidenceRecord)
    provenance: ProvenanceRecord = field(default_factory=ProvenanceRecord)
    transferable: bool = False
    lineage: tuple[str, ...] = ()
    implementation: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "domains", tuple(sorted({str(x).casefold() for x in self.domains})))
        object.__setattr__(self, "tags", tuple(sorted({str(x).casefold() for x in self.tags})))
        object.__setattr__(self, "preconditions", tuple(dict(x) for x in self.preconditions))
        object.__setattr__(self, "postconditions", tuple(dict(x) for x in self.postconditions))
        object.__setattr__(self, "dependencies", tuple(str(x) for x in self.dependencies))
        object.__setattr__(self, "lineage", tuple(str(x) for x in self.lineage))
        object.__setattr__(self, "implementation", dict(self.implementation))
        if not self.capability_id.strip() or not self.name.strip() or not self.domains:
            raise ValueError("capability_id, name, and domains must be non-empty")
        if self.schema_version != 1:
            raise ValueError("unsupported capability schema version")
        if not isinstance(self.evidence, EvidenceRecord):
            raise TypeError("evidence must be EvidenceRecord")

    @property
    def trusted(self) -> bool:
        return self.evidence.trusted_for_execution

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = {
            **asdict(self.evidence),
            "level": self.evidence.level.value,
        }
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapabilitySpec":
        payload = dict(data)
        evidence = payload.get("evidence", {})
        payload["evidence"] = (
            evidence if isinstance(evidence, EvidenceRecord)
            else EvidenceRecord(**evidence)
        )
        provenance = payload.get("provenance", {})
        payload["provenance"] = (
            provenance if isinstance(provenance, ProvenanceRecord)
            else ProvenanceRecord(**provenance)
        )
        for key in ("domains", "tags", "preconditions", "postconditions", "dependencies", "lineage"):
            if key in payload:
                payload[key] = tuple(payload[key])
        return cls(**payload)


@dataclass(frozen=True)
class PlannedStep:
    step_id: str
    subtask: SubtaskSpec
    capability_id: str
    input_bindings: Mapping[str, str]


@dataclass(frozen=True)
class TaskPlan:
    plan_id: str
    task_id: str
    task_fingerprint: str
    steps: tuple[PlannedStep, ...]
    unresolved: tuple[str, ...] = ()
    planner_version: str = "universal-planner-v1"

    @property
    def complete(self) -> bool:
        return bool(self.steps) and not self.unresolved


@dataclass(frozen=True)
class StepExecution:
    step_id: str
    capability_id: str
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Any]
    succeeded: bool
    error: str = ""


@dataclass(frozen=True)
class VerificationReport:
    passed: bool
    level: VerificationLevel
    checks_passed: tuple[str, ...] = ()
    checks_failed: tuple[str, ...] = ()
    verifier: str = ""
    verified_outputs: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LearningGateReport:
    cases_generated: int
    cases_passed: int
    seed_count: int
    held_out_passed: bool
    adversarial_passed: bool
    independent_verifier: str
    failures: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return (
            self.cases_generated >= 3
            and self.cases_passed == self.cases_generated
            and self.seed_count >= 3
            and self.held_out_passed
            and self.adversarial_passed
            and bool(self.independent_verifier.strip())
            and not self.failures
        )
