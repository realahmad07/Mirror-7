"""Canonical, versioned blueprint contracts for rules, procedures, and transformations.

Legacy records are adapted as unverified. A legacy trust label or a test string is
not silently promoted into proof of correctness.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import re
from typing import Any, Iterable

class BlueprintValidationError(ValueError):
    """Raised when a blueprint violates the canonical contract."""

class BlueprintKind(str, Enum):
    FACT = "fact"
    PROCEDURE = "procedure"
    FORMULA = "formula"
    POLICY = "policy"
    CONVERSATION = "conversation"
    HEURISTIC = "heuristic"

class EvidenceLevel(str, Enum):
    UNVERIFIED = "unverified"
    CURATED = "curated"
    EXECUTION_TESTED = "execution_tested"
    FORMALLY_VERIFIED = "formally_verified"
    INDEPENDENTLY_REPRODUCED = "independently_reproduced"

@dataclass(frozen=True)
class BlueprintProvenance:
    source: str = "unknown"
    source_ref: str = ""
    license_spdx: str = "NOASSERTION"
    content_sha256: str = ""
    captured_at: str = ""

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise BlueprintValidationError("provenance.source must not be empty")
        if self.content_sha256 and not re.fullmatch(r"[0-9a-f]{64}", self.content_sha256):
            raise BlueprintValidationError("provenance.content_sha256 must be a lowercase SHA-256 digest")

@dataclass(frozen=True)
class VerificationEvidence:
    level: EvidenceLevel = EvidenceLevel.UNVERIFIED
    checks_passed: tuple[str, ...] = ()
    checks_failed: tuple[str, ...] = ()
    verifier: str = ""
    verified_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", EvidenceLevel(self.level))
        object.__setattr__(self, "checks_passed", tuple(str(x) for x in self.checks_passed))
        object.__setattr__(self, "checks_failed", tuple(str(x) for x in self.checks_failed))
        if self.level in {EvidenceLevel.EXECUTION_TESTED, EvidenceLevel.FORMALLY_VERIFIED,
                          EvidenceLevel.INDEPENDENTLY_REPRODUCED} and not self.checks_passed:
            raise BlueprintValidationError(f"{self.level.value} evidence requires at least one named passing check")
        if self.level in {EvidenceLevel.FORMALLY_VERIFIED, EvidenceLevel.INDEPENDENTLY_REPRODUCED} and not self.verifier.strip():
            raise BlueprintValidationError(f"{self.level.value} evidence requires a named verifier")

_SLOT_RE = re.compile(r"\{(v\d+)\}")

@dataclass(frozen=True)
class Blueprint:
    blueprint_id: str
    kind: BlueprintKind
    domain: str
    intent: str
    prompt_pattern: str
    response_template: str
    slots: tuple[str, ...] = ()
    derived_expr: str = ""
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    provenance: BlueprintProvenance = field(default_factory=BlueprintProvenance)
    verification: VerificationEvidence = field(default_factory=VerificationEvidence)
    legacy_trust_tier: int = 2
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", BlueprintKind(self.kind))
        for name in ("slots", "preconditions", "postconditions", "dependencies"):
            object.__setattr__(self, name, tuple(str(x) for x in getattr(self, name)))
        if self.schema_version != 1:
            raise BlueprintValidationError(f"unsupported blueprint schema_version: {self.schema_version}")
        for name in ("blueprint_id", "domain", "intent", "prompt_pattern", "response_template"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise BlueprintValidationError(f"{name} must not be empty")
        if not 1 <= int(self.legacy_trust_tier) <= 3:
            raise BlueprintValidationError("legacy_trust_tier must be in the range 1..3")
        if self.blueprint_id in self.dependencies:
            raise BlueprintValidationError("a blueprint cannot depend on itself")
        if len(set(self.dependencies)) != len(self.dependencies) or len(set(self.slots)) != len(self.slots):
            raise BlueprintValidationError("dependencies and slots must be unique")
        if not isinstance(self.provenance, BlueprintProvenance):
            raise BlueprintValidationError("provenance must be a BlueprintProvenance")
        if not isinstance(self.verification, VerificationEvidence):
            raise BlueprintValidationError("verification must be VerificationEvidence")

    @staticmethod
    def _digest(prompt: str, response: str) -> str:
        return hashlib.sha256((prompt + "\0" + response).encode("utf-8")).hexdigest()

    @classmethod
    def from_crystal_rule(cls, rule: Any) -> "Blueprint":
        """Adapt a persisted CrystalRule without upgrading its evidence level."""
        expression = str(getattr(rule, "derived_expr", "") or "")
        tool = str(getattr(rule, "abstract_tool", "") or "")
        prompt = str(getattr(rule, "abstract_prompt", "") or "")
        response = str(getattr(rule, "abstract_response", "") or "")
        slots = tuple(sorted(set(_SLOT_RE.findall(prompt)), key=lambda value: int(value[1:])))
        kind = BlueprintKind.FORMULA if expression else (BlueprintKind.PROCEDURE if tool else BlueprintKind.FACT)
        return cls(
            blueprint_id=f"crystal:{int(rule.rule_id)}", kind=kind,
            domain=str(getattr(rule, "domain", "general") or "general"),
            intent="legacy_crystal_rule", prompt_pattern=prompt, response_template=response,
            slots=slots, derived_expr=expression,
            provenance=BlueprintProvenance(source="legacy_crystal_rule",
                source_ref=f"crystal://rule/{int(rule.rule_id)}", content_sha256=cls._digest(prompt, response)),
            verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
            legacy_trust_tier=int(getattr(rule, "trust_tier", 2)),
        )

    @classmethod
    def from_roadmap_record(cls, record: Any) -> "Blueprint":
        """Adapt a roadmap record; test_code is not evidence until actually executed."""
        prompt, response = str(getattr(record, "prompt", "") or ""), str(getattr(record, "response", "") or "")
        stage = str(getattr(record, "stage", "legacy_roadmap") or "legacy_roadmap")
        domain = str(getattr(record, "domain", "general") or "general")
        tool = str(getattr(record, "tool", "") or "")
        spdx = str(getattr(record, "spdx_license", "NOASSERTION") or "NOASSERTION")
        slots = tuple(sorted(set(_SLOT_RE.findall(prompt)), key=lambda value: int(value[1:])))
        kind = BlueprintKind.PROCEDURE if tool or str(getattr(record, "test_code", "")).strip() else BlueprintKind.FACT
        return cls(
            blueprint_id=f"roadmap:{stage}:{cls._digest(prompt, response)[:16]}", kind=kind,
            domain=domain, intent=stage.lower(), prompt_pattern=prompt, response_template=response, slots=slots,
            provenance=BlueprintProvenance(source="roadmap_blueprint_record", source_ref=stage,
                license_spdx=spdx, content_sha256=cls._digest(prompt, response)),
            verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
            legacy_trust_tier=int(getattr(record, "trust_tier", 2)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["provenance"] = asdict(self.provenance)
        data["verification"] = {**asdict(self.verification), "level": self.verification.level.value}
        return data

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Blueprint":
        payload = dict(data)
        payload["provenance"] = BlueprintProvenance(**payload.get("provenance", {}))
        payload["verification"] = VerificationEvidence(**payload.get("verification", {}))
        return cls(**payload)

class BlueprintDependencyGraph:
    """Resolve blueprint prerequisites in deterministic dependency-first order."""
    def __init__(self, blueprints: Iterable[Blueprint] = ()) -> None:
        self._blueprints: dict[str, Blueprint] = {}
        for blueprint in blueprints:
            self.add(blueprint)

    def add(self, blueprint: Blueprint) -> None:
        if not isinstance(blueprint, Blueprint):
            raise TypeError("blueprint must be a Blueprint")
        if blueprint.blueprint_id in self._blueprints:
            raise BlueprintValidationError(f"duplicate blueprint id: {blueprint.blueprint_id}")
        self._blueprints[blueprint.blueprint_id] = blueprint

    def resolve(self, blueprint_id: str) -> tuple[Blueprint, ...]:
        if blueprint_id not in self._blueprints:
            raise BlueprintValidationError(f"unknown blueprint: {blueprint_id}")
        ordered: list[Blueprint] = []
        visiting: list[str] = []
        visited: set[str] = set()
        def visit(current_id: str) -> None:
            if current_id in visited:
                return
            if current_id in visiting:
                cycle = " -> ".join(visiting[visiting.index(current_id):] + [current_id])
                raise BlueprintValidationError(f"dependency cycle: {cycle}")
            blueprint = self._blueprints.get(current_id)
            if blueprint is None:
                raise BlueprintValidationError(f"missing dependency: {current_id}")
            visiting.append(current_id)
            for dependency_id in sorted(blueprint.dependencies):
                visit(dependency_id)
            visiting.pop()
            visited.add(current_id)
            ordered.append(blueprint)
        visit(blueprint_id)
        return tuple(ordered)
