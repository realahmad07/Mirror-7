"""Universal knowledge foundation: explicit epistemic state, provenance and conflicts.

Phase 1 provides a single domain-neutral knowledge contract in front of the
representation graph. It preserves source/evidence boundaries and never treats
retrieval, assumptions, or legacy trust labels as proof.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import re
from typing import Any, Iterable, Mapping

from .blueprint_contracts import Blueprint, EvidenceLevel, VerificationEvidence
from .universal_representation import (
    EpistemicStatus,
    RepresentationKind,
    RepresentationNode,
    RepresentationProvenance,
    _freeze_json,
    _thaw_json,
    canonical_json,
    stable_representation_id,
)


class KnowledgeValidationError(ValueError):
    """Raised when a knowledge record violates the universal contract."""


class KnowledgeConflictError(KnowledgeValidationError):
    """Raised when a knowledge identifier is reused for different content."""


class KnowledgeKind(str, Enum):
    FACT = "fact"
    DEFINITION = "definition"
    RULE = "rule"
    FORMULA = "formula"
    PROCEDURE = "procedure"
    CONSTRAINT = "constraint"
    OBSERVATION = "observation"
    MODEL = "model"
    ASSUMPTION = "assumption"
    HEURISTIC = "heuristic"


_TRUSTED_FACT_EVIDENCE = {
    EvidenceLevel.FORMALLY_VERIFIED,
    EvidenceLevel.INDEPENDENTLY_REPRODUCED,
}
_OPERATION_EVIDENCE = {
    EvidenceLevel.EXECUTION_TESTED,
    EvidenceLevel.FORMALLY_VERIFIED,
    EvidenceLevel.INDEPENDENTLY_REPRODUCED,
}
_KIND_TO_REPRESENTATION = {
    KnowledgeKind.FACT: RepresentationKind.FACT,
    KnowledgeKind.DEFINITION: RepresentationKind.CONCEPT,
    KnowledgeKind.RULE: RepresentationKind.RULE,
    KnowledgeKind.FORMULA: RepresentationKind.FORMULA,
    KnowledgeKind.PROCEDURE: RepresentationKind.PROCEDURE,
    KnowledgeKind.CONSTRAINT: RepresentationKind.CONSTRAINT,
    KnowledgeKind.OBSERVATION: RepresentationKind.OBSERVATION,
    KnowledgeKind.MODEL: RepresentationKind.STRUCTURE,
    KnowledgeKind.ASSUMPTION: RepresentationKind.FACT,
    KnowledgeKind.HEURISTIC: RepresentationKind.RULE,
}


def _normalize(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[\w]+", str(value).casefold(), flags=re.UNICODE))


def _unique(values: Iterable[Any], field_name: str) -> tuple[str, ...]:
    result = tuple(str(item).strip() for item in values)
    if any(not item for item in result):
        raise KnowledgeValidationError(f"{field_name} cannot contain empty values")
    if len(set(result)) != len(result):
        raise KnowledgeValidationError(f"{field_name} values must be unique")
    return result


@dataclass(frozen=True)
class KnowledgeRecord:
    """One normalized claim or reusable knowledge item, independent of domain."""

    knowledge_id: str
    kind: KnowledgeKind
    domain: str
    subject: str
    predicate: str
    object_value: Any
    statement: str
    epistemic_status: EpistemicStatus = EpistemicStatus.ASSERTED
    content: Mapping[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    provenance: RepresentationProvenance = field(default_factory=RepresentationProvenance)
    verification: VerificationEvidence = field(default_factory=VerificationEvidence)
    capability_contract: Mapping[str, Any] | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "kind", KnowledgeKind(self.kind))
            object.__setattr__(self, "epistemic_status", EpistemicStatus(self.epistemic_status))
        except ValueError as exc:
            raise KnowledgeValidationError(str(exc)) from exc
        for name in ("knowledge_id", "domain", "subject", "predicate", "statement"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise KnowledgeValidationError(f"{name} must not be empty")
        if self.schema_version != 1:
            raise KnowledgeValidationError(f"unsupported knowledge schema: {self.schema_version}")
        if not isinstance(self.provenance, RepresentationProvenance):
            raise KnowledgeValidationError("provenance must be RepresentationProvenance")
        if not isinstance(self.verification, VerificationEvidence):
            raise KnowledgeValidationError("verification must be VerificationEvidence")
        if not isinstance(self.content, Mapping):
            raise KnowledgeValidationError("content must be a mapping")
        if self.capability_contract is not None and not isinstance(self.capability_contract, Mapping):
            raise KnowledgeValidationError("capability_contract must be a mapping or None")
        object.__setattr__(self, "knowledge_id", self.knowledge_id.strip())
        object.__setattr__(self, "domain", self.domain.strip().casefold())
        object.__setattr__(self, "subject", self.subject.strip())
        object.__setattr__(self, "predicate", self.predicate.strip().casefold())
        object.__setattr__(self, "statement", self.statement.strip())
        object.__setattr__(self, "object_value", _freeze_json(self.object_value, "object_value"))
        object.__setattr__(self, "content", _freeze_json(self.content, "content"))
        object.__setattr__(self, "tags", _unique(self.tags, "tags"))
        if self.capability_contract is not None:
            object.__setattr__(
                self, "capability_contract",
                _freeze_json(self.capability_contract, "capability_contract"),
            )
        if self.kind is KnowledgeKind.ASSUMPTION and self.epistemic_status is not EpistemicStatus.ASSUMED:
            raise KnowledgeValidationError("assumption records must retain ASSUMED epistemic status")
        if self.epistemic_status is EpistemicStatus.ASSUMED and self.kind is not KnowledgeKind.ASSUMPTION:
            # A fact may be a working assumption, but it must be explicitly typed as such.
            raise KnowledgeValidationError("ASSUMED records must use KnowledgeKind.ASSUMPTION")

    @property
    def claim_key(self) -> str:
        """Stable key for a proposition independent of its asserted object value."""
        return stable_representation_id("claim-key", {
            "domain": self.domain,
            "subject": _normalize(self.subject),
            "predicate": self.predicate,
        })

    @property
    def object_fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(self.object_value).encode("utf-8")).hexdigest()

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def has_passing_evidence(self) -> bool:
        evidence = self.verification
        return bool(evidence.checks_passed) and not evidence.checks_failed

    @property
    def assertable(self) -> bool:
        """Only source-backed, formally/independently verified claims may be asserted as facts."""
        return (
            self.kind in {KnowledgeKind.FACT, KnowledgeKind.DEFINITION, KnowledgeKind.OBSERVATION}
            and self.epistemic_status not in {
                EpistemicStatus.ASSUMED, EpistemicStatus.DISPUTED, EpistemicStatus.UNKNOWN
            }
            and self.verification.level in _TRUSTED_FACT_EVIDENCE
            and self.has_passing_evidence
            and self.provenance.source != "unknown"
            and bool(self.provenance.source_ref.strip())
            and bool(self.provenance.content_sha256)
            and (
                self.content.get("knowledge_source") != "canonical_blueprint"
                or self.content.get("claim_evidence_reliable") is True
            )
        )

    @property
    def executable_procedure(self) -> bool:
        """A procedure can become a capability only with an explicit typed operation contract."""
        return (
            self.kind is KnowledgeKind.PROCEDURE
            and self.capability_contract is not None
            and self.verification.level in _OPERATION_EVIDENCE
            and self.has_passing_evidence
            and self.provenance.source != "unknown"
            and bool(self.provenance.source_ref.strip())
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "knowledge_id": self.knowledge_id,
            "kind": self.kind.value,
            "domain": self.domain,
            "subject": self.subject,
            "predicate": self.predicate,
            "object_value": _thaw_json(self.object_value),
            "statement": self.statement,
            "epistemic_status": self.epistemic_status.value,
            "content": _thaw_json(self.content),
            "tags": list(self.tags),
            "provenance": self.provenance.to_dict(),
            "verification": {
                "level": self.verification.level.value,
                "checks_passed": list(self.verification.checks_passed),
                "checks_failed": list(self.verification.checks_failed),
                "verifier": self.verification.verifier,
                "verified_at": self.verification.verified_at,
            },
            "capability_contract": (
                _thaw_json(self.capability_contract)
                if self.capability_contract is not None else None
            ),
            "schema_version": self.schema_version,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KnowledgeRecord":
        data = dict(value)
        data["kind"] = KnowledgeKind(data["kind"])
        data["epistemic_status"] = EpistemicStatus(data.get("epistemic_status", "asserted"))
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
        data["tags"] = tuple(data.get("tags", ()))
        return cls(**data)

    @classmethod
    def from_blueprint(
        cls,
        blueprint: Blueprint,
        *,
        claim_evidence_report: Any = None,
    ) -> "KnowledgeRecord":
        """Expose a blueprint without promoting evidence; facts also need claim-level review."""
        if not isinstance(blueprint, Blueprint):
            raise TypeError("blueprint must be a Blueprint")
        kind_map = {
            "fact": KnowledgeKind.FACT,
            "procedure": KnowledgeKind.PROCEDURE,
            "formula": KnowledgeKind.FORMULA,
            "policy": KnowledgeKind.RULE,
            "heuristic": KnowledgeKind.HEURISTIC,
            "conversation": KnowledgeKind.DEFINITION,
        }
        provenance = blueprint.provenance
        return cls(
            knowledge_id=f"blueprint:{blueprint.blueprint_id}",
            kind=kind_map[blueprint.kind.value],
            domain=blueprint.domain,
            subject=blueprint.intent,
            predicate=blueprint.kind.value,
            object_value=blueprint.response_template,
            statement=blueprint.response_template,
            epistemic_status=EpistemicStatus.ASSERTED,
            content={
                "prompt_pattern": blueprint.prompt_pattern,
                "slots": list(blueprint.slots),
                "derived_expr": blueprint.derived_expr,
                "preconditions": list(blueprint.preconditions),
                "postconditions": list(blueprint.postconditions),
                "dependencies": list(blueprint.dependencies),
                "legacy_trust_tier": blueprint.legacy_trust_tier,
                "knowledge_source": "canonical_blueprint",
                "claim_evidence_reliable": bool(
                    claim_evidence_report is not None
                    and getattr(claim_evidence_report, "reliable", False)
                ),
                "verified_claim_ids": list(
                    getattr(claim_evidence_report, "verified_claim_ids", ())
                ),
            },
            tags=tuple(dict.fromkeys((blueprint.domain, blueprint.intent, blueprint.kind.value))),
            provenance=RepresentationProvenance(
                source=provenance.source,
                source_ref=provenance.source_ref,
                license_spdx=provenance.license_spdx,
                content_sha256=provenance.content_sha256,
                captured_at=provenance.captured_at,
                parent_ids=blueprint.dependencies,
            ),
            verification=blueprint.verification,
        )

    @classmethod
    def from_crystal_rule(cls, rule: Any) -> "KnowledgeRecord":
        """Adapt a legacy CrystalRule through the unverified canonical blueprint boundary."""
        return cls.from_blueprint(Blueprint.from_crystal_rule(rule))

    def to_representation_node(self) -> RepresentationNode:
        """Adapt into Phase 2 graph structure without changing epistemic/evidence status."""
        return RepresentationNode(
            node_id=f"knowledge:{self.knowledge_id}",
            kind=_KIND_TO_REPRESENTATION[self.kind],
            domain=self.domain,
            label=self.statement[:240],
            content={
                "node_role": "knowledge_record",
                "knowledge_record": self.to_dict(),
                "claim_key": self.claim_key,
                "object_fingerprint": self.object_fingerprint,
            },
            value_type="record",
            status=self.epistemic_status,
            attributes={
                "knowledge_kind": self.kind.value,
                "assertable": self.assertable,
                "executable_procedure": self.executable_procedure,
            },
            provenance=self.provenance,
            verification=self.verification,
        )

    def to_capability_spec(self):
        """Bridge a procedure to Phase 4 only when its typed contract is explicit."""
        if not self.executable_procedure:
            raise KnowledgeValidationError(
                "knowledge item is not an evidence-eligible procedure with a typed capability contract"
            )
        from .capability_model import CapabilitySpec, PortContract

        contract = dict(self.capability_contract or {})
        try:
            domains = tuple(contract.get("domains", (self.domain,)))
            intents = tuple(contract["intents"])
            inputs = tuple(PortContract.from_dict(item) for item in contract["inputs"])
            outputs = tuple(PortContract.from_dict(item) for item in contract["outputs"])
        except (KeyError, TypeError, ValueError) as exc:
            raise KnowledgeValidationError(f"invalid procedure capability contract: {exc}") from exc
        return CapabilitySpec(
            capability_id=f"knowledge:{self.knowledge_id}",
            name=self.subject,
            description=self.statement,
            domains=domains,
            intents=intents,
            inputs=inputs,
            outputs=outputs,
            preconditions=tuple(contract.get("preconditions", ())),
            postconditions=tuple(contract.get("postconditions", ())),
            dependencies=tuple(contract.get("dependencies", ())),
            provenance=self.provenance,
            verification=self.verification,
        )


@dataclass(frozen=True)
class KnowledgeConflict:
    claim_key: str
    domain: str
    subject: str
    predicate: str
    knowledge_ids: tuple[str, ...]
    object_fingerprints: tuple[str, ...]


@dataclass(frozen=True)
class KnowledgeSearchResult:
    record: KnowledgeRecord
    score: float
    conflicted: bool
    assertable: bool


class KnowledgeRegistry:
    """Deterministic local knowledge catalog with explicit conflict-aware retrieval."""

    def __init__(self, records: Iterable[KnowledgeRecord] = ()) -> None:
        self._records: dict[str, KnowledgeRecord] = {}
        for record in records:
            self.add(record)

    def add(self, record: KnowledgeRecord) -> str:
        if not isinstance(record, KnowledgeRecord):
            raise TypeError("record must be KnowledgeRecord")
        previous = self._records.get(record.knowledge_id)
        if previous is not None and previous.fingerprint != record.fingerprint:
            raise KnowledgeConflictError(
                f"knowledge identifier has conflicting content: {record.knowledge_id}"
            )
        self._records[record.knowledge_id] = record
        return record.knowledge_id

    def get(self, knowledge_id: str) -> KnowledgeRecord | None:
        return self._records.get(str(knowledge_id))

    def list(
        self,
        *,
        domain: str | None = None,
        kind: KnowledgeKind | str | None = None,
        limit: int = 10000,
    ) -> tuple[KnowledgeRecord, ...]:
        items = self._records.values()
        if domain is not None:
            key = str(domain).strip().casefold()
            items = (item for item in items if item.domain == key)
        if kind is not None:
            selected_kind = KnowledgeKind(kind)
            items = (item for item in items if item.kind is selected_kind)
        return tuple(sorted(items, key=lambda item: item.knowledge_id)[:max(0, min(int(limit), 10000))])

    def conflicts(self) -> tuple[KnowledgeConflict, ...]:
        groups: dict[str, list[KnowledgeRecord]] = {}
        for item in self._records.values():
            groups.setdefault(item.claim_key, []).append(item)
        result: list[KnowledgeConflict] = []
        for claim_key, records in groups.items():
            value_groups: dict[str, list[KnowledgeRecord]] = {}
            for item in records:
                value_groups.setdefault(item.object_fingerprint, []).append(item)
            if len(value_groups) < 2:
                continue
            result.append(KnowledgeConflict(
                claim_key=claim_key,
                domain=records[0].domain,
                subject=records[0].subject,
                predicate=records[0].predicate,
                knowledge_ids=tuple(sorted(item.knowledge_id for item in records)),
                object_fingerprints=tuple(sorted(value_groups)),
            ))
        return tuple(sorted(result, key=lambda item: item.claim_key))

    def query(
        self,
        text: str,
        *,
        domain: str | None = None,
        kind: KnowledgeKind | str | None = None,
        limit: int = 20,
        include_assumptions: bool = True,
    ) -> tuple[KnowledgeSearchResult, ...]:
        """Return transparent lexical matches; this is not semantic entailment or proof."""
        if not isinstance(text, str) or not text.strip():
            raise KnowledgeValidationError("query text must be non-empty")
        terms = set(_tokens(text))
        if not terms:
            return ()
        conflicts = {item.claim_key for item in self.conflicts()}
        matches: list[KnowledgeSearchResult] = []
        for record in self._records.values():
            if domain is not None and record.domain != str(domain).strip().casefold():
                continue
            if kind is not None and record.kind is not KnowledgeKind(kind):
                continue
            if not include_assumptions and (
                record.kind is KnowledgeKind.ASSUMPTION
                or record.epistemic_status is EpistemicStatus.ASSUMED
            ):
                continue
            haystack = " ".join((
                record.subject, record.predicate, record.statement,
                " ".join(record.tags), canonical_json(record.content),
            ))
            present = terms.intersection(_tokens(haystack))
            if not present:
                continue
            score = len(present) / len(terms)
            conflicted = record.claim_key in conflicts or record.epistemic_status is EpistemicStatus.DISPUTED
            matches.append(KnowledgeSearchResult(
                record=record,
                score=score,
                conflicted=conflicted,
                assertable=record.assertable and not conflicted,
            ))
        matches.sort(key=lambda item: (
            -item.score,
            item.conflicted,
            item.record.knowledge_id,
        ))
        return tuple(matches[:max(0, min(int(limit), 1000))])

    def __len__(self) -> int:
        return len(self._records)
