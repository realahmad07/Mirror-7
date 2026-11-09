"""Domain-specific, evidence-carrying verification (Strategic Phase 6).

This module does not pretend to supply proof assistants, compilers, simulators or
scientific instruments. It defines the fail-closed contract through which such
independent domain verifiers must report their results to Mirror.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import re
from typing import Any, Callable, Iterable, Mapping

from .capability_model import CapabilitySpec
from .execution import ExecutionStatus, PlanExecutionReport
from .universal_representation import _freeze_json, _thaw_json, canonical_json
from .verification import VerificationContext


class DomainVerificationError(ValueError):
    """Raised for malformed domain verification contracts or evidence."""


class DomainDecision(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class DomainVerificationStatus(str, Enum):
    VERIFIED = "verified"
    FAILED = "failed"
    BLOCKED = "blocked"


def _slug(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise DomainVerificationError(f"{field_name} must be a string")
    normalized = value.strip().casefold()
    if not normalized or not re.fullmatch(r"[a-z0-9][a-z0-9_.:-]{0,127}", normalized):
        raise DomainVerificationError(f"{field_name} must be a non-empty identifier")
    return normalized


@dataclass(frozen=True)
class DomainEvidenceArtifact:
    """Immutable evidence bound to one capability, step and execution.

    evidence_payload is frozen JSON data. Its SHA-256 digest is computed from the
    canonical payload; callers cannot supply an unrelated digest. evidence_ref
    identifies the underlying report, proof, dataset, source bundle or simulation
    artifact. The reference is an audit locator, not proof that an external source
    is authentic; the registered verifier remains responsible for checking it.
    """

    evidence_id: str
    domain: str
    evidence_kind: str
    decision: DomainDecision
    step_id: str
    capability_id: str
    execution_fingerprint: str
    verifier_id: str
    evidence_ref: str
    evidence_payload: Mapping[str, Any]
    verified_conditions: tuple[str, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        for name in ("evidence_id", "step_id", "capability_id", "execution_fingerprint",
                     "verifier_id", "evidence_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DomainVerificationError(f"{name} must not be empty")
            object.__setattr__(self, name, value.strip())
        object.__setattr__(self, "domain", _slug(self.domain, "domain"))
        object.__setattr__(self, "evidence_kind", _slug(self.evidence_kind, "evidence_kind"))
        object.__setattr__(self, "decision", DomainDecision(self.decision))
        if not isinstance(self.detail, str):
            raise DomainVerificationError("detail must be a string")
        if not isinstance(self.evidence_payload, Mapping) or not self.evidence_payload:
            raise DomainVerificationError("evidence_payload must be a non-empty JSON mapping")
        frozen = _freeze_json(dict(self.evidence_payload), "domain_evidence.payload")
        object.__setattr__(self, "evidence_payload", frozen)
        conditions = tuple(str(value).strip() for value in self.verified_conditions)
        if any(not value for value in conditions) or len(set(conditions)) != len(conditions):
            raise DomainVerificationError("verified_conditions must be non-empty and unique")
        object.__setattr__(self, "verified_conditions", conditions)
        if self.decision is not DomainDecision.PASS and not self.detail.strip():
            raise DomainVerificationError("failed or inconclusive evidence requires a reason")

    @property
    def payload_sha256(self) -> str:
        payload = canonical_json(_thaw_json(self.evidence_payload)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def fingerprint(self) -> str:
        body = {
            "evidence_id": self.evidence_id,
            "domain": self.domain,
            "evidence_kind": self.evidence_kind,
            "decision": self.decision.value,
            "step_id": self.step_id,
            "capability_id": self.capability_id,
            "execution_fingerprint": self.execution_fingerprint,
            "verifier_id": self.verifier_id,
            "evidence_ref": self.evidence_ref,
            "payload_sha256": self.payload_sha256,
            "verified_conditions": list(self.verified_conditions),
            "detail": self.detail,
        }
        return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


DomainEvidenceCallback = Callable[[VerificationContext], DomainEvidenceArtifact]


@dataclass(frozen=True)
class DomainVerificationCheck:
    """An allow-listed domain verifier; it must return bound evidence, not a bool."""

    check_id: str
    verifier_id: str
    domain: str
    evidence_kind: str
    callback: DomainEvidenceCallback

    def __post_init__(self) -> None:
        object.__setattr__(self, "check_id", _slug(self.check_id, "check_id"))
        object.__setattr__(self, "verifier_id", _slug(self.verifier_id, "verifier_id"))
        object.__setattr__(self, "domain", _slug(self.domain, "domain"))
        object.__setattr__(self, "evidence_kind", _slug(self.evidence_kind, "evidence_kind"))
        if not callable(self.callback):
            raise DomainVerificationError("domain verifier callback must be callable")


class DomainVerifierRegistry:
    """Host-controlled allow-list of domain-specific evidence producers."""

    def __init__(self) -> None:
        self._checks: dict[tuple[str, str], DomainVerificationCheck] = {}

    def register(self, check: DomainVerificationCheck, *, independent: bool) -> None:
        if not isinstance(check, DomainVerificationCheck):
            raise TypeError("check must be DomainVerificationCheck")
        if independent is not True:
            raise DomainVerificationError("domain verifier must be explicitly marked independent")
        key = (check.domain, check.check_id)
        previous = self._checks.get(key)
        if previous is not None and previous != check:
            raise DomainVerificationError(
                f"conflicting domain verifier registration: {check.domain}/{check.check_id}"
            )
        self._checks[key] = check

    def get(self, domain: str, check_id: str) -> DomainVerificationCheck | None:
        return self._checks.get((_slug(domain, "domain"), _slug(check_id, "check_id")))

    def list(self, *, domain: str | None = None) -> tuple[DomainVerificationCheck, ...]:
        values = self._checks.values()
        if domain is not None:
            selected = _slug(domain, "domain")
            values = (item for item in values if item.domain == selected)
        return tuple(sorted(values, key=lambda item: (item.domain, item.check_id)))


@dataclass(frozen=True)
class DomainVerificationRequirement:
    """Evidence kinds and independent checker floor for one capability/domain."""

    domain: str
    evidence_kinds: tuple[str, ...]
    minimum_independent_verifiers: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", _slug(self.domain, "domain"))
        kinds = tuple(_slug(value, "evidence_kind") for value in self.evidence_kinds)
        if not kinds or len(set(kinds)) != len(kinds):
            raise DomainVerificationError("evidence_kinds must be non-empty and unique")
        object.__setattr__(self, "evidence_kinds", kinds)
        count = self.minimum_independent_verifiers
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise DomainVerificationError("minimum_independent_verifiers must be >= 1")


@dataclass(frozen=True)
class DomainCheckResult:
    step_id: str
    capability_id: str
    domain: str
    check_id: str
    verifier_id: str
    decision: DomainDecision
    evidence_fingerprint: str = ""
    payload_sha256: str = ""
    detail: str = ""


@dataclass(frozen=True)
class DomainCapabilityVerification:
    step_id: str
    capability_id: str
    domain: str
    passed: bool
    evidence_passed: tuple[str, ...]
    evidence_failed: tuple[str, ...]
    verified_conditions: tuple[str, ...]
    verifier_ids: tuple[str, ...]
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class DomainVerificationReport:
    plan_id: str
    task_id: str
    status: DomainVerificationStatus
    results: tuple[DomainCheckResult, ...]
    capabilities: tuple[DomainCapabilityVerification, ...]
    verified_conditions: tuple[str, ...]
    blockers: tuple[str, ...]
    execution_fingerprint: str

    @property
    def verified(self) -> bool:
        return self.status is DomainVerificationStatus.VERIFIED

    @property
    def verifier_ids(self) -> tuple[str, ...]:
        return tuple(sorted({item.verifier_id for item in self.results if item.verifier_id}))

    @property
    def fingerprint(self) -> str:
        payload = {
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "status": self.status.value,
            "execution_fingerprint": self.execution_fingerprint,
            "results": [
                {
                    "step_id": item.step_id,
                    "capability_id": item.capability_id,
                    "domain": item.domain,
                    "check_id": item.check_id,
                    "verifier_id": item.verifier_id,
                    "decision": item.decision.value,
                    "evidence_fingerprint": item.evidence_fingerprint,
                    "payload_sha256": item.payload_sha256,
                    "detail": item.detail,
                }
                for item in self.results
            ],
            "verified_conditions": list(self.verified_conditions),
            "blockers": list(self.blockers),
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class DomainVerificationEngine:
    """Require domain-specific evidence for every executed capability step.

    required_checks maps capability ID to evidence-kind -> registered check ID.
    A check passes only when its artifact is bound to the exact step, capability,
    domain and execution fingerprint, and covers declared postconditions.
    """

    VERSION = "domain-evidence-verification-v1"

    def __init__(self, registry: DomainVerifierRegistry) -> None:
        if not isinstance(registry, DomainVerifierRegistry):
            raise TypeError("registry must be DomainVerifierRegistry")
        self.registry = registry

    def verify(
        self,
        execution: PlanExecutionReport,
        capabilities: Iterable[CapabilitySpec],
        *,
        requirements: Mapping[str, DomainVerificationRequirement],
        required_checks: Mapping[str, Mapping[str, str]],
        _candidate_capability_ids: Iterable[str] = (),
    ) -> DomainVerificationReport:
        if not isinstance(execution, PlanExecutionReport):
            raise TypeError("execution must be PlanExecutionReport")
        if not isinstance(requirements, Mapping) or not isinstance(required_checks, Mapping):
            raise TypeError("requirements and required_checks must be mappings")
        if execution.status is not ExecutionStatus.SUCCEEDED:
            return DomainVerificationReport(
                execution.plan_id, execution.task_id, DomainVerificationStatus.BLOCKED,
                (), (), (), ("execution did not succeed; domain verification cannot pass",),
                execution.fingerprint,
            )

        catalog = tuple(capabilities)
        capability_map = {item.capability_id: item for item in catalog}
        candidate_ids = set(_candidate_capability_ids)
        if not candidate_ids.issubset(capability_map):
            raise DomainVerificationError("candidate verification references an absent capability")
        for candidate_id in candidate_ids:
            candidate = capability_map[candidate_id]
            if candidate.executable or candidate.provenance.source not in {
                "mirror_uol_synthesizer", "mirror_cross_domain_transfer",
            }:
                raise DomainVerificationError(
                    "only unverified UOL synthesis/transfer proposals may use candidate verification"
                )
        results: list[DomainCheckResult] = []
        summaries: list[DomainCapabilityVerification] = []
        all_conditions: set[str] = set()
        blockers: list[str] = []
        seen_evidence_ids: set[str] = set()

        for step in execution.steps:
            local_blockers: list[str] = []
            passed: list[str] = []
            failed: list[str] = []
            conditions: set[str] = set()
            verifiers: set[str] = set()
            capability = capability_map.get(step.capability_id)
            requirement = requirements.get(step.capability_id)
            check_map = required_checks.get(step.capability_id)

            if capability is None:
                local_blockers.append(f"missing capability contract: {step.capability_id}")
            if requirement is None or not isinstance(requirement, DomainVerificationRequirement):
                local_blockers.append(f"missing domain verification requirement: {step.capability_id}")
            if not isinstance(check_map, Mapping):
                local_blockers.append(f"missing required domain checks: {step.capability_id}")

            if local_blockers:
                blockers.extend(local_blockers)
                summaries.append(DomainCapabilityVerification(
                    step.step_id, step.capability_id,
                    requirement.domain if isinstance(requirement, DomainVerificationRequirement) else "",
                    False, (), (), (), (), tuple(local_blockers),
                ))
                continue

            assert capability is not None
            assert isinstance(requirement, DomainVerificationRequirement)
            assert isinstance(check_map, Mapping)
            if not capability.executable and capability.capability_id not in candidate_ids:
                local_blockers.append(
                    f"capability is not evidence-eligible for domain verification: {capability.capability_id}"
                )
            if requirement.domain not in capability.domains and "*" not in capability.domains:
                local_blockers.append(
                    f"verification domain {requirement.domain} is not declared by {capability.capability_id}"
                )
            expected_kinds = set(requirement.evidence_kinds)
            provided_kinds = {_slug(str(key), "required evidence kind") for key in check_map}
            if provided_kinds != expected_kinds:
                local_blockers.append(
                    f"required check map must match evidence kinds for {capability.capability_id}"
                )

            context = VerificationContext(
                execution.task_id, execution.plan_id, step, capability, execution.fingerprint,
            )
            if not local_blockers:
                for evidence_kind in requirement.evidence_kinds:
                    check_id = check_map.get(evidence_kind)
                    if check_id is None:
                        # Accept enum-like mapping keys only after normalizing them.
                        check_id = next(
                            (value for key, value in check_map.items()
                             if _slug(str(key), "required evidence kind") == evidence_kind),
                            None,
                        )
                    check = self.registry.get(requirement.domain, str(check_id)) if check_id else None
                    if check is None:
                        failed.append(evidence_kind)
                        results.append(DomainCheckResult(
                            step.step_id, capability.capability_id, requirement.domain,
                            str(check_id or "missing"), "", DomainDecision.FAIL,
                            detail=f"no registered verifier for required evidence kind {evidence_kind}",
                        ))
                        continue
                    if check.evidence_kind != evidence_kind:
                        failed.append(evidence_kind)
                        results.append(DomainCheckResult(
                            step.step_id, capability.capability_id, requirement.domain,
                            check.check_id, check.verifier_id, DomainDecision.FAIL,
                            detail="registered verifier evidence kind does not match policy",
                        ))
                        continue
                    try:
                        artifact = check.callback(context)
                        if not isinstance(artifact, DomainEvidenceArtifact):
                            raise DomainVerificationError("verifier must return DomainEvidenceArtifact, not bool")
                        mismatches = []
                        expected = {
                            "domain": requirement.domain,
                            "evidence_kind": evidence_kind,
                            "step_id": step.step_id,
                            "capability_id": capability.capability_id,
                            "execution_fingerprint": execution.fingerprint,
                            "verifier_id": check.verifier_id,
                        }
                        for name, value in expected.items():
                            if getattr(artifact, name) != value:
                                mismatches.append(name)
                        if mismatches:
                            raise DomainVerificationError(
                                "evidence binding mismatch: " + ", ".join(mismatches)
                            )
                        if artifact.evidence_id in seen_evidence_ids:
                            raise DomainVerificationError("evidence_id was reused in this execution")
                        seen_evidence_ids.add(artifact.evidence_id)
                        undeclared = sorted(set(artifact.verified_conditions) - set(capability.postconditions))
                        if undeclared:
                            raise DomainVerificationError(
                                "evidence asserts undeclared postconditions: " + ", ".join(undeclared)
                            )
                        if artifact.decision is DomainDecision.PASS:
                            if not artifact.verified_conditions:
                                raise DomainVerificationError(
                                    "passing evidence must verify at least one declared postcondition"
                                )
                            passed.append(evidence_kind)
                            conditions.update(artifact.verified_conditions)
                        elif artifact.decision is DomainDecision.FAIL:
                            failed.append(evidence_kind)
                        else:
                            local_blockers.append(
                                f"{evidence_kind} is inconclusive: {artifact.detail}"
                            )
                        verifiers.add(check.verifier_id)
                        results.append(DomainCheckResult(
                            step.step_id, capability.capability_id, requirement.domain,
                            check.check_id, check.verifier_id, artifact.decision,
                            artifact.fingerprint, artifact.payload_sha256, artifact.detail,
                        ))
                    except Exception as exc:
                        failed.append(evidence_kind)
                        results.append(DomainCheckResult(
                            step.step_id, capability.capability_id, requirement.domain,
                            check.check_id, check.verifier_id, DomainDecision.FAIL,
                            detail=f"{type(exc).__name__}: {str(exc)[:500]}",
                        ))

            missing_conditions = sorted(set(capability.postconditions) - conditions)
            if missing_conditions:
                local_blockers.append(
                    "postconditions not independently evidenced: " + ", ".join(missing_conditions)
                )
            if len(verifiers) < requirement.minimum_independent_verifiers:
                local_blockers.append(
                    f"requires {requirement.minimum_independent_verifiers} distinct verifier identities; "
                    f"received {len(verifiers)}"
                )
            local_blockers = list(dict.fromkeys(local_blockers))
            blockers.extend(local_blockers)
            capability_passed = not failed and not local_blockers and len(passed) == len(expected_kinds)
            summaries.append(DomainCapabilityVerification(
                step.step_id, capability.capability_id, requirement.domain, capability_passed,
                tuple(passed), tuple(failed), tuple(sorted(conditions)), tuple(sorted(verifiers)),
                tuple(local_blockers),
            ))
            if capability_passed:
                all_conditions.update(conditions)

        if not summaries or len(summaries) != len(execution.steps):
            blockers.append("one or more execution steps lack domain verification")
        if any(item.evidence_failed for item in summaries):
            status = DomainVerificationStatus.FAILED
        elif blockers or any(not item.passed for item in summaries):
            status = DomainVerificationStatus.BLOCKED
        else:
            status = DomainVerificationStatus.VERIFIED
        return DomainVerificationReport(
            execution.plan_id, execution.task_id, status, tuple(results), tuple(summaries),
            tuple(sorted(all_conditions)), tuple(dict.fromkeys(blockers)), execution.fingerprint,
        )


    def verify_candidate(
        self,
        execution: PlanExecutionReport,
        candidate: CapabilitySpec,
        *,
        requirements: Mapping[str, DomainVerificationRequirement],
        required_checks: Mapping[str, Mapping[str, str]],
    ) -> DomainVerificationReport:
        """Independently verify one unpromoted UOL candidate without making it executable.

        Candidate verification produces evidence only. It does not alter the capability's
        UNVERIFIED state; repeated verified observations must still pass the learning gate.
        """
        if not isinstance(candidate, CapabilitySpec):
            raise TypeError("candidate must be CapabilitySpec")
        if candidate.executable or candidate.provenance.source not in {
            "mirror_uol_synthesizer", "mirror_cross_domain_transfer",
        }:
            raise DomainVerificationError("candidate must be an unverified UOL synthesis/transfer proposal")
        return self.verify(
            execution, (candidate,), requirements=requirements,
            required_checks=required_checks,
            _candidate_capability_ids=(candidate.capability_id,),
        )
