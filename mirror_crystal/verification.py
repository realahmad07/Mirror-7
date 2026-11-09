"""Independent, claim-specific verification of capability executions (Phase 8).

Verifier callbacks are registered separately from operation handlers. Execution success
alone never becomes verification evidence; every required check must independently pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Any, Callable, Iterable, Mapping

from .capability_model import CapabilitySpec
from .execution import PlanExecutionReport, ExecutionStatus, StepExecutionResult
from .universal_representation import canonical_json


class VerificationError(ValueError):
    """Raised for invalid verification policy or reports."""


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class VerificationContext:
    task_id: str
    plan_id: str
    step: StepExecutionResult
    capability: CapabilitySpec
    execution_fingerprint: str


VerificationCallback = Callable[[VerificationContext], bool]


@dataclass(frozen=True)
class IndependentCheck:
    check_id: str
    verifier_id: str
    callback: VerificationCallback
    verifies_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.check_id.strip() or not self.verifier_id.strip():
            raise VerificationError("check_id and verifier_id must not be empty")
        if not callable(self.callback):
            raise TypeError("verification callback must be callable")
        conditions = tuple(str(value).strip() for value in self.verifies_conditions)
        if any(not value for value in conditions) or len(set(conditions)) != len(conditions):
            raise VerificationError("verified condition identifiers must be non-empty and unique")
        object.__setattr__(self, "verifies_conditions", conditions)


class IndependentVerifierRegistry:
    """Explicit independent checker allow-list; deliberately distinct from operation handlers."""
    def __init__(self) -> None:
        self._checks: dict[tuple[str, str], IndependentCheck] = {}

    def register(self, check: IndependentCheck, *, independent: bool) -> None:
        if not isinstance(check, IndependentCheck):
            raise TypeError("check must be IndependentCheck")
        if independent is not True:
            raise VerificationError("verification callbacks must be explicitly marked independent")
        key = (check.verifier_id, check.check_id)
        previous = self._checks.get(key)
        if previous is not None and previous.callback is not check.callback:
            raise VerificationError(f"conflicting verifier check: {check.verifier_id}/{check.check_id}")
        self._checks[key] = check

    def get(self, verifier_id: str, check_id: str) -> IndependentCheck | None:
        return self._checks.get((str(verifier_id), str(check_id)))

    def for_check(self, check_id: str) -> tuple[IndependentCheck, ...]:
        return tuple(sorted(
            (item for (verifier, name), item in self._checks.items() if name == check_id),
            key=lambda item: item.verifier_id,
        ))


@dataclass(frozen=True)
class CheckResult:
    step_id: str
    capability_id: str
    check_id: str
    verifier_id: str
    passed: bool
    verified_conditions: tuple[str, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.step_id or not self.capability_id or not self.check_id or not self.verifier_id:
            raise VerificationError("check result identifiers must not be empty")
        if not isinstance(self.passed, bool):
            raise VerificationError("check result passed must be boolean")
        object.__setattr__(self, "verified_conditions", tuple(self.verified_conditions))


@dataclass(frozen=True)
class CapabilityVerification:
    capability_id: str
    passed: bool
    checks_passed: tuple[str, ...]
    checks_failed: tuple[str, ...]
    verified_conditions: tuple[str, ...]
    verifier_ids: tuple[str, ...]


@dataclass(frozen=True)
class VerificationReport:
    plan_id: str
    task_id: str
    status: VerificationStatus
    results: tuple[CheckResult, ...]
    capabilities: tuple[CapabilityVerification, ...]
    verified_conditions: tuple[str, ...]
    blockers: tuple[str, ...] = ()
    execution_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", VerificationStatus(self.status))
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "verified_conditions", tuple(sorted(set(self.verified_conditions))))
        object.__setattr__(self, "blockers", tuple(self.blockers))

    @property
    def verified(self) -> bool:
        return self.status is VerificationStatus.VERIFIED

    @property
    def verifier_ids(self) -> tuple[str, ...]:
        return tuple(sorted({item.verifier_id for item in self.results}))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json({
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "status": self.status.value,
            "execution_fingerprint": self.execution_fingerprint,
            "results": [
                {
                    "step_id": item.step_id,
                    "capability_id": item.capability_id,
                    "check_id": item.check_id,
                    "verifier_id": item.verifier_id,
                    "passed": item.passed,
                    "verified_conditions": list(item.verified_conditions),
                    "detail": item.detail,
                }
                for item in self.results
            ],
            "verified_conditions": list(self.verified_conditions),
            "blockers": list(self.blockers),
        }).encode("utf-8")).hexdigest()


class IndependentVerificationEngine:
    """Verify every executed step with explicitly required independent checks."""

    VERSION = "independent-verification-v1"

    def __init__(self, registry: IndependentVerifierRegistry) -> None:
        if not isinstance(registry, IndependentVerifierRegistry):
            raise TypeError("registry must be IndependentVerifierRegistry")
        self.registry = registry

    def verify(
        self,
        execution: PlanExecutionReport,
        capabilities: Iterable[CapabilitySpec],
        *,
        required_checks: Mapping[str, Iterable[str]],
    ) -> VerificationReport:
        if not isinstance(execution, PlanExecutionReport):
            raise TypeError("execution must be PlanExecutionReport")
        if not isinstance(required_checks, Mapping):
            raise TypeError("required_checks must be a mapping")
        capability_map = {item.capability_id: item for item in capabilities}
        if execution.status is not ExecutionStatus.SUCCEEDED:
            return VerificationReport(
                execution.plan_id, execution.task_id, VerificationStatus.BLOCKED,
                (), (), (), ("execution did not succeed; independent verification cannot pass",),
                execution.fingerprint,
            )

        results: list[CheckResult] = []
        capability_results: list[CapabilityVerification] = []
        blockers: list[str] = []
        all_conditions: set[str] = set()
        step_capability_ids = {step.capability_id for step in execution.steps}

        for step in execution.steps:
            capability = capability_map.get(step.capability_id)
            if capability is None:
                blockers.append(f"missing capability contract for {step.capability_id}")
                continue
            if not capability.executable:
                blockers.append(f"{step.capability_id} is not evidence-eligible for verification")
                capability_results.append(CapabilityVerification(
                    step.capability_id, False, (), (), (), ()
                ))
                continue
            required = tuple(dict.fromkeys(str(value).strip() for value in required_checks.get(
                step.capability_id, ()
            )))
            if not required or any(not item for item in required):
                blockers.append(f"{step.capability_id} has no explicit required independent checks")
                capability_results.append(CapabilityVerification(
                    step.capability_id, False, (), (), (), ()
                ))
                continue

            passed_ids: list[str] = []
            failed_ids: list[str] = []
            conditions: set[str] = set()
            verifier_ids: set[str] = set()
            for check_id in required:
                checks = self.registry.for_check(check_id)
                if not checks:
                    failed_ids.append(check_id)
                    results.append(CheckResult(
                        step.step_id, step.capability_id, check_id, "missing",
                        False, (), "required independent check is not registered",
                    ))
                    continue
                # A required check must have one unambiguous owner. Multiple
                # verifier implementations must be named explicitly, not guessed.
                if len(checks) != 1:
                    failed_ids.append(check_id)
                    results.append(CheckResult(
                        step.step_id, step.capability_id, check_id, "ambiguous",
                        False, (), "required check id is registered by multiple verifiers",
                    ))
                    continue
                check = checks[0]
                undeclared = sorted(set(check.verifies_conditions) - set(capability.postconditions))
                if undeclared:
                    failed_ids.append(check_id)
                    results.append(CheckResult(
                        step.step_id, step.capability_id, check_id, check.verifier_id,
                        False, (), "check references undeclared postconditions: " + ", ".join(undeclared),
                    ))
                    continue
                context = VerificationContext(
                    execution.task_id, execution.plan_id, step, capability,
                    execution.fingerprint,
                )
                try:
                    result = check.callback(context)
                    if not isinstance(result, bool):
                        raise TypeError("independent check must return a boolean")
                    passed = result
                    detail = "" if passed else "independent check returned false"
                except Exception as exc:
                    passed = False
                    detail = f"{type(exc).__name__}: {str(exc)[:500]}"
                results.append(CheckResult(
                    step.step_id, step.capability_id, check.check_id, check.verifier_id,
                    passed, check.verifies_conditions if passed else (), detail,
                ))
                verifier_ids.add(check.verifier_id)
                if passed:
                    passed_ids.append(check_id)
                    conditions.update(check.verifies_conditions)
                else:
                    failed_ids.append(check_id)

            missing_conditions = sorted(set(capability.postconditions) - conditions)
            if missing_conditions:
                blockers.append(
                    f"{capability.capability_id} postconditions not independently verified: "
                    + ", ".join(missing_conditions)
                )
            capability_passed = not failed_ids and not missing_conditions
            capability_results.append(CapabilityVerification(
                capability.capability_id,
                capability_passed,
                tuple(passed_ids),
                tuple(failed_ids),
                tuple(sorted(conditions)),
                tuple(sorted(verifier_ids)),
            ))
            if capability_passed:
                all_conditions.update(conditions)

        # Ensure every step's capability has a corresponding passing verification.
        if len(capability_results) != len(execution.steps):
            blockers.append("one or more execution steps could not be verified")
        status = (
            VerificationStatus.VERIFIED
            if not blockers and all(item.passed for item in capability_results)
            else VerificationStatus.FAILED
        )
        return VerificationReport(
            execution.plan_id, execution.task_id, status, tuple(results),
            tuple(capability_results), tuple(sorted(all_conditions)),
            tuple(dict.fromkeys(blockers)), execution.fingerprint,
        )
