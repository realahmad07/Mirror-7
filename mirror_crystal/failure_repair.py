"""Failure diagnosis and evidence-gated targeted repair (Strategic Phase 7).

The module explains failures from composition, synthesis, execution and domain
verification reports. Its repair path uses explicit host-oracle counterexamples
to run bounded UOL CEGIS. A repair is only a proposal: it never executes
unverified capabilities, changes registries, promotes evidence or persists data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Callable, Iterable, Mapping

from .blueprint_contracts import EvidenceLevel
from .capability_composition import CompositionPlan, PlanStatus
from .capability_model import CapabilitySpec
from .capability_synthesizer import (
    CapabilitySynthesizer,
    Counterexample,
    SynthesisGoal,
    SynthesisReport,
    SynthesisStatus,
)
from .domain_verification import (
    DomainDecision,
    DomainVerificationReport,
    DomainVerificationStatus,
)
from .execution import ExecutionStatus, PlanExecutionReport
from .operational_language import Program, UOL, UOLError, _freeze, _matches
from .universal_representation import canonical_json


class FailureRepairError(ValueError):
    """Raised when diagnosis or a targeted repair request is malformed."""


class FailureKind(str, Enum):
    NONE = "none"
    COMPOSITION_BLOCKED = "composition_blocked"
    SYNTHESIS_NO_SOLUTION = "synthesis_no_solution"
    SYNTHESIS_BUDGET_EXHAUSTED = "synthesis_budget_exhausted"
    SYNTHESIS_COUNTEREXAMPLE_LIMIT = "synthesis_counterexample_limit"
    EXECUTION_FAILED = "execution_failed"
    EXECUTION_BLOCKED = "execution_blocked"
    DOMAIN_EVIDENCE_REJECTED = "domain_evidence_rejected"
    DOMAIN_VERIFICATION_BLOCKED = "domain_verification_blocked"
    PROGRAM_COUNTEREXAMPLE = "program_counterexample"


class RepairStatus(str, Enum):
    PROPOSED = "proposed"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class FailureDiagnosis:
    """A deterministic diagnosis tied to a specific source report."""

    source: str
    source_id: str
    kind: FailureKind
    message: str
    recommended_action: str
    step_id: str = ""
    capability_id: str = ""
    evidence_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("source", "source_id", "message", "recommended_action"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise FailureRepairError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        object.__setattr__(self, "kind", FailureKind(self.kind))
        object.__setattr__(
            self, "evidence_kinds",
            tuple(sorted({str(item).strip() for item in self.evidence_kinds if str(item).strip()})),
        )


@dataclass(frozen=True)
class RepairProposal:
    """A revised program proposal that still requires independent verification."""

    status: RepairStatus
    diagnosis: FailureDiagnosis
    original_program_fingerprint: str
    synthesis_report: SynthesisReport
    counterexamples: tuple[Counterexample, ...]
    program: Program | None = None
    capability_candidate: CapabilitySpec | None = None
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", RepairStatus(self.status))
        object.__setattr__(self, "counterexamples", tuple(self.counterexamples))
        object.__setattr__(self, "blockers", tuple(str(item) for item in self.blockers))
        if self.status is RepairStatus.PROPOSED:
            if self.program is None or self.capability_candidate is None:
                raise FailureRepairError("a proposed repair requires a program and capability candidate")
            if self.capability_candidate.executable or (
                self.capability_candidate.verification.level is not EvidenceLevel.UNVERIFIED
            ):
                raise FailureRepairError("a repair candidate must remain UNVERIFIED")
        elif self.program is not None or self.capability_candidate is not None:
            raise FailureRepairError("a blocked repair cannot expose an executable candidate")

    @property
    def ready_for_independent_verification(self) -> bool:
        return self.status is RepairStatus.PROPOSED

    @property
    def requires_explicit_promotion(self) -> bool:
        return True


class FailureRepairEngine:
    """Diagnose the smallest evidenced failure and propose bounded UOL repairs."""

    VERSION = "failure-diagnosis-targeted-repair-v1"

    def __init__(
        self,
        language: UOL | None = None,
        synthesizer: CapabilitySynthesizer | None = None,
    ) -> None:
        if synthesizer is not None and not isinstance(synthesizer, CapabilitySynthesizer):
            raise TypeError("synthesizer must be CapabilitySynthesizer")
        if synthesizer is not None and language is not None and synthesizer.language is not language:
            raise FailureRepairError("language must be the same instance used by the synthesizer")
        self.language = language or (synthesizer.language if synthesizer is not None else UOL())
        self.synthesizer = synthesizer or CapabilitySynthesizer(self.language)

    def diagnose(
        self,
        report: CompositionPlan | SynthesisReport | PlanExecutionReport | DomainVerificationReport,
    ) -> FailureDiagnosis:
        """Diagnose a known report; never infer success from missing evidence."""
        if isinstance(report, CompositionPlan):
            if report.status is PlanStatus.READY:
                return FailureDiagnosis(
                    "composition", report.plan_id, FailureKind.NONE,
                    "composition plan is ready",
                    "execute only through the registered, bounded execution engine",
                )
            message = "; ".join(report.blockers) or "composition plan is blocked"
            return FailureDiagnosis(
                "composition", report.plan_id, FailureKind.COMPOSITION_BLOCKED,
                message,
                "resolve the listed missing capability, dependency, input binding or verified precondition; do not bypass the planner",
            )

        if isinstance(report, SynthesisReport):
            if report.status is SynthesisStatus.SYNTHESIZED and report.program is not None:
                return FailureDiagnosis(
                    "synthesis", report.program.program_id, FailureKind.NONE,
                    "bounded synthesis passed its supplied training and held-out examples",
                    "obtain independent domain verification before treating the candidate as trusted",
                )
            if report.status is SynthesisStatus.BUDGET_EXHAUSTED:
                kind = FailureKind.SYNTHESIS_BUDGET_EXHAUSTED
                action = "review the failed search space, then selectively increase bounded search depth or candidate budget and retain held-out tests"
            elif report.status is SynthesisStatus.COUNTEREXAMPLE_LIMIT:
                kind = FailureKind.SYNTHESIS_COUNTEREXAMPLE_LIMIT
                action = "supply a fresh disjoint counterexample pool and a stable host oracle; do not reuse held-out inputs"
            else:
                kind = FailureKind.SYNTHESIS_NO_SOLUTION
                action = "inspect the typed input/output contract and available verified primitives; add only justified rules or examples"
            return FailureDiagnosis(
                "synthesis", "synthesis-report", kind,
                "; ".join(report.blockers) or f"synthesis ended with {report.status.value}",
                action,
            )

        if isinstance(report, PlanExecutionReport):
            failed = next((item for item in report.steps if not item.success), None)
            if report.status is ExecutionStatus.SUCCEEDED:
                return FailureDiagnosis(
                    "execution", report.plan_id, FailureKind.NONE,
                    "all planned steps executed successfully",
                    "continue to independent outcome verification; execution alone is not proof of correctness",
                )
            if report.status is ExecutionStatus.FAILED:
                detail = failed.error if failed is not None else report.error
                return FailureDiagnosis(
                    "execution", report.plan_id, FailureKind.EXECUTION_FAILED,
                    detail or report.error or "execution failed without a step detail",
                    "repair only the failing handler or its declared contract; collect fresh tests and verification evidence before replay; do not retry side effects automatically",
                    step_id=failed.step_id if failed is not None else "",
                    capability_id=failed.capability_id if failed is not None else "",
                )
            return FailureDiagnosis(
                "execution", report.plan_id, FailureKind.EXECUTION_BLOCKED,
                report.error or "execution was blocked",
                "resolve the exact policy, evidence-eligibility, binding or handler-registration blocker before executing",
                step_id=failed.step_id if failed is not None else "",
                capability_id=failed.capability_id if failed is not None else "",
            )

        if isinstance(report, DomainVerificationReport):
            rejected = tuple(
                item for item in report.results if item.decision is DomainDecision.FAIL
            )
            inconclusive = tuple(
                item for item in report.results if item.decision is DomainDecision.INCONCLUSIVE
            )
            evidence_kinds = tuple(item.check_id for item in rejected + inconclusive)
            if report.status is DomainVerificationStatus.VERIFIED:
                return FailureDiagnosis(
                    "domain_verification", report.plan_id, FailureKind.NONE,
                    "required domain evidence passed and is bound to this execution",
                    "retain the report and continue only through the explicit learning/promotion gates",
                )
            if rejected:
                details = [
                    f"{item.check_id} ({item.verifier_id or 'unidentified verifier'}): {item.detail or item.decision.value}"
                    for item in rejected
                ]
                return FailureDiagnosis(
                    "domain_verification", report.plan_id, FailureKind.DOMAIN_EVIDENCE_REJECTED,
                    "; ".join(details),
                    "inspect the rejecting evidence and underlying artifact; correct the capability or verifier and obtain fresh execution-bound evidence; never lower the verification threshold to force a pass",
                    step_id=rejected[0].step_id,
                    capability_id=rejected[0].capability_id,
                    evidence_kinds=evidence_kinds,
                )
            blockers = "; ".join(report.blockers) or "; ".join(
                item.detail for item in inconclusive if item.detail
            ) or "required independent evidence is incomplete"
            return FailureDiagnosis(
                "domain_verification", report.plan_id, FailureKind.DOMAIN_VERIFICATION_BLOCKED,
                blockers,
                "restore missing independent verifier coverage or resolve inconclusive evidence, then rerun verification against a fresh execution fingerprint",
                step_id=inconclusive[0].step_id if inconclusive else "",
                capability_id=inconclusive[0].capability_id if inconclusive else "",
                evidence_kinds=evidence_kinds,
            )

        raise TypeError(
            "report must be CompositionPlan, SynthesisReport, PlanExecutionReport or DomainVerificationReport"
        )

    @staticmethod
    def _equal(left: Any, right: Any) -> bool:
        if isinstance(left, bool) or isinstance(right, bool):
            return type(left) is type(right) and left == right
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
        return left == right

    def propose_program_repair(
        self,
        goal: SynthesisGoal,
        original_program: Program,
        *,
        counterexample_inputs: Iterable[Mapping[str, Any]],
        counterexample_oracle: Callable[[Mapping[str, Any]], Any],
        max_rounds: int = 8,
        counterexample_budget: int = 32,
    ) -> RepairProposal:
        """Use fresh host-oracle counterexamples to propose a bounded revised UOL program.

        The existing program must demonstrably fail at least one fresh probe.
        All probes are checked for schema, type, duplication and train/validation
        leakage before the oracle is called. A successful candidate is replayed
        over every original train/held-out case and every counterexample probe.
        """
        if not isinstance(goal, SynthesisGoal):
            raise TypeError("goal must be SynthesisGoal")
        if not isinstance(original_program, Program):
            raise TypeError("original_program must be Program")
        if not callable(counterexample_oracle):
            raise FailureRepairError("an explicit host-supplied counterexample oracle is required")
        if original_program.domain != goal.domain or original_program.inputs != goal.inputs:
            raise FailureRepairError("original program domain and typed inputs must match the repair goal")
        try:
            self.language.validate(original_program)
        except UOLError as exc:
            raise FailureRepairError(f"original program is invalid: {exc}") from exc

        probes = tuple(dict(item) for item in counterexample_inputs)
        if not probes:
            raise FailureRepairError("at least one fresh counterexample probe is required")
        if len(probes) > 1000:
            raise FailureRepairError("counterexample probe pool exceeds 1000 inputs")
        required_names = {item.name for item in goal.inputs}
        used = {
            canonical_json(dict(item.inputs))
            for item in goal.training + goal.validation
        }
        seen: set[str] = set()
        for probe in probes:
            if set(probe) != required_names:
                raise FailureRepairError("counterexample probe inputs must match the declared input schema")
            key = canonical_json(probe)
            if key in used or key in seen:
                raise FailureRepairError("counterexample probes must be unique and disjoint from train/validation")
            seen.add(key)
            for port in goal.inputs:
                try:
                    value = _freeze(probe[port.name])
                    if not _matches(value, port.value_type):
                        raise FailureRepairError(f"counterexample input {port.name!r} has the wrong type")
                except (TypeError, ValueError) as exc:
                    raise FailureRepairError(f"invalid counterexample input {port.name!r}: {exc}") from exc

        expected_by_key: dict[str, Any] = {}
        failures: list[Counterexample] = []
        for probe in probes:
            key = canonical_json(probe)
            try:
                expected = _freeze(counterexample_oracle(dict(probe)))
                if not _matches(expected, goal.output_type):
                    raise FailureRepairError("counterexample oracle returned a value with the wrong output type")
            except FailureRepairError:
                raise
            except Exception as exc:
                raise FailureRepairError(f"counterexample oracle failed: {type(exc).__name__}: {exc}") from exc
            expected_by_key[key] = expected
            try:
                actual = self.language.execute(original_program, probe)
            except (UOLError, RuntimeError, TypeError, ValueError, OverflowError) as exc:
                failures.append(Counterexample(
                    dict(probe), expected, None, 0,
                    f"original program execution failed: {type(exc).__name__}: {exc}",
                ))
                continue
            if not self._equal(actual, expected):
                failures.append(Counterexample(
                    dict(probe), expected, actual, 0,
                    "original program output disagreed with the supplied oracle",
                ))

        if not failures:
            raise FailureRepairError("original program passed every supplied probe; no evidenced repair is justified")

        diagnosis = FailureDiagnosis(
            "program", original_program.program_id, FailureKind.PROGRAM_COUNTEREXAMPLE,
            f"original program failed {len(failures)} of {len(probes)} fresh oracle probes",
            "use the captured counterexamples to synthesize a bounded alternative; keep the result unverified until independent domain checks pass",
        )

        def cached_oracle(values: Mapping[str, Any]) -> Any:
            key = canonical_json(dict(values))
            if key not in expected_by_key:
                raise FailureRepairError("synthesizer requested an input outside the approved probe pool")
            return expected_by_key[key]

        report = self.synthesizer.synthesize(
            goal,
            counterexample_inputs=probes,
            counterexample_oracle=cached_oracle,
            max_rounds=max_rounds,
            counterexample_budget=counterexample_budget,
        )
        if not report.succeeded or report.program is None:
            return RepairProposal(
                RepairStatus.BLOCKED, diagnosis, original_program.fingerprint,
                report, tuple(failures), blockers=report.blockers or (
                    f"bounded repair ended with {report.status.value}",
                ),
            )
        if report.program.fingerprint == original_program.fingerprint:
            raise FailureRepairError("repair search returned the same program despite a recorded counterexample")

        # Defense in depth: replay all original acceptance examples and the fresh
        # host-oracle probes before exposing a candidate.
        replay_cases = tuple(goal.training) + tuple(goal.validation)
        for case in replay_cases:
            try:
                actual = self.language.execute(report.program, case.inputs)
            except Exception as exc:
                raise FailureRepairError(f"repaired program failed original replay: {type(exc).__name__}: {exc}") from exc
            if not self._equal(actual, case.expected):
                raise FailureRepairError("repaired program failed an original training/held-out case")
        for probe in probes:
            actual = self.language.execute(report.program, probe)
            expected = expected_by_key[canonical_json(probe)]
            if not self._equal(actual, expected):
                raise FailureRepairError("repaired program failed a fresh counterexample replay")

        candidate = self.synthesizer.to_candidate_capability(
            report, intent=goal.name,
        )
        if candidate.verification.level is not EvidenceLevel.UNVERIFIED or candidate.executable:
            raise FailureRepairError("repair candidate crossed the independent verification boundary")

        return RepairProposal(
            RepairStatus.PROPOSED, diagnosis, original_program.fingerprint,
            report, tuple(failures), program=report.program,
            capability_candidate=candidate,
        )
