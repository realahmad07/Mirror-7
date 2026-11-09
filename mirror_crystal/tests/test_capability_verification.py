from mirror_crystal.blueprint_contracts import EvidenceLevel, VerificationEvidence
from mirror_crystal.capability_model import CapabilitySpec, PortContract
from mirror_crystal.execution import (
    ExecutionStatus, PlanExecutionReport, StepExecutionResult,
)
from mirror_crystal.universal_representation import RepresentationProvenance
from mirror_crystal.verification import (
    IndependentCheck, IndependentVerifierRegistry, IndependentVerificationEngine,
    VerificationStatus,
)


def positive_result(context):
    return context.step.outputs.get("answer", 0) > 0


def has_result(context):
    return "answer" in context.step.outputs


def false_check(context):
    return False


def cap():
    return CapabilitySpec(
        capability_id="math.solve",
        name="Solve",
        description="test",
        domains=("mathematics",),
        intents=("solve",),
        inputs=(PortContract("x", "number"),),
        outputs=(PortContract("answer", "number"),),
        postconditions=("answer_positive",),
        provenance=RepresentationProvenance(source="verification-tests"),
        verification=VerificationEvidence(
            level=EvidenceLevel.EXECUTION_TESTED,
            checks_passed=("executor-test",),
            verifier="executor",
        ),
    )


def execution(*, answer=42, succeeded=True):
    step = StepExecutionResult(
        "step-1", "subtask-1", "math.solve", succeeded,
        {"x": 2}, {"answer": answer} if succeeded else {},
        "" if succeeded else "operation failed",
    )
    return PlanExecutionReport(
        "plan-1", "task-1",
        ExecutionStatus.SUCCEEDED if succeeded else ExecutionStatus.FAILED,
        (step,), "" if succeeded else "operation failed",
    )


def registry():
    result = IndependentVerifierRegistry()
    result.register(IndependentCheck(
        "positive", "math-independent-checker",
        positive_result, ("answer_positive",),
    ), independent=True)
    return result


def test_independent_check_verifies_execution_postcondition():
    report = IndependentVerificationEngine(registry()).verify(
        execution(), (cap(),), required_checks={"math.solve": ("positive",)},
    )
    assert report.status is VerificationStatus.VERIFIED
    assert report.verified_conditions == ("answer_positive",)
    assert report.capabilities[0].passed
    assert report.verifier_ids == ("math-independent-checker",)


def test_missing_required_check_fails_closed():
    report = IndependentVerificationEngine(registry()).verify(
        execution(), (cap(),), required_checks={"math.solve": ("not-registered",)},
    )
    assert report.status is VerificationStatus.FAILED
    assert not report.verified
    assert report.blockers


def test_check_cannot_claim_undeclared_postcondition():
    checks = IndependentVerifierRegistry()
    checks.register(IndependentCheck(
        "bad-claim", "checker", has_result, ("not-a-declared-postcondition",),
    ), independent=True)
    report = IndependentVerificationEngine(checks).verify(
        execution(), (cap(),), required_checks={"math.solve": ("bad-claim",)},
    )
    assert report.status is VerificationStatus.FAILED
    assert "undeclared" in report.results[0].detail


def test_false_independent_check_fails_verification():
    checks = IndependentVerifierRegistry()
    checks.register(IndependentCheck(
        "false", "checker", false_check, ("answer_positive",),
    ), independent=True)
    report = IndependentVerificationEngine(checks).verify(
        execution(), (cap(),), required_checks={"math.solve": ("false",)},
    )
    assert report.status is VerificationStatus.FAILED
    assert report.capabilities[0].checks_failed == ("false",)


def test_executor_failure_cannot_be_verified():
    report = IndependentVerificationEngine(registry()).verify(
        execution(succeeded=False), (cap(),), required_checks={"math.solve": ("positive",)},
    )
    assert report.status is VerificationStatus.BLOCKED
    assert not report.results


def test_required_independent_identity_is_explicit():
    checks = IndependentVerifierRegistry()
    import pytest
    with pytest.raises(ValueError):
        checks.register(IndependentCheck("x", "checker", has_result), independent=False)


def test_execution_report_fingerprint_is_bound_into_verification():
    original = execution()
    report = IndependentVerificationEngine(registry()).verify(
        original, (cap(),), required_checks={"math.solve": ("positive",)},
    )
    altered = execution(answer=43)
    assert report.execution_fingerprint == original.fingerprint
    assert report.execution_fingerprint != altered.fingerprint
