from __future__ import annotations

import pytest

from mirror_crystal.blueprint_contracts import EvidenceLevel, VerificationEvidence
from mirror_crystal.capability_model import CapabilitySpec, PortContract
from mirror_crystal.domain_verification import (
    DomainDecision,
    DomainEvidenceArtifact,
    DomainVerificationEngine,
    DomainVerificationRequirement,
    DomainVerificationStatus,
    DomainVerificationCheck,
    DomainVerifierRegistry,
)
from mirror_crystal.execution import (
    ExecutionStatus,
    PlanExecutionReport,
    StepExecutionResult,
)


def capability(domain: str, cap_id: str | None = None) -> CapabilitySpec:
    return CapabilitySpec(
        capability_id=cap_id or f"cap:{domain}",
        name=f"{domain} operation",
        description="Bounded test capability",
        domains=(domain,),
        intents=("evaluate",),
        inputs=(PortContract("value", "integer"),),
        outputs=(PortContract("result", "integer"),),
        postconditions=("result_is_valid",),
        verification=VerificationEvidence(
            level=EvidenceLevel.EXECUTION_TESTED,
            checks_passed=("host-operation-tests",),
            verifier="host-tests",
        ),
    )


def execution(cap: CapabilitySpec) -> PlanExecutionReport:
    step = StepExecutionResult(
        step_id="step-1",
        subtask_id="task-unit-1",
        capability_id=cap.capability_id,
        success=True,
        inputs={"value": 4},
        outputs={"result": 8},
    )
    return PlanExecutionReport(
        plan_id="plan-1",
        task_id="task-1",
        status=ExecutionStatus.SUCCEEDED,
        steps=(step,),
    )


def evidence(context, *, domain, kind, verifier, decision=DomainDecision.PASS,
             conditions=("result_is_valid",), evidence_id=None, fingerprint=None):
    return DomainEvidenceArtifact(
        evidence_id=evidence_id or f"{verifier}:{kind}:{context.step.step_id}",
        domain=domain,
        evidence_kind=kind,
        decision=decision,
        step_id=context.step.step_id,
        capability_id=context.capability.capability_id,
        execution_fingerprint=fingerprint or context.execution_fingerprint,
        verifier_id=verifier,
        evidence_ref=f"evidence://{verifier}/{kind}/case-1",
        evidence_payload={"observed": 8, "method": kind},
        verified_conditions=conditions,
        detail="" if decision is DomainDecision.PASS else f"{decision.value} by {verifier}",
    )


def run_policy(domain, kinds, *, callbacks=None, min_verifiers=2, cap=None):
    cap = cap or capability(domain)
    registry = DomainVerifierRegistry()
    checks = {}
    for index, kind in enumerate(kinds):
        verifier = f"verifier-{index + 1}"
        check_id = f"check-{kind}"

        def callback(context, domain=domain, kind=kind, verifier=verifier):
            if callbacks and kind in callbacks:
                return callbacks[kind](context, domain, kind, verifier)
            return evidence(context, domain=domain, kind=kind, verifier=verifier)

        registry.register(
            DomainVerificationCheck(check_id, verifier, domain, kind, callback),
            independent=True,
        )
        checks[kind] = check_id
    report = DomainVerificationEngine(registry).verify(
        execution(cap),
        (cap,),
        requirements={
            cap.capability_id: DomainVerificationRequirement(
                domain, tuple(kinds), min_verifiers,
            )
        },
        required_checks={cap.capability_id: checks},
    )
    return report


@pytest.mark.parametrize(
    ("domain", "kinds"),
    [
        ("programming", ("test_suite", "static_analysis")),
        ("mathematics", ("exact_computation", "formal_proof")),
        ("physics", ("dimensional_analysis", "simulation")),
        ("biology", ("source_corroboration", "experimental_observation")),
        ("engineering", ("simulation", "tolerance_analysis")),
    ],
)
def test_domain_policy_requires_typed_evidence_and_binds_every_step(domain, kinds):
    report = run_policy(domain, kinds)
    assert report.status is DomainVerificationStatus.VERIFIED
    assert report.verified
    assert report.capabilities[0].verified_conditions == ("result_is_valid",)
    assert len(report.verifier_ids) == 2
    assert all(item.evidence_fingerprint and len(item.payload_sha256) == 64 for item in report.results)


def test_boolean_callback_is_rejected_as_not_evidence():
    cap = capability("programming")
    registry = DomainVerifierRegistry()
    registry.register(
        DomainVerificationCheck("tests", "checker-a", "programming", "test_suite", lambda _ctx: True),
        independent=True,
    )
    registry.register(
        DomainVerificationCheck(
            "static", "checker-b", "programming", "static_analysis",
            lambda ctx: evidence(ctx, domain="programming", kind="static_analysis", verifier="checker-b"),
        ),
        independent=True,
    )
    report = DomainVerificationEngine(registry).verify(
        execution(cap), (cap,),
        requirements={cap.capability_id: DomainVerificationRequirement(
            "programming", ("test_suite", "static_analysis"), 2,
        )},
        required_checks={cap.capability_id: {"test_suite": "tests", "static_analysis": "static"}},
    )
    assert report.status is DomainVerificationStatus.FAILED
    assert any("DomainVerificationError" in item.detail for item in report.results)


def test_evidence_from_different_execution_cannot_be_replayed():
    cap = capability("mathematics")

    def stale(context):
        return evidence(
            context, domain="mathematics", kind="exact_computation", verifier="checker-a",
            fingerprint="0" * 64,
        )

    report = run_policy(
        "mathematics", ("exact_computation", "formal_proof"),
        callbacks={"exact_computation": stale}, cap=cap,
    )
    assert report.status is DomainVerificationStatus.FAILED
    assert any("execution_fingerprint" in item.detail for item in report.results)


def test_evidence_cannot_claim_undeclared_postcondition():
    cap = capability("physics")

    def forged_condition(context, domain, kind, verifier):
        return evidence(
            context, domain=domain, kind=kind, verifier=verifier,
            conditions=("invented_condition",),
        )

    report = run_policy(
        "physics", ("dimensional_analysis", "simulation"),
        callbacks={"dimensional_analysis": forged_condition}, cap=cap,
    )
    assert report.status is DomainVerificationStatus.FAILED
    assert any("undeclared postconditions" in item.detail for item in report.results)


def test_inconclusive_evidence_blocks_instead_of_passing():
    cap = capability("biology")

    def inconclusive(context, domain, kind, verifier):
        return evidence(
            context, domain=domain, kind=kind, verifier=verifier,
            decision=DomainDecision.INCONCLUSIVE, conditions=(),
        )

    report = run_policy(
        "biology", ("source_corroboration", "experimental_observation"),
        callbacks={"experimental_observation": inconclusive}, cap=cap,
    )
    assert report.status is DomainVerificationStatus.BLOCKED
    assert not report.verified
    assert any("inconclusive" in item for item in report.blockers)


def test_two_check_ids_from_one_verifier_do_not_fake_independence():
    cap = capability("engineering")
    registry = DomainVerifierRegistry()
    for kind, check_id in (("simulation", "sim"), ("tolerance_analysis", "tolerance")):
        registry.register(
            DomainVerificationCheck(
                check_id, "same-verifier", "engineering", kind,
                lambda ctx, kind=kind: evidence(
                    ctx, domain="engineering", kind=kind, verifier="same-verifier",
                ),
            ),
            independent=True,
        )
    report = DomainVerificationEngine(registry).verify(
        execution(cap), (cap,),
        requirements={cap.capability_id: DomainVerificationRequirement(
            "engineering", ("simulation", "tolerance_analysis"), 2,
        )},
        required_checks={cap.capability_id: {
            "simulation": "sim", "tolerance_analysis": "tolerance",
        }},
    )
    assert report.status is DomainVerificationStatus.BLOCKED
    assert any("distinct verifier identities" in item for item in report.blockers)


def test_missing_domain_check_fails_closed():
    cap = capability("mathematics")
    report = run_policy("mathematics", ("exact_computation", "formal_proof"), callbacks={
        "formal_proof": lambda context, domain, kind, verifier: evidence(
            context, domain=domain, kind=kind, verifier=verifier,
        )
    }, cap=cap)
    # run_policy registers both callbacks; explicitly exercise a requirement that
    # names a check which the registry does not contain.
    registry = DomainVerifierRegistry()
    registry.register(
        DomainVerificationCheck(
            "exact", "checker-a", "mathematics", "exact_computation",
            lambda ctx: evidence(ctx, domain="mathematics", kind="exact_computation", verifier="checker-a"),
        ),
        independent=True,
    )
    missing = DomainVerificationEngine(registry).verify(
        execution(cap), (cap,),
        requirements={cap.capability_id: DomainVerificationRequirement(
            "mathematics", ("exact_computation", "formal_proof"), 2,
        )},
        required_checks={cap.capability_id: {
            "exact_computation": "exact", "formal_proof": "absent",
        }},
    )
    assert missing.status is DomainVerificationStatus.FAILED
    assert any("no registered verifier" in item.detail for item in missing.results)


def test_artifact_requires_a_real_evidence_locator_and_payload():
    with pytest.raises(ValueError):
        DomainEvidenceArtifact(
            evidence_id="e", domain="math", evidence_kind="proof",
            decision=DomainDecision.PASS, step_id="s", capability_id="c",
            execution_fingerprint="f", verifier_id="v", evidence_ref="",
            evidence_payload={"proof": "x"}, verified_conditions=("p",),
        )
    with pytest.raises(ValueError):
        DomainEvidenceArtifact(
            evidence_id="e", domain="math", evidence_kind="proof",
            decision=DomainDecision.PASS, step_id="s", capability_id="c",
            execution_fingerprint="f", verifier_id="v", evidence_ref="evidence://proof/1",
            evidence_payload={}, verified_conditions=("p",),
        )


def test_artifact_payload_digest_changes_when_payload_changes():
    cap = capability("mathematics")
    context = type("Context", (), {
        "step": execution(cap).steps[0],
        "capability": cap,
        "execution_fingerprint": execution(cap).fingerprint,
    })()
    first = evidence(context, domain="mathematics", kind="exact_computation", verifier="checker")
    changed = DomainEvidenceArtifact(
        evidence_id=first.evidence_id, domain="mathematics",
        evidence_kind="exact_computation", decision=DomainDecision.PASS,
        step_id=first.step_id, capability_id=first.capability_id,
        execution_fingerprint=first.execution_fingerprint, verifier_id=first.verifier_id,
        evidence_ref=first.evidence_ref, evidence_payload={"observed": 9},
        verified_conditions=first.verified_conditions,
    )
    assert first.payload_sha256 != changed.payload_sha256
    assert first.fingerprint != changed.fingerprint



def test_unverified_capability_cannot_be_promoted_by_domain_evidence():
    cap = CapabilitySpec(
        capability_id="cap:unverified",
        name="Unverified operation",
        description="Must remain blocked",
        domains=("mathematics",),
        intents=("evaluate",),
        inputs=(PortContract("value", "integer"),),
        outputs=(PortContract("result", "integer"),),
        postconditions=("result_is_valid",),
        verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
    )
    report = run_policy(
        "mathematics", ("exact_computation", "formal_proof"), cap=cap,
    )
    assert report.status is DomainVerificationStatus.BLOCKED
    assert not report.verified
    assert any("not evidence-eligible" in item for item in report.blockers)
