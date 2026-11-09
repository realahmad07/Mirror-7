import pytest

from mirror_crystal.blueprint_contracts import EvidenceLevel, VerificationEvidence
from mirror_crystal.capability_composition import (
    CompositionPlan, InputBinding, PlanStatus, PlanStep,
)
from mirror_crystal.capability_learning import (
    CapabilityLearningEngine, CapabilityLearningError, CapabilityLearningObservation,
)
from mirror_crystal.capability_model import CapabilitySpec, PortContract
from mirror_crystal.execution import (
    ExecutionStatus, PlanExecutionReport, StepExecutionResult,
)
from mirror_crystal.task_understanding import TaskIntent, TaskSpec
from mirror_crystal.universal_representation import RepresentationProvenance
from mirror_crystal.verification import (
    CapabilityVerification, CheckResult, VerificationReport, VerificationStatus,
)


def capability(identifier, intent, *, inputs=(), outputs=(), post=()):
    return CapabilitySpec(
        capability_id=identifier,
        name=identifier,
        description="learning fixture",
        domains=("mathematics",),
        intents=(intent,),
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        postconditions=tuple(post),
        provenance=RepresentationProvenance(source="learning-tests"),
        verification=VerificationEvidence(
            level=EvidenceLevel.EXECUTION_TESTED,
            checks_passed=("source-capability-tests",),
            verifier="source-tests",
        ),
    )


def source_capabilities():
    return (
        capability(
            "produce", "build",
            inputs=(PortContract("amount", "number"),),
            outputs=(PortContract("temporary", "number"),),
        ),
        capability(
            "check", "verify",
            inputs=(PortContract("temporary", "number"),),
            outputs=(PortContract("result", "number"),),
            post=("result_verified",),
        ),
    )


def observation(index, *, verified=True, alternate_structure=False):
    task = TaskSpec(
        task_id=f"task-{index}",
        request=f"Build and independently verify amount {index}",
        goal="Build and verify an amount",
        intent=TaskIntent.BUILD,
        domain="mathematics",
        known_inputs={"amount": index + 1},
    )
    first_id, second_id = f"step-a-{index}", f"step-b-{index}"
    plan = CompositionPlan(
        f"plan-{index}", task.task_id, PlanStatus.READY,
        (
            PlanStep(
                first_id, "producer-unit", "produce",
                input_bindings=(InputBinding("amount", "task_input", "amount"),),
            ),
            PlanStep(
                second_id, "verifier-unit", "check",
                dependency_step_ids=() if alternate_structure else (first_id,),
                input_bindings=(
                    InputBinding("temporary", "step_output", "temporary", first_id),
                ),
                expected_postconditions=("result_verified",),
            ),
        ),
        verification_requirements=("result_verified",),
    )
    execution = PlanExecutionReport(
        plan.plan_id, task.task_id, ExecutionStatus.SUCCEEDED,
        (
            StepExecutionResult(
                first_id, "producer-unit", "produce", True,
                {"amount": index + 1}, {"temporary": (index + 1) * 2},
            ),
            StepExecutionResult(
                second_id, "verifier-unit", "check", True,
                {"temporary": (index + 1) * 2}, {"result": (index + 1) * 4},
            ),
        ),
    )
    check_results = []
    cap_results = []
    for step_id, capability_id in ((first_id, "produce"), (second_id, "check")):
        check_results.extend((
            CheckResult(step_id, capability_id, "independent-shape", "checker-alpha", verified),
            CheckResult(
                step_id, capability_id, "independent-semantics", "checker-beta",
                verified, ("result_verified",) if capability_id == "check" and verified else (),
            ),
        ))
        cap_results.append(CapabilityVerification(
            capability_id, verified,
            ("independent-shape", "independent-semantics") if verified else (),
            () if verified else ("independent-shape",),
            ("result_verified",) if capability_id == "check" and verified else (),
            ("checker-alpha", "checker-beta"),
        ))
    verification = VerificationReport(
        plan.plan_id, task.task_id,
        VerificationStatus.VERIFIED if verified else VerificationStatus.FAILED,
        tuple(check_results), tuple(cap_results),
        ("result_verified",) if verified else (),
        () if verified else ("independent check failed",),
        execution.fingerprint,
    )
    return CapabilityLearningObservation(task, plan, execution, verification)


def test_three_distinct_verified_runs_learn_a_reusable_composite():
    observations = tuple(observation(index) for index in range(3))
    proposal = CapabilityLearningEngine().propose(observations, source_capabilities())
    learned = proposal.capability
    assert proposal.requires_explicit_promotion
    assert learned.capability_id.startswith("learned:")
    assert learned.executable
    assert learned.verification.level is EvidenceLevel.INDEPENDENTLY_REPRODUCED
    assert proposal.observation_task_ids == ("task-0", "task-1", "task-2")
    assert proposal.verifier_ids == ("checker-alpha", "checker-beta")
    assert [item.name for item in learned.inputs] == ["amount"]
    assert [item.name for item in learned.outputs] == ["result"]
    assert learned.postconditions == ("result_verified",)


def test_learning_requires_minimum_distinct_successful_observations():
    observations = (observation(0), observation(1))
    with pytest.raises(CapabilityLearningError, match="at least 3"):
        CapabilityLearningEngine().propose(observations, source_capabilities())


def test_failed_or_unverified_observation_never_teaches_a_capability():
    observations = (observation(0), observation(1), observation(2, verified=False))
    with pytest.raises(CapabilityLearningError, match="independently verified"):
        CapabilityLearningEngine().propose(observations, source_capabilities())


def test_different_composition_structures_do_not_merge():
    observations = (
        observation(0), observation(1),
        observation(2, alternate_structure=True),
    )
    with pytest.raises(CapabilityLearningError, match="same typed composition structure"):
        CapabilityLearningEngine().propose(observations, source_capabilities())


def test_insufficient_verifier_diversity_blocks_learning():
    samples = []
    for index in range(3):
        item = observation(index)
        checks = tuple(
            CheckResult(
                result.step_id, result.capability_id, result.check_id,
                "only-one-verifier", result.passed, result.verified_conditions,
            )
            for result in item.verification.results
        )
        caps = tuple(
            CapabilityVerification(
                result.capability_id, result.passed, result.checks_passed,
                result.checks_failed, result.verified_conditions,
                ("only-one-verifier",),
            )
            for result in item.verification.capabilities
        )
        report = VerificationReport(
            item.plan.plan_id, item.task.task_id, VerificationStatus.VERIFIED,
            checks, caps, item.verification.verified_conditions, (),
            item.execution.fingerprint,
        )
        samples.append(CapabilityLearningObservation(
            item.task, item.plan, item.execution, report,
        ))
    with pytest.raises(CapabilityLearningError, match="distinct independent verifier"):
        CapabilityLearningEngine().propose(samples, source_capabilities())


def test_durable_promotion_is_explicit_and_uses_existing_capability_vault_api():
    proposal = CapabilityLearningEngine().propose(
        tuple(observation(index) for index in range(3)),
        source_capabilities(),
    )

    class Vault:
        def __init__(self):
            self.stored = None

        def store_capability(self, capability):
            self.stored = capability
            return capability.capability_id

    vault = Vault()
    result = CapabilityLearningEngine.persist(vault, proposal)
    assert result == proposal.capability.capability_id
    assert vault.stored == proposal.capability
