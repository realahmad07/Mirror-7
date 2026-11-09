from mirror_crystal.blueprint_contracts import EvidenceLevel, VerificationEvidence
from mirror_crystal.capability_composition import (
    CapabilityCompositionPlanner, PlanStatus,
)
from mirror_crystal.capability_matching import CapabilityMatcher
from mirror_crystal.capability_model import CapabilitySpec, PortContract
from mirror_crystal.task_understanding import TaskIntent, TaskSpec, TaskSubtask
from mirror_crystal.universal_representation import RepresentationProvenance


def verified():
    return VerificationEvidence(
        level=EvidenceLevel.EXECUTION_TESTED,
        checks_passed=("isolated-contract-test",),
        verifier="pytest",
    )


def capability(identifier, intent, *, inputs=(), outputs=(), pre=(), post=(), deps=(), evidence=None):
    return CapabilitySpec(
        capability_id=identifier,
        name=identifier,
        description="composition fixture",
        domains=("mathematics",),
        intents=(intent,),
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        preconditions=tuple(pre),
        postconditions=tuple(post),
        dependencies=tuple(deps),
        provenance=RepresentationProvenance(source="composition-test"),
        verification=evidence or verified(),
    )


def two_step_task(*, known_inputs=None):
    return TaskSpec(
        task_id="compose-1",
        request="Build a value then verify the value",
        goal="Build then verify",
        intent=TaskIntent.BUILD,
        domain="mathematics",
        known_inputs=known_inputs or {},
        subtasks=(
            TaskSubtask("make", "Build a value", TaskIntent.BUILD, 0),
            TaskSubtask("verify", "Verify the value", TaskIntent.VERIFY, 1, ("make",)),
        ),
    )


def test_planner_builds_dependency_order_and_typed_output_binding():
    producer = capability(
        "producer", "build",
        outputs=(PortContract("value", "integer", unit="m"),),
        post=("value_created",),
    )
    consumer = capability(
        "consumer", "verify",
        inputs=(PortContract("value", "number", unit="m"),),
        outputs=(PortContract("result", "boolean"),),
        pre=("value_created",),
    )
    task = two_step_task()
    report = CapabilityMatcher().match(task, (producer, consumer))
    plan = CapabilityCompositionPlanner().compose(
        task, (producer, consumer), match_report=report,
        verified_conditions=("value_created",),
    )
    assert plan.status is PlanStatus.READY
    assert [step.capability_id for step in plan.steps] == ["producer", "consumer"]
    binding = plan.steps[1].input_bindings[0]
    assert binding.source == "step_output"
    assert binding.input_name == "value"
    assert binding.source_step_id == plan.steps[0].step_id
    assert "producer:value_created" in plan.verification_requirements


def test_unverified_capabilities_cannot_enter_a_plan():
    unverified_build = capability(
        "unverified-build", "build",
        outputs=(PortContract("value", "number"),),
        evidence=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
    )
    unverified_verify = capability(
        "unverified-verify", "verify",
        inputs=(PortContract("value", "number"),),
        evidence=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
    )
    task = two_step_task()
    plan = CapabilityCompositionPlanner().compose(task, (unverified_build, unverified_verify))
    assert plan.status is PlanStatus.BLOCKED
    assert not plan.steps
    assert any("no evidence-eligible" in item for item in plan.blockers)


def test_unbound_required_input_blocks_plan():
    producer = capability("producer", "build", outputs=(PortContract("other", "number"),))
    consumer = capability("consumer", "verify", inputs=(PortContract("value", "number"),))
    task = two_step_task()
    plan = CapabilityCompositionPlanner().compose(task, (producer, consumer))
    assert plan.status is PlanStatus.BLOCKED
    assert any("required input is unbound: value" in item for item in plan.blockers)


def test_preconditions_require_explicit_verified_conditions():
    producer = capability("producer", "build", outputs=(PortContract("value", "number"),))
    consumer = capability(
        "consumer", "verify",
        inputs=(PortContract("value", "number"),),
        pre=("independent_measurement",),
    )
    task = two_step_task()
    plan = CapabilityCompositionPlanner().compose(task, (producer, consumer))
    assert plan.status is PlanStatus.BLOCKED
    assert any("requires verified precondition" in item for item in plan.blockers)


def test_wrong_units_do_not_connect_outputs_to_inputs():
    producer = capability("producer", "build", outputs=(PortContract("value", "number", unit="cm"),))
    consumer = capability("consumer", "verify", inputs=(PortContract("value", "number", unit="m"),))
    task = two_step_task()
    plan = CapabilityCompositionPlanner().compose(task, (producer, consumer))
    assert plan.status is PlanStatus.BLOCKED
    assert any("required input is unbound" in item for item in plan.blockers)


def test_planner_rejects_report_for_another_task():
    task = two_step_task()
    other = TaskSpec(
        task_id="other", request="Build a value", goal="Build",
        intent=TaskIntent.BUILD, domain="mathematics",
    )
    report = CapabilityMatcher().match(other, ())
    import pytest
    with pytest.raises(ValueError):
        CapabilityCompositionPlanner().compose(task, (), match_report=report)
