from __future__ import annotations

import time

from mirror_crystal.blueprint_contracts import EvidenceLevel, VerificationEvidence
from mirror_crystal.capability_composition import (
    CompositionPlan, InputBinding, PlanStatus, PlanStep,
)
from mirror_crystal.capability_model import CapabilitySpec, PortContract
from mirror_crystal.execution import (
    CapabilityExecutionEngine, ExecutionPolicy, ExecutionStatus,
    NativeOperationRegistry,
)
from mirror_crystal.universal_representation import RepresentationProvenance


def _evidence():
    return VerificationEvidence(
        level=EvidenceLevel.EXECUTION_TESTED,
        checks_passed=("native-handler-contract",),
        verifier="test-suite",
    )


def add_numbers(inputs):
    return {"result": inputs["left"] + inputs["right"]}


def fail_operation(inputs):
    raise RuntimeError("intentional failure")


def echo_value(inputs):
    return {"value": inputs["value"]}


def slow_operation(inputs):
    time.sleep(5)
    return {"result": 1}


def wrong_output(inputs):
    return {"result": "not-a-number"}


def unknown_output(inputs):
    return {"result": 3, "undeclared": 4}


def cap(identifier, *, inputs=(), outputs=()):
    return CapabilitySpec(
        capability_id=identifier,
        name=identifier,
        description="execution test capability",
        domains=("general",),
        intents=("any",),
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        provenance=RepresentationProvenance(source="execution-tests"),
        verification=_evidence(),
    )


def one_step_plan(capability_id="add"):
    return CompositionPlan(
        "plan-add", "task-add", PlanStatus.READY,
        (PlanStep(
            "step-add", "task:task-add", capability_id,
            input_bindings=(
                InputBinding("left", "task_input", "left"),
                InputBinding("right", "task_input", "right"),
            ),
        ),),
    )


def test_registered_native_handler_executes_in_child_and_validates_outputs():
    capability = cap(
        "add",
        inputs=(PortContract("left", "number"), PortContract("right", "number")),
        outputs=(PortContract("result", "number"),),
    )
    registry = NativeOperationRegistry()
    registry.register("add", add_numbers)
    report = CapabilityExecutionEngine(registry).execute(
        one_step_plan(), (capability,), task_inputs={"left": 8, "right": 5},
    )
    assert report.status is ExecutionStatus.SUCCEEDED
    assert report.steps[0].outputs["result"] == 13
    assert report.steps[0].success


def test_missing_handler_fails_closed_without_execution():
    capability = cap(
        "add",
        inputs=(PortContract("left", "number"), PortContract("right", "number")),
        outputs=(PortContract("result", "number"),),
    )
    report = CapabilityExecutionEngine(NativeOperationRegistry()).execute(
        one_step_plan(), (capability,), task_inputs={"left": 1, "right": 2},
    )
    assert report.status is ExecutionStatus.BLOCKED
    assert "no native handler registered" in report.error


def test_unverified_capability_cannot_execute():
    capability = cap(
        "add",
        inputs=(PortContract("left", "number"), PortContract("right", "number")),
        outputs=(PortContract("result", "number"),),
    )
    unverified = CapabilitySpec(
        **{
            **capability.to_dict(),
            "inputs": capability.inputs,
            "outputs": capability.outputs,
            "provenance": capability.provenance,
            "verification": VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
        }
    )
    registry = NativeOperationRegistry()
    registry.register("add", add_numbers)
    report = CapabilityExecutionEngine(registry).execute(
        one_step_plan(), (unverified,), task_inputs={"left": 1, "right": 2},
    )
    assert report.status is ExecutionStatus.BLOCKED
    assert "not evidence-eligible" in report.error


def test_execution_stops_on_first_operation_failure():
    first = cap("first", outputs=(PortContract("value", "number"),))
    second = cap(
        "second", inputs=(PortContract("value", "number"),),
        outputs=(PortContract("value", "number"),),
    )
    plan = CompositionPlan(
        "plan-stop", "task-stop", PlanStatus.READY,
        (
            PlanStep("step-1", "subtask-1", "first"),
            PlanStep(
                "step-2", "subtask-2", "second", ("step-1",),
                (InputBinding("value", "step_output", "value", "step-1"),),
            ),
        ),
    )
    registry = NativeOperationRegistry()
    registry.register("first", fail_operation)
    registry.register("second", echo_value)
    report = CapabilityExecutionEngine(registry).execute(plan, (first, second))
    assert report.status is ExecutionStatus.FAILED
    assert len(report.steps) == 1
    assert "intentional failure" in report.error


def test_wrong_typed_output_and_undeclared_output_fail_the_step():
    capability = cap("bad", outputs=(PortContract("result", "number"),))
    for handler in (wrong_output, unknown_output):
        registry = NativeOperationRegistry()
        registry.register("bad", handler)
        report = CapabilityExecutionEngine(registry).execute(
            CompositionPlan(
                "plan-bad", "task-bad", PlanStatus.READY,
                (PlanStep("step-bad", "task:task-bad", "bad"),),
            ),
            (capability,),
        )
        assert report.status is ExecutionStatus.FAILED
        assert len(report.steps) == 1


def test_step_timeout_terminates_the_child_process():
    capability = cap("slow", outputs=(PortContract("result", "number"),))
    registry = NativeOperationRegistry()
    registry.register("slow", slow_operation)
    policy = ExecutionPolicy(timeout_seconds=2.0, max_cpu_seconds=3)
    report = CapabilityExecutionEngine(registry, policy).execute(
        CompositionPlan(
            "plan-slow", "task-slow", PlanStatus.READY,
            (PlanStep("step-slow", "task:task-slow", "slow"),),
        ),
        (capability,),
    )
    assert report.status is ExecutionStatus.FAILED
    assert "timeout" in report.error


def test_blocked_plan_is_never_run():
    capability = cap("add", outputs=(PortContract("result", "number"),))
    registry = NativeOperationRegistry()
    registry.register("add", add_numbers)
    blocked = CompositionPlan(
        "plan-blocked", "task-blocked", PlanStatus.BLOCKED, (),
        blockers=("missing inputs",),
    )
    report = CapabilityExecutionEngine(registry).execute(blocked, (capability,))
    assert report.status is ExecutionStatus.BLOCKED
    assert not report.steps


def test_input_json_limits_are_enforced_before_child_starts():
    capability = cap(
        "add",
        inputs=(PortContract("left", "string"),),
        outputs=(PortContract("result", "number"),),
    )
    registry = NativeOperationRegistry()
    registry.register("add", add_numbers)
    plan = CompositionPlan(
        "plan-limit", "task-limit", PlanStatus.READY,
        (PlanStep(
            "step-limit", "task:task-limit", "add",
            input_bindings=(InputBinding("left", "task_input", "left"),),
        ),),
    )
    report = CapabilityExecutionEngine(
        registry, ExecutionPolicy(max_input_bytes=8)
    ).execute(plan, (capability,), task_inputs={"left": "this is too long"})
    assert report.status is ExecutionStatus.FAILED
    assert "max_input_bytes" in report.error
