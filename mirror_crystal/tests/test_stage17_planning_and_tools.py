"""Stage 17 Test Suite: Planning & Tool Intelligence Subsystem.

Tests:
1. Tool definition, schema/parameter validation, and execution.
2. Hierarchical task DAG acyclicity validation and cycle rejection.
3. Dependency-ordered plan execution.
4. Automated failure recovery via retries.
5. Automated tool fallback routing when primary tool is exhausted.
6. Result verification contracts ensuring non-empty, valid outputs.
7. Downstream dependency skip propagation on unrecoverable failures.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
import pytest

from mirror_crystal.planning_and_tools import (
    ExecutionPlan,
    FailureRecoveryManager,
    PlanningAndToolEngine,
    PlanningError,
    ResultVerificationContract,
    StepStatus,
    TaskDecomposer,
    TaskNode,
    ToolDefinition,
    ToolRegistry,
    ToolRiskProfile,
    ToolExecutionMode,
)



def _test_adder(a: int, b: int) -> int:
    return a + b


def _append_order(log_path: str, step: str) -> None:
    path = Path(log_path)
    values = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    values.append(step)
    path.write_text(json.dumps(values), encoding="utf-8")


def _test_step_one(log_path: str, msg: str) -> str:
    _append_order(log_path, "step1")
    return f"Step 1 Done: {msg}"


def _test_step_two(log_path: str, prev: str) -> str:
    _append_order(log_path, "step2")
    return f"Step 2 Done with {prev}"


def _test_flaky_tool(state_path: str) -> str:
    with sqlite3.connect(state_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS attempts (n INTEGER NOT NULL)")
        row = conn.execute("SELECT n FROM attempts LIMIT 1").fetchone()
        count = 1 if row is None else int(row[0]) + 1
        conn.execute("DELETE FROM attempts")
        conn.execute("INSERT INTO attempts(n) VALUES (?)", (count,))
        conn.commit()
    if count < 2:
        raise ConnectionError("Transient network timeout")
    return "Success on retry"


def _test_primary_failure() -> str:
    raise RuntimeError("Primary service unavailable")


def _test_fallback_success() -> str:
    return "Output from fallback service"


def _test_fatal_failure() -> str:
    raise PermissionError("Fatal authentication error")


def _test_downstream() -> str:
    return "Should not run"


def _test_slow_handler() -> str:
    time.sleep(3.0)
    return "late result"


def _test_current_pid() -> int:
    return os.getpid()


def _test_large_output() -> str:
    return "x" * 10000


def _test_mutating_value(resource: str, approved: bool = False) -> dict:
    return {"updated": resource, "approved": approved}


def test_tool_definition_and_parameter_validation():
    """Verify tool parameter contracts enforce required keys."""
    tool = ToolDefinition(
        name="adder",
        description="Adds two integers",
        risk_profile=ToolRiskProfile.READ_ONLY,
        required_parameters=["a", "b"],
        handler=_test_adder,
    )
    assert tool.execute(a=10, b=25) == 35

    # Missing required parameter must fail
    with pytest.raises(PlanningError, match="missing required parameter 'b'"):
        tool.execute(a=10)


def test_dag_acyclicity_and_cycle_rejection():
    """Verify cyclic dependencies in task planning are detected and rejected."""
    # 1. Valid acyclic DAG: t1 -> t2 -> t3
    valid_subtasks = [
        {"task_id": "t1", "description": "Fetch data"},
        {"task_id": "t2", "description": "Process data", "dependencies": ["t1"]},
        {"task_id": "t3", "description": "Save report", "dependencies": ["t2"]},
    ]
    plan = TaskDecomposer.create_plan_from_subtasks("Pipeline Goal", valid_subtasks)
    assert len(plan.tasks) == 3

    # 2. Cyclic dependency: c1 -> c2 -> c1
    cyclic_subtasks = [
        {"task_id": "c1", "description": "Step 1", "dependencies": ["c2"]},
        {"task_id": "c2", "description": "Step 2", "dependencies": ["c1"]},
    ]
    with pytest.raises(PlanningError, match="Cyclic dependency detected"):
        TaskDecomposer.create_plan_from_subtasks("Cyclic Goal", cyclic_subtasks)


def test_dependency_ordered_plan_execution(tmp_path):
    """Verify tasks execute strictly in dependency order."""
    registry = ToolRegistry()
    log_path = str(tmp_path / "execution-order.json")
    registry.register(ToolDefinition(name="s1", description="s1", handler=_test_step_one))
    registry.register(ToolDefinition(name="s2", description="s2", handler=_test_step_two))

    plan = TaskDecomposer.create_plan_from_subtasks(
        goal="Order test",
        subtasks_data=[
            {"task_id": "task_b", "assigned_tool": "s2", "tool_parameters": {"log_path": log_path, "prev": "data"}, "dependencies": ["task_a"]},
            {"task_id": "task_a", "assigned_tool": "s1", "tool_parameters": {"log_path": log_path, "msg": "hello"}, "dependencies": []},
        ],
    )

    engine = PlanningAndToolEngine(registry=registry)
    executed_plan = engine.execute_plan(plan)

    assert executed_plan.overall_status == StepStatus.COMPLETED
    assert json.loads(Path(log_path).read_text(encoding="utf-8")) == ["step1", "step2"]
    assert executed_plan.tasks["task_a"].status == StepStatus.COMPLETED
    assert executed_plan.tasks["task_b"].status == StepStatus.COMPLETED


def test_automated_failure_recovery_via_retries(tmp_path):
    """Verify transient tool failures are retried and recovered."""
    registry = ToolRegistry()
    state_path = str(tmp_path / "attempts.sqlite")
    registry.register(ToolDefinition(
        name="flaky", description="flaky tool", required_parameters=["state_path"], handler=_test_flaky_tool
    ))

    plan = TaskDecomposer.create_plan_from_subtasks(
        goal="Retry Goal",
        subtasks_data=[
            {"task_id": "t_flaky", "assigned_tool": "flaky", "tool_parameters": {"state_path": state_path}, "max_retries": 2},
        ],
    )

    engine = PlanningAndToolEngine(registry=registry)
    executed = engine.execute_plan(plan)

    assert executed.overall_status == StepStatus.COMPLETED
    task = executed.tasks["t_flaky"]
    assert task.status == StepStatus.COMPLETED
    assert task.output == "Success on retry"
    with sqlite3.connect(state_path) as conn:
        assert conn.execute("SELECT n FROM attempts").fetchone()[0] == 2


def test_tool_fallback_routing_on_exhausted_primary():
    """Verify execution engine falls back to secondary tool when primary tool is exhausted."""
    registry = ToolRegistry()

    registry.register(ToolDefinition(name="primary_calc", description="primary", handler=_test_primary_failure))
    registry.register(ToolDefinition(name="fallback_calc", description="fallback", handler=_test_fallback_success))

    plan = TaskDecomposer.create_plan_from_subtasks(
        goal="Fallback Goal",
        subtasks_data=[
            {"task_id": "calc_task", "assigned_tool": "primary_calc", "max_retries": 1},
        ],
    )

    engine = PlanningAndToolEngine(registry=registry)
    engine.register_fallback_tool("primary_calc", "fallback_calc")

    executed = engine.execute_plan(plan)
    assert executed.overall_status == StepStatus.COMPLETED
    task = executed.tasks["calc_task"]
    assert task.assigned_tool == "fallback_calc"
    assert task.output == "Output from fallback service"


def test_result_verification_contract_rejects_empty_output():
    """Verify result verification contract rejects blank or null results."""
    task = TaskNode(task_id="t1", description="desc")

    # 1. Null output fails
    v1, msg1 = ResultVerificationContract.verify_result(task, None)
    assert v1 is False
    assert "null output" in msg1

    # 2. Blank output fails
    v2, msg2 = ResultVerificationContract.verify_result(task, "   ")
    assert v2 is False
    assert "blank string" in msg2

    # 3. Structured error fails
    v3, msg3 = ResultVerificationContract.verify_result(task, {"error": "Access Denied"})
    assert v3 is False
    assert "Access Denied" in msg3

    # 4. Valid result passes
    v4, msg4 = ResultVerificationContract.verify_result(task, {"data": [1, 2, 3]})
    assert v4 is True


def test_downstream_skip_propagation_on_fatal_failure():
    """Verify that if an upstream task fails completely, downstream tasks are safely SKIPPED."""
    registry = ToolRegistry()

    registry.register(ToolDefinition(name="fatal", description="fatal", handler=_test_fatal_failure))
    registry.register(ToolDefinition(name="downstream", description="downstream", handler=_test_downstream))

    plan = TaskDecomposer.create_plan_from_subtasks(
        goal="Failure skip goal",
        subtasks_data=[
            {"task_id": "root_task", "assigned_tool": "fatal", "max_retries": 0},
            {"task_id": "child_task", "assigned_tool": "downstream", "dependencies": ["root_task"]},
        ],
    )

    engine = PlanningAndToolEngine(registry=registry)
    executed = engine.execute_plan(plan)

    assert executed.overall_status == StepStatus.FAILED
    assert executed.tasks["root_task"].status == StepStatus.FAILED
    assert executed.tasks["child_task"].status == StepStatus.SKIPPED


def test_stage17_execution_runs_in_child_process_and_enforces_timeout():
    parent_pid = os.getpid()
    pid_tool = ToolDefinition(name="pid", description="Returns worker pid", handler=_test_current_pid)
    worker_pid = pid_tool.execute()
    assert worker_pid != parent_pid

    slow_tool = ToolDefinition(
        name="slow",
        description="Sleeps past its deadline",
        handler=_test_slow_handler,
        timeout_seconds=0.15,
    )
    started = time.monotonic()
    with pytest.raises(PlanningError, match="execution timeout"):
        slow_tool.execute()
    assert time.monotonic() - started < 2.0


def test_stage17_tool_output_is_bounded_and_json_serializable():
    oversized = ToolDefinition(
        name="large", description="Returns too much", handler=_test_large_output, max_output_bytes=128
    )
    with pytest.raises(PlanningError, match="OutputLimitExceeded"):
        oversized.execute()


def test_stage17_json_schema_and_safe_mutation_policy_are_enforced():
    schema = {
        "type": "object",
        "properties": {
            "resource": {"type": "string", "minLength": 1},
            "approved": {"type": "boolean"},
        },
        "required": ["resource"],
        "additionalProperties": False,
    }
    tool = ToolDefinition(
        name="safe-update",
        description="Updates a safe resource",
        risk_profile=ToolRiskProfile.MUTATING_SAFE,
        handler=_test_mutating_value,
        parameter_schema=schema,
    )
    locked_registry = ToolRegistry()
    locked_registry.register(tool)
    with pytest.raises(PlanningError, match="disabled by execution policy"):
        locked_registry.execute("safe-update", resource="cache")
    with pytest.raises(PlanningError, match="JSON Schema type"):
        locked_registry.execute("safe-update", resource=42)
    with pytest.raises(PlanningError, match="not allowed"):
        locked_registry.execute("safe-update", resource="cache", unknown=True)

    enabled_registry = ToolRegistry(allow_safe_mutations=True)
    enabled_registry.register(tool)
    assert enabled_registry.execute("safe-update", resource="cache") == {
        "updated": "cache", "approved": False
    }


def test_stage17_privileged_tools_require_application_authorizer():
    schema = {
        "type": "object",
        "properties": {"resource": {"type": "string"}},
        "required": ["resource"],
        "additionalProperties": False,
    }
    tool = ToolDefinition(
        name="privileged",
        description="Privileged update",
        risk_profile=ToolRiskProfile.MUTATING_PRIVILEGED,
        handler=_test_mutating_value,
        parameter_schema=schema,
    )
    denied = ToolRegistry()
    denied.register(tool)
    with pytest.raises(PlanningError, match="explicit application authorization"):
        denied.execute("privileged", resource="production-config")

    authorized = ToolRegistry(
        privileged_authorizer=lambda registered_tool, params: registered_tool.name == "privileged"
    )
    authorized.register(tool)
    result = authorized.execute("privileged", resource="production-config")
    assert result["updated"] == "production-config"


def test_stage17_fallback_cannot_escalate_risk_profile():
    registry = ToolRegistry()
    registry.register(ToolDefinition(name="readonly", description="read", handler=_test_adder))
    privileged = ToolDefinition(
        name="privileged-fallback",
        description="privileged",
        risk_profile=ToolRiskProfile.MUTATING_PRIVILEGED,
        parameter_schema={"type": "object"},
        handler=_test_mutating_value,
    )
    registry.register(privileged)
    engine = PlanningAndToolEngine(registry=registry)
    with pytest.raises(PlanningError, match="higher risk profile"):
        engine.register_fallback_tool("readonly", "privileged-fallback")


def test_stage17_mutating_tool_cannot_bypass_registry_authorization():
    schema = {
        "type": "object",
        "properties": {"resource": {"type": "string"}},
        "required": ["resource"],
        "additionalProperties": False,
    }
    tool = ToolDefinition(
        name="mutating-direct",
        description="Must be called through registry",
        risk_profile=ToolRiskProfile.MUTATING_SAFE,
        handler=_test_mutating_value,
        parameter_schema=schema,
    )
    with pytest.raises(PlanningError, match="through ToolRegistry"):
        tool.execute(resource="cache")


def test_stage17_registry_rejects_duplicate_tools_and_duplicate_task_ids():
    registry = ToolRegistry()
    tool = ToolDefinition(name="unique", description="Unique name", handler=_test_adder)
    registry.register(tool)
    with pytest.raises(PlanningError, match="already registered"):
        registry.register(tool)

    with pytest.raises(PlanningError, match="Duplicate task_id"):
        TaskDecomposer.create_plan_from_subtasks(
            "Duplicate task IDs",
            [
                {"task_id": "same", "description": "first"},
                {"task_id": "same", "description": "second"},
            ],
        )


def test_stage17_rejects_invalid_tool_timeout_and_schema_root():
    with pytest.raises(PlanningError, match="timeout_seconds"):
        ToolDefinition(
            name="invalid-timeout",
            description="Invalid deadline",
            handler=_test_adder,
            timeout_seconds=float("nan"),
        )
    tool = ToolDefinition(
        name="invalid-schema",
        description="Schema validation",
        handler=_test_adder,
        parameter_schema={"type": "array"},
    )
    with pytest.raises(PlanningError, match="root type"):
        tool.execute(a=1, b=2)
