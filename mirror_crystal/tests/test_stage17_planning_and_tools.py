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
    def sample_adder(a: int, b: int) -> int:
        return a + b

    tool = ToolDefinition(
        name="adder",
        description="Adds two integers",
        risk_profile=ToolRiskProfile.READ_ONLY,
        required_parameters=["a", "b"],
        handler=sample_adder,
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


def test_dependency_ordered_plan_execution():
    """Verify tasks execute strictly in dependency order."""
    registry = ToolRegistry()
    execution_order = []

    def tool_step1(msg: str) -> str:
        execution_order.append("step1")
        return f"Step 1 Done: {msg}"

    def tool_step2(prev: str) -> str:
        execution_order.append("step2")
        return f"Step 2 Done with {prev}"

    registry.register(ToolDefinition(name="s1", description="s1", handler=tool_step1))
    registry.register(ToolDefinition(name="s2", description="s2", handler=tool_step2))

    plan = TaskDecomposer.create_plan_from_subtasks(
        goal="Order test",
        subtasks_data=[
            {"task_id": "task_b", "assigned_tool": "s2", "tool_parameters": {"prev": "data"}, "dependencies": ["task_a"]},
            {"task_id": "task_a", "assigned_tool": "s1", "tool_parameters": {"msg": "hello"}, "dependencies": []},
        ],
    )

    engine = PlanningAndToolEngine(registry=registry)
    executed_plan = engine.execute_plan(plan)

    assert executed_plan.overall_status == StepStatus.COMPLETED
    assert execution_order == ["step1", "step2"]
    assert executed_plan.tasks["task_a"].status == StepStatus.COMPLETED
    assert executed_plan.tasks["task_b"].status == StepStatus.COMPLETED


def test_automated_failure_recovery_via_retries():
    """Verify transient tool failures are retried and recovered."""
    registry = ToolRegistry()
    attempts = 0

    def flaky_tool() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ConnectionError("Transient network timeout")
        return "Success on retry"

    registry.register(ToolDefinition(name="flaky", description="flaky tool", handler=flaky_tool))

    plan = TaskDecomposer.create_plan_from_subtasks(
        goal="Retry Goal",
        subtasks_data=[
            {"task_id": "t_flaky", "assigned_tool": "flaky", "max_retries": 2},
        ],
    )

    engine = PlanningAndToolEngine(registry=registry)
    executed = engine.execute_plan(plan)

    assert executed.overall_status == StepStatus.COMPLETED
    task = executed.tasks["t_flaky"]
    assert task.status == StepStatus.COMPLETED
    assert task.output == "Success on retry"
    assert attempts == 2


def test_tool_fallback_routing_on_exhausted_primary():
    """Verify execution engine falls back to secondary tool when primary tool is exhausted."""
    registry = ToolRegistry()

    def primary_failing() -> str:
        raise RuntimeError("Primary service unavailable")

    def fallback_working() -> str:
        return "Output from fallback service"

    registry.register(ToolDefinition(name="primary_calc", description="primary", handler=primary_failing))
    registry.register(ToolDefinition(name="fallback_calc", description="fallback", handler=fallback_working))

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

    def fatal_tool() -> None:
        raise PermissionError("Fatal authentication error")

    def downstream_tool() -> str:
        return "Should not run"

    registry.register(ToolDefinition(name="fatal", description="fatal", handler=fatal_tool))
    registry.register(ToolDefinition(name="downstream", description="downstream", handler=downstream_tool))

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
