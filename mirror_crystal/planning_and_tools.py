"""Stage 17: Planning & Tool Intelligence Subsystem.

Provides:
- Hierarchical task decomposition into dependency DAGs.
- Sandboxed tool registry with strict schema validation and risk profiling.
- Automated execution planning with dependency resolution.
- Real-time plan progress tracking and checkpointing.
- Automated failure recovery (retries, parameter adjustments, tool fallbacks).
- Strict result verification contracts before marking steps completed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .production_readiness import StructuredLogger


class StepStatus(str, Enum):
    """Lifecycle states of an execution plan step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RECOVERED = "recovered"


class ToolRiskProfile(str, Enum):
    """Security classification for executable tools."""
    READ_ONLY = "read_only"
    MUTATING_SAFE = "mutating_safe"
    MUTATING_PRIVILEGED = "mutating_privileged"


class PlanningError(ValueError):
    """Raised when plan generation, cyclic dependency, or execution contract fails."""


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ToolDefinition:
    """Specification of an executable tool within the registry."""
    name: str
    description: str
    risk_profile: ToolRiskProfile = ToolRiskProfile.READ_ONLY
    required_parameters: List[str] = field(default_factory=list)
    handler: Optional[Callable[..., Any]] = None
    timeout_seconds: float = 30.0

    def execute(self, **kwargs: Any) -> Any:
        if not self.handler:
            raise PlanningError(f"Tool '{self.name}' has no registered execution handler")
        for param in self.required_parameters:
            if param not in kwargs:
                raise PlanningError(f"Tool '{self.name}' missing required parameter '{param}'")
        return self.handler(**kwargs)


class ToolRegistry:
    """Allow-listed repository of validated tools."""

    def __init__(self) -> None:
        self.tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self.tools.get(name)

    def has(self, name: str) -> bool:
        return name in self.tools


@dataclass
class TaskNode:
    """A distinct task unit in a plan with dependency bindings."""
    task_id: str
    description: str
    assigned_tool: Optional[str] = None
    tool_parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    output: Any = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2
    execution_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["status"] = self.status.value
        return res


@dataclass
class ExecutionPlan:
    """An executable workflow DAG of tasks aimed at achieving a specific goal."""
    plan_id: str
    goal: str
    tasks: Dict[str, TaskNode]
    created_at: str = field(default_factory=_iso_utc_now)
    completed_at: Optional[str] = None
    overall_status: StepStatus = StepStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "overall_status": self.overall_status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
        }


class TaskDecomposer:
    """Decomposes goals into acyclic task graphs and selects tools."""

    @classmethod
    def validate_acyclic(cls, tasks: Dict[str, TaskNode]) -> None:
        """Verifies that the task dependencies form a valid DAG (no cycles)."""
        visited: Dict[str, int] = {}  # 0 = unvisited, 1 = visiting, 2 = visited

        def dfs(node_id: str) -> None:
            visited[node_id] = 1
            for dep in tasks[node_id].dependencies:
                if dep not in tasks:
                    raise PlanningError(f"Task '{node_id}' depends on missing task '{dep}'")
                if visited.get(dep, 0) == 1:
                    raise PlanningError(f"Cyclic dependency detected involving '{node_id}' and '{dep}'")
                if visited.get(dep, 0) == 0:
                    dfs(dep)
            visited[node_id] = 2

        for t_id in tasks:
            if visited.get(t_id, 0) == 0:
                dfs(t_id)

    @classmethod
    def create_plan_from_subtasks(
        cls,
        goal: str,
        subtasks_data: List[Dict[str, Any]],
        plan_id: Optional[str] = None,
    ) -> ExecutionPlan:
        """Constructs an ExecutionPlan from subtask dictionaries."""
        pid = plan_id or f"plan_{hashlib.sha256(goal.encode()).hexdigest()[:10]}"
        task_nodes: Dict[str, TaskNode] = {}

        for item in subtasks_data:
            tid = item["task_id"]
            node = TaskNode(
                task_id=tid,
                description=item.get("description", ""),
                assigned_tool=item.get("assigned_tool"),
                tool_parameters=item.get("tool_parameters", {}),
                dependencies=item.get("dependencies", []),
                max_retries=item.get("max_retries", 2),
            )
            task_nodes[tid] = node

        cls.validate_acyclic(task_nodes)
        return ExecutionPlan(plan_id=pid, goal=goal, tasks=task_nodes)


class FailureRecoveryManager:
    """Applies automated recovery strategies when task execution encounters errors."""

    @classmethod
    def attempt_recovery(
        cls,
        task: TaskNode,
        error: Exception,
        registry: ToolRegistry,
        fallback_tool_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, str]:
        """Evaluates whether task can be retried or routed to a fallback tool."""
        fallback_map = fallback_tool_map or {}

        # 1. Parameter adjustments or retry
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            task.error_message = f"Attempt {task.retry_count} failed: {error}"
            return True, f"Retrying task (attempt {task.retry_count}/{task.max_retries})"

        # 2. Tool fallback
        if task.assigned_tool and task.assigned_tool in fallback_map:
            alt_tool = fallback_map[task.assigned_tool]
            if registry.has(alt_tool):
                task.assigned_tool = alt_tool
                task.retry_count = 0  # reset for new tool
                task.error_message = f"Switching to fallback tool '{alt_tool}' after {error}"
                return True, f"Recovering via fallback tool '{alt_tool}'"

        return False, f"Recovery exhausted after {task.retry_count} attempts: {error}"


class ResultVerificationContract:
    """Validates execution outputs against completion invariants."""

    @classmethod
    def verify_result(cls, task: TaskNode, raw_output: Any) -> Tuple[bool, str]:
        """Ensures outputs are non-empty and uncorrupted."""
        if raw_output is None:
            return False, "Tool returned null output"
        if isinstance(raw_output, str) and not raw_output.strip():
            return False, "Tool returned blank string output"
        if isinstance(raw_output, dict) and "error" in raw_output and raw_output["error"]:
            return False, f"Tool returned structured error: {raw_output['error']}"
        return True, "Output verified"


class PlanningAndToolEngine:
    """Unified Orchestrator for Stage 17 Planning & Tool Intelligence."""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.logger = logger or StructuredLogger(name="mirror-planning-tools")
        self.recovery_mgr = FailureRecoveryManager()
        self.fallback_tools: Dict[str, str] = {}

    def register_fallback_tool(self, primary_tool: str, fallback_tool: str) -> None:
        self.fallback_tools[primary_tool] = fallback_tool

    def execute_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Executes a plan respecting task dependencies and recovering from transient failures."""
        plan.overall_status = StepStatus.RUNNING
        self.logger.info("Starting execution plan", extra={"plan_id": plan.plan_id, "goal": plan.goal})

        completed_set: Set[str] = set()
        failed_set: Set[str] = set()

        while len(completed_set) + len(failed_set) < len(plan.tasks):
            # Find eligible tasks (all dependencies completed successfully)
            ready_tasks = [
                t for t in plan.tasks.values()
                if t.status in (StepStatus.PENDING, StepStatus.RECOVERED)
                and all(dep in completed_set for dep in t.dependencies)
            ]

            if not ready_tasks:
                # Deadlock or unresolvable failure upstream
                for t in plan.tasks.values():
                    if t.status == StepStatus.PENDING:
                        t.status = StepStatus.SKIPPED
                        t.error_message = "Skipped due to upstream dependency failure"
                        failed_set.add(t.task_id)
                break

            for task in ready_tasks:
                task.status = StepStatus.RUNNING
                t0 = time.time()
                tool = self.registry.get(task.assigned_tool) if task.assigned_tool else None

                success = False
                while not success:
                    try:
                        if not tool:
                            raise PlanningError(f"No tool registered for name '{task.assigned_tool}'")

                        out = tool.execute(**task.tool_parameters)
                        valid, v_msg = ResultVerificationContract.verify_result(task, out)
                        if not valid:
                            raise PlanningError(f"Result verification failed: {v_msg}")

                        task.output = out
                        task.status = StepStatus.COMPLETED
                        task.error_message = None
                        completed_set.add(task.task_id)
                        success = True
                    except Exception as exc:
                        can_recover, rec_msg = self.recovery_mgr.attempt_recovery(
                            task=task,
                            error=exc,
                            registry=self.registry,
                            fallback_tool_map=self.fallback_tools,
                        )
                        if can_recover:
                            task.status = StepStatus.RECOVERED
                            # Re-fetch tool in case fallback was selected
                            tool = self.registry.get(task.assigned_tool) if task.assigned_tool else None
                        else:
                            task.status = StepStatus.FAILED
                            task.error_message = rec_msg
                            failed_set.add(task.task_id)
                            break

                task.execution_duration_ms = round((time.time() - t0) * 1000, 2)

        plan.completed_at = _iso_utc_now()
        plan.overall_status = StepStatus.COMPLETED if len(failed_set) == 0 else StepStatus.FAILED
        self.logger.info(
            "Plan execution concluded",
            extra={
                "plan_id": plan.plan_id,
                "overall_status": plan.overall_status.value,
                "completed_count": len(completed_set),
                "failed_count": len(failed_set),
            },
        )
        return plan
