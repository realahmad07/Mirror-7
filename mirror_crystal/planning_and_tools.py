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
import json
import math
import multiprocessing
import os
import re
import signal
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


class ToolExecutionMode(str, Enum):
    """Execution boundary for a registered tool handler."""
    ISOLATED_PROCESS = "isolated_process"
    CONTAINER = "container"


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _echo_tool(text: str = "") -> str:
    """Built-in read-only echo handler used by the HTTP planning demo."""
    return text


def _tool_process_worker(send_conn: Any, handler: Callable[..., Any], parameters: Dict[str, Any],
                         timeout_seconds: float, memory_limit_mb: int, max_output_bytes: int) -> None:
    """Execute an allow-listed handler in a killable child process and return JSON only."""
    try:
        if os.name == "posix":
            try:
                os.setsid()
            except OSError:
                pass
            try:
                import resource
                cpu = max(1, int(math.ceil(timeout_seconds)))
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
                resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
                resource.setrlimit(resource.RLIMIT_FSIZE, (1_048_576, 1_048_576))
                # Bound additional address-space growth without assuming a fixed
                # baseline for Python's inherited runtime mappings.
                try:
                    with open("/proc/self/statm", "r", encoding="ascii") as handle:
                        current_pages = int(handle.read().split()[0])
                    page_size = os.sysconf("SC_PAGE_SIZE")
                    current_vms = current_pages * page_size
                    limit = current_vms + max(64, int(memory_limit_mb)) * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
                except (OSError, ValueError, IndexError, AttributeError):
                    pass
            except (ImportError, OSError, ValueError):
                pass

        result = handler(**parameters)
        try:
            encoded = json.dumps({"ok": True, "result": result}, allow_nan=False,
                                 separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            encoded = json.dumps({"ok": False, "error_type": "NonJSONResult",
                                  "error": f"Tool result must be JSON-serializable: {exc}"}).encode("utf-8")
        if len(encoded) > max_output_bytes:
            encoded = json.dumps({"ok": False, "error_type": "OutputLimitExceeded",
                                  "error": f"Tool result exceeds {max_output_bytes} bytes"}).encode("utf-8")
    except BaseException as exc:
        encoded = json.dumps({"ok": False, "error_type": type(exc).__name__,
                              "error": str(exc)[:2000]}).encode("utf-8")
    try:
        send_conn.send_bytes(encoded)
    except (BrokenPipeError, OSError):
        pass
    finally:
        send_conn.close()


@dataclass
class ToolDefinition:
    """Allow-listed tool contract with schema validation and bounded execution."""

    name: str
    description: str
    risk_profile: ToolRiskProfile = ToolRiskProfile.READ_ONLY
    required_parameters: List[str] = field(default_factory=list)
    handler: Optional[Callable[..., Any]] = None
    timeout_seconds: float = 30.0
    parameter_schema: Dict[str, Any] = field(default_factory=dict)
    execution_mode: ToolExecutionMode = ToolExecutionMode.ISOLATED_PROCESS
    container_source: Optional[str] = None
    memory_limit_mb: int = 256
    max_output_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", self.name):
            raise PlanningError("Tool name must be 1-64 safe identifier characters")
        if not isinstance(self.description, str) or not self.description.strip():
            raise PlanningError("Tool description must not be empty")
        if not isinstance(self.risk_profile, ToolRiskProfile):
            raise PlanningError("risk_profile must be a ToolRiskProfile")
        if not isinstance(self.execution_mode, ToolExecutionMode):
            raise PlanningError("execution_mode must be a ToolExecutionMode")
        if isinstance(self.timeout_seconds, bool) or not math.isfinite(float(self.timeout_seconds)) or not (0.05 <= self.timeout_seconds <= 300):
            raise PlanningError("timeout_seconds must be finite and between 0.05 and 300")
        if isinstance(self.memory_limit_mb, bool) or not isinstance(self.memory_limit_mb, int) or not (64 <= self.memory_limit_mb <= 8192):
            raise PlanningError("memory_limit_mb must be an integer between 64 and 8192")
        if isinstance(self.max_output_bytes, bool) or not isinstance(self.max_output_bytes, int) or not (1 <= self.max_output_bytes <= 10_000_000):
            raise PlanningError("max_output_bytes must be an integer between 1 and 10000000")
        if not isinstance(self.required_parameters, list) or any(not isinstance(x, str) for x in self.required_parameters):
            raise PlanningError("required_parameters must be a list of strings")
        if not isinstance(self.parameter_schema, dict):
            raise PlanningError("parameter_schema must be a JSON Schema object")
        if self.risk_profile != ToolRiskProfile.READ_ONLY and not self.parameter_schema:
            raise PlanningError("Mutating tools require an explicit parameter_schema")
        if self.execution_mode == ToolExecutionMode.CONTAINER:
            if not isinstance(self.container_source, str) or not self.container_source.strip():
                raise PlanningError("Container tools require container_source defining run(params)")
        elif self.handler is None or not callable(self.handler):
            raise PlanningError("Process-isolated tools require a callable handler")

    @staticmethod
    def _matches_type(value: Any, expected: str) -> bool:
        if expected == "object":
            return isinstance(value, dict)
        if expected == "array":
            return isinstance(value, list)
        if expected == "string":
            return isinstance(value, str)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "null":
            return value is None
        raise PlanningError(f"Unsupported JSON Schema type: {expected}")

    @classmethod
    def _validate_value(cls, value: Any, schema: Dict[str, Any], path: str) -> None:
        expected = schema.get("type")
        if expected and not cls._matches_type(value, expected):
            raise PlanningError(f"Parameter '{path}' must have JSON Schema type '{expected}'")
        if "enum" in schema and value not in schema["enum"]:
            raise PlanningError(f"Parameter '{path}' is not one of the allowed enum values")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in schema and value < schema["minimum"]:
                raise PlanningError(f"Parameter '{path}' is below minimum")
            if "maximum" in schema and value > schema["maximum"]:
                raise PlanningError(f"Parameter '{path}' exceeds maximum")
        if isinstance(value, str):
            if "minLength" in schema and len(value) < schema["minLength"]:
                raise PlanningError(f"Parameter '{path}' is shorter than minLength")
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                raise PlanningError(f"Parameter '{path}' exceeds maxLength")
        if isinstance(value, list):
            if "minItems" in schema and len(value) < schema["minItems"]:
                raise PlanningError(f"Parameter '{path}' has too few items")
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                raise PlanningError(f"Parameter '{path}' has too many items")
            item_schema = schema.get("items")
            if item_schema:
                for index, item in enumerate(value):
                    cls._validate_value(item, item_schema, f"{path}[{index}]")
        if isinstance(value, dict):
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            if not isinstance(properties, dict) or not isinstance(required, list):
                raise PlanningError(f"Tool schema for '{path}' is malformed")
            missing = [key for key in required if key not in value]
            if missing:
                raise PlanningError(f"Parameter object '{path}' is missing required keys: {', '.join(missing)}")
            additional = schema.get("additionalProperties", False)
            for key, item in value.items():
                if key in properties:
                    cls._validate_value(item, properties[key], f"{path}.{key}")
                elif additional is False:
                    raise PlanningError(f"Parameter '{path}.{key}' is not allowed by the tool schema")
                elif isinstance(additional, dict):
                    cls._validate_value(item, additional, f"{path}.{key}")

    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        if not isinstance(parameters, dict):
            raise PlanningError(f"Tool '{self.name}' parameters must be an object")
        missing = [param for param in self.required_parameters if param not in parameters]
        if missing:
            raise PlanningError(f"Tool '{self.name}' missing required parameter '{missing[0]}'")
        if self.parameter_schema:
            if self.parameter_schema.get("type", "object") != "object":
                raise PlanningError("Tool parameter_schema root type must be object")
            self._validate_value(parameters, {"type": "object", **self.parameter_schema}, self.name)
        try:
            encoded = json.dumps(parameters, allow_nan=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise PlanningError(f"Tool '{self.name}' parameters must be JSON-safe: {exc}") from exc
        if len(encoded) > 512_000:
            raise PlanningError(f"Tool '{self.name}' parameters exceed 512 KB")

    def execute(self, **kwargs: Any) -> Any:
        """Direct execution is restricted to read-only tools; mutation requires registry policy."""
        if self.risk_profile != ToolRiskProfile.READ_ONLY:
            raise PlanningError("Mutating tools must be executed through ToolRegistry policy checks")
        self.validate_parameters(kwargs)
        return self._execute_validated(kwargs)

    def _execute_validated(self, kwargs: Dict[str, Any]) -> Any:
        self.validate_parameters(kwargs)
        if self.execution_mode == ToolExecutionMode.CONTAINER:
            from .container_sandbox import run_python_in_container
            parameters_json = json.dumps(kwargs, allow_nan=False, separators=(",", ":"))
            source = (
                "import json\n"
                + self.container_source
                + "\n_params = json.loads(" + repr(parameters_json) + ")\n"
                + "_result = run(_params)\n"
                + "print(json.dumps(_result, allow_nan=False))\n"
            )
            result = run_python_in_container(
                source,
                timeout_seconds=min(60.0, self.timeout_seconds),
                memory_limit_mb=self.memory_limit_mb,
            )
            if not result.ok:
                raise PlanningError(f"Container tool '{self.name}' failed: {result.error}")
            lines = result.output.splitlines()
            if not lines:
                raise PlanningError(f"Container tool '{self.name}' returned no JSON result")
            try:
                raw = json.loads(lines[-1])
                encoded = json.dumps(raw, allow_nan=False).encode("utf-8")
            except (ValueError, TypeError) as exc:
                raise PlanningError(f"Container tool '{self.name}' returned invalid JSON: {exc}") from exc
            if len(encoded) > self.max_output_bytes:
                raise PlanningError(f"Tool '{self.name}' output exceeds {self.max_output_bytes} bytes")
            return raw

        context = multiprocessing.get_context("fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn")
        recv_conn, send_conn = context.Pipe(duplex=False)
        process = context.Process(
            target=_tool_process_worker,
            args=(send_conn, self.handler, kwargs, self.timeout_seconds, self.memory_limit_mb, self.max_output_bytes),
            name=f"mirror-tool-{self.name}",
        )
        process.daemon = True
        try:
            process.start()
            send_conn.close()
            if not recv_conn.poll(self.timeout_seconds):
                self._terminate_process_tree(process)
                raise PlanningError(f"Tool '{self.name}' exceeded {self.timeout_seconds:.3f}s execution timeout")
            try:
                raw_bytes = recv_conn.recv_bytes(maxlength=self.max_output_bytes + 16_384)
            except (EOFError, OSError) as exc:
                raise PlanningError(f"Tool '{self.name}' worker exited without a valid result") from exc
            process.join(timeout=0.5)
            if process.is_alive():
                self._terminate_process_tree(process)
            try:
                response = json.loads(raw_bytes.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise PlanningError(f"Tool '{self.name}' worker returned malformed response data") from exc
            if not response.get("ok"):
                raise PlanningError(
                    f"Tool '{self.name}' failed ({response.get('error_type', 'ToolError')}): "
                    f"{response.get('error', 'unspecified error')}"
                )
            return response.get("result")
        except (OSError, TypeError, ValueError) as exc:
            if isinstance(exc, PlanningError):
                raise
            raise PlanningError(f"Tool '{self.name}' could not start isolated execution: {exc}") from exc
        finally:
            try:
                recv_conn.close()
                send_conn.close()
            except OSError:
                pass
            if process.is_alive():
                self._terminate_process_tree(process)

    @staticmethod
    def _terminate_process_tree(process: Any) -> None:
        if not process.is_alive():
            process.join(timeout=0.1)
            return
        if os.name == "posix" and process.pid:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.join(timeout=0.2)
                if process.is_alive():
                    os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                process.terminate()
        else:
            process.terminate()
        process.join(timeout=1.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=1.0)


class ToolRegistry:
    """Allow-listed tool registry with enforced risk policies and authorization."""

    def __init__(
        self,
        *,
        allow_safe_mutations: bool = False,
        privileged_authorizer: Optional[Callable[[ToolDefinition, Dict[str, Any]], bool]] = None,
    ) -> None:
        self.tools: Dict[str, ToolDefinition] = {}
        self.allow_safe_mutations = bool(allow_safe_mutations)
        self.privileged_authorizer = privileged_authorizer

    def register(self, tool: ToolDefinition) -> None:
        if not isinstance(tool, ToolDefinition):
            raise PlanningError("Only ToolDefinition instances can be registered")
        if tool.name in self.tools:
            raise PlanningError(f"Tool '{tool.name}' is already registered")
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self.tools.get(name)

    def has(self, name: str) -> bool:
        return name in self.tools

    def execute(self, name: str, **kwargs: Any) -> Any:
        tool = self.get(name)
        if tool is None:
            raise PlanningError(f"No tool registered for name '{name}'")
        tool.validate_parameters(kwargs)
        if tool.risk_profile == ToolRiskProfile.MUTATING_SAFE and not self.allow_safe_mutations:
            raise PlanningError(f"Safe mutation tool '{name}' is disabled by execution policy")
        if tool.risk_profile == ToolRiskProfile.MUTATING_PRIVILEGED:
            if self.privileged_authorizer is None or not self.privileged_authorizer(tool, kwargs):
                raise PlanningError(f"Privileged tool '{name}' requires explicit application authorization")
        return tool._execute_validated(kwargs)


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
        primary = self.registry.get(primary_tool)
        fallback = self.registry.get(fallback_tool)
        if primary is None or fallback is None:
            raise PlanningError("Both primary and fallback tools must be registered before fallback mapping")
        risk_rank = {
            ToolRiskProfile.READ_ONLY: 0,
            ToolRiskProfile.MUTATING_SAFE: 1,
            ToolRiskProfile.MUTATING_PRIVILEGED: 2,
        }
        if risk_rank[fallback.risk_profile] > risk_rank[primary.risk_profile]:
            raise PlanningError("Fallback tools cannot have a higher risk profile than the primary tool")
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

                        out = self.registry.execute(tool.name, **task.tool_parameters)
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
