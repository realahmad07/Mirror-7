"""Controlled, fail-closed execution of ready capability plans (Phase 7).

Only explicitly registered, importable native handlers can run. Each step is run in
a disposable child process with time, memory, input, output, and step-count bounds.
This is process isolation for trusted handlers, not a substitute for a hardened
container/VM policy when executing untrusted code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import multiprocessing
import os
import pickle
import time
from typing import Any, Callable, Iterable, Mapping

from .capability_model import CapabilitySpec
from .capability_composition import CompositionPlan, PlanStatus
from .universal_representation import _freeze_json, _thaw_json, canonical_json


OperationHandler = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class ExecutionError(ValueError):
    """Raised for invalid execution policies or requests."""


class ExecutionStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ExecutionPolicy:
    timeout_seconds: float = 3.0
    max_steps: int = 32
    max_input_bytes: int = 65536
    max_output_bytes: int = 65536
    max_memory_bytes: int = 536870912
    max_cpu_seconds: int = 3
    start_method: str = "spawn"

    def __post_init__(self) -> None:
        if isinstance(self.timeout_seconds, bool):
            raise ExecutionError("timeout_seconds must be finite and positive")
        try:
            timeout = float(self.timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise ExecutionError("timeout_seconds must be finite and positive") from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise ExecutionError("timeout_seconds must be finite and positive")
        object.__setattr__(self, "timeout_seconds", timeout)
        for name in ("max_steps", "max_input_bytes", "max_output_bytes", "max_memory_bytes", "max_cpu_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ExecutionError(f"{name} must be a positive integer")
        if self.start_method not in multiprocessing.get_all_start_methods():
            raise ExecutionError(f"unsupported multiprocessing start method: {self.start_method}")


@dataclass(frozen=True)
class StepExecutionResult:
    step_id: str
    subtask_id: str
    capability_id: str
    success: bool
    inputs: Mapping[str, Any] = field(default_factory=dict)
    outputs: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.step_id or not self.subtask_id or not self.capability_id:
            raise ExecutionError("step execution identifiers must not be empty")
        if not isinstance(self.success, bool):
            raise ExecutionError("step success must be a boolean")
        if isinstance(self.duration_ms, bool) or not math.isfinite(float(self.duration_ms)) or self.duration_ms < 0:
            raise ExecutionError("duration_ms must be a finite non-negative number")
        object.__setattr__(self, "inputs", _freeze_json(self.inputs, "execution.inputs"))
        object.__setattr__(self, "outputs", _freeze_json(self.outputs, "execution.outputs"))
        object.__setattr__(self, "duration_ms", float(self.duration_ms))
        if self.success and self.error:
            raise ExecutionError("successful step cannot contain an error")
        if not self.success and not self.error:
            raise ExecutionError("failed step must contain an error")


@dataclass(frozen=True)
class PlanExecutionReport:
    plan_id: str
    task_id: str
    status: ExecutionStatus
    steps: tuple[StepExecutionResult, ...]
    error: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ExecutionStatus(self.status))
        object.__setattr__(self, "steps", tuple(self.steps))
        if not self.plan_id or not self.task_id:
            raise ExecutionError("execution report requires plan_id and task_id")
        if self.status is ExecutionStatus.SUCCEEDED and (
            self.error or any(not step.success for step in self.steps)
        ):
            raise ExecutionError("successful report cannot contain failed steps")
        if self.status is not ExecutionStatus.SUCCEEDED and not self.error:
            raise ExecutionError("non-successful report must include an error")

    @property
    def succeeded(self) -> bool:
        return self.status is ExecutionStatus.SUCCEEDED

    @property
    def fingerprint(self) -> str:
        payload = {
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "status": self.status.value,
            "steps": [
                {
                    "step_id": step.step_id,
                    "capability_id": step.capability_id,
                    "success": step.success,
                    "inputs": _thaw_json(step.inputs),
                    "outputs": _thaw_json(step.outputs),
                    "error": step.error,
                }
                for step in self.steps
            ],
            "error": self.error,
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _child_run(send_conn, handler: OperationHandler, inputs: Mapping[str, Any], policy: ExecutionPolicy) -> None:
    try:
        if os.name == "posix":
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_CPU, (policy.max_cpu_seconds, policy.max_cpu_seconds))
                resource.setrlimit(resource.RLIMIT_AS, (policy.max_memory_bytes, policy.max_memory_bytes))
            except (ImportError, OSError, ValueError):
                pass
        result = handler(dict(inputs))
        if not isinstance(result, Mapping):
            raise TypeError("operation handler must return a mapping")
        try:
            payload = json.dumps(
                dict(result), sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise TypeError(f"operation output is not finite JSON data: {exc}") from exc
        if len(payload) > policy.max_output_bytes:
            raise ValueError("operation output exceeds max_output_bytes")
        send_conn.send_bytes(json.dumps({"ok": True, "payload": payload.decode("utf-8")}).encode("utf-8"))
    except Exception as exc:
        message = f"{type(exc).__name__}: {str(exc)[:1000]}"
        try:
            send_conn.send_bytes(json.dumps({"ok": False, "error": message}).encode("utf-8"))
        except BaseException:
            pass
    finally:
        send_conn.close()


class NativeOperationRegistry:
    """Allow-list of host-supplied, trusted native handlers; no dynamic code evaluation."""
    def __init__(self) -> None:
        self._handlers: dict[str, OperationHandler] = {}

    def register(self, capability_id: str, handler: OperationHandler) -> None:
        if not isinstance(capability_id, str) or not capability_id.strip():
            raise ExecutionError("capability_id must not be empty")
        if not callable(handler):
            raise TypeError("handler must be callable")
        try:
            pickle.dumps(handler)
        except Exception as exc:
            raise ExecutionError("handler must be importable and process-serializable") from exc
        key = capability_id.strip()
        if key in self._handlers and self._handlers[key] is not handler:
            raise ExecutionError(f"handler already registered for {key}")
        self._handlers[key] = handler

    def get(self, capability_id: str) -> OperationHandler | None:
        return self._handlers.get(str(capability_id))


class CapabilityExecutionEngine:
    """Execute a ready plan stepwise; stop immediately after the first failure."""

    def __init__(
        self,
        registry: NativeOperationRegistry,
        policy: ExecutionPolicy | None = None,
    ) -> None:
        if not isinstance(registry, NativeOperationRegistry):
            raise TypeError("registry must be NativeOperationRegistry")
        self.registry = registry
        self.policy = policy or ExecutionPolicy()

    @staticmethod
    def _validate_outputs(capability: CapabilitySpec, outputs: Mapping[str, Any]) -> None:
        declared = {port.name: port for port in capability.outputs}
        unknown = sorted(set(outputs) - set(declared))
        if unknown:
            raise ExecutionError("operation returned undeclared outputs: " + ", ".join(unknown))
        missing = sorted(
            port.name for port in capability.outputs
            if port.required and port.name not in outputs
        )
        if missing:
            raise ExecutionError("operation omitted required outputs: " + ", ".join(missing))
        invalid = sorted(
            name for name, value in outputs.items()
            if not declared[name].accepts_value(value)
        )
        if invalid:
            raise ExecutionError("operation returned wrong output types: " + ", ".join(invalid))

    def _run_handler(self, handler: OperationHandler, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = canonical_json(inputs).encode("utf-8")
        if len(payload) > self.policy.max_input_bytes:
            raise ExecutionError("step inputs exceed max_input_bytes")
        context = multiprocessing.get_context(self.policy.start_method)
        recv_conn, send_conn = context.Pipe(duplex=False)
        process = context.Process(
            target=_child_run,
            args=(send_conn, handler, dict(inputs), self.policy),
            daemon=True,
        )
        started = time.monotonic()
        try:
            process.start()
            send_conn.close()
            if not recv_conn.poll(self.policy.timeout_seconds):
                process.terminate()
                process.join(timeout=0.5)
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    process.join(timeout=0.5)
                raise TimeoutError("operation exceeded execution timeout")
            raw = recv_conn.recv_bytes(self.policy.max_output_bytes + 4096)
            process.join(timeout=0.5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
            envelope = json.loads(raw.decode("utf-8"))
            if not envelope.get("ok"):
                raise ExecutionError(str(envelope.get("error") or "operation failed"))
            outputs = json.loads(envelope["payload"])
            if not isinstance(outputs, dict):
                raise ExecutionError("operation returned a non-object payload")
            return outputs
        except (EOFError, OSError, json.JSONDecodeError) as exc:
            raise ExecutionError(f"operation process failed: {type(exc).__name__}") from exc
        finally:
            recv_conn.close()
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
            if process.exitcode is None and process.pid is not None:
                process.join(timeout=0.1)

    def execute(
        self,
        plan: CompositionPlan,
        capabilities: Iterable[CapabilitySpec],
        *,
        task_inputs: Mapping[str, Any] | None = None,
        task_context: Mapping[str, Any] | None = None,
    ) -> PlanExecutionReport:
        if not isinstance(plan, CompositionPlan):
            raise TypeError("plan must be a CompositionPlan")
        if not plan.ready:
            return PlanExecutionReport(
                plan.plan_id, plan.task_id, ExecutionStatus.BLOCKED, (),
                "composition plan is blocked",
            )
        if not plan.steps:
            return PlanExecutionReport(
                plan.plan_id, plan.task_id, ExecutionStatus.BLOCKED, (),
                "composition plan contains no steps",
            )
        if len(plan.steps) > self.policy.max_steps:
            return PlanExecutionReport(
                plan.plan_id, plan.task_id, ExecutionStatus.BLOCKED, (),
                "composition plan exceeds max_steps",
            )
        step_ids = [step.step_id for step in plan.steps]
        if len(set(step_ids)) != len(step_ids):
            return PlanExecutionReport(
                plan.plan_id, plan.task_id, ExecutionStatus.BLOCKED, (),
                "composition plan contains duplicate step identifiers",
            )
        capability_items = tuple(capabilities)
        capability_map = {item.capability_id: item for item in capability_items}
        if len(capability_map) != len(capability_items):
            return PlanExecutionReport(
                plan.plan_id, plan.task_id, ExecutionStatus.BLOCKED, (),
                "capability catalog contains duplicate identifiers",
            )
        known = dict(task_inputs or {})
        context_values = dict(task_context or {})
        step_results: dict[str, StepExecutionResult] = {}
        completed: set[str] = set()
        ordered_results: list[StepExecutionResult] = []

        for step in plan.steps:
            capability = capability_map.get(step.capability_id)
            if capability is None:
                return PlanExecutionReport(plan.plan_id, plan.task_id, ExecutionStatus.BLOCKED,
                    tuple(ordered_results), f"unknown capability: {step.capability_id}")
            if not capability.executable:
                return PlanExecutionReport(plan.plan_id, plan.task_id, ExecutionStatus.BLOCKED,
                    tuple(ordered_results), f"capability is not evidence-eligible: {capability.capability_id}")
            missing_steps = sorted(set(step.dependency_step_ids) - completed)
            if missing_steps:
                return PlanExecutionReport(plan.plan_id, plan.task_id, ExecutionStatus.BLOCKED,
                    tuple(ordered_results), "step dependencies are not complete: " + ", ".join(missing_steps))
            handler = self.registry.get(capability.capability_id)
            if handler is None:
                return PlanExecutionReport(plan.plan_id, plan.task_id, ExecutionStatus.BLOCKED,
                    tuple(ordered_results), f"no native handler registered: {capability.capability_id}")
            binding_map = {item.input_name: item for item in step.input_bindings}
            declared_input_names = {port.name for port in capability.inputs}
            unknown_bindings = sorted(set(binding_map) - declared_input_names)
            if unknown_bindings:
                return PlanExecutionReport(plan.plan_id, plan.task_id, ExecutionStatus.BLOCKED,
                    tuple(ordered_results), "bindings reference undeclared inputs: " + ", ".join(unknown_bindings))
            if len(binding_map) != len(step.input_bindings):
                return PlanExecutionReport(plan.plan_id, plan.task_id, ExecutionStatus.BLOCKED,
                    tuple(ordered_results), f"duplicate input binding for step {step.step_id}")
            values: dict[str, Any] = {}
            for binding in step.input_bindings:
                if binding.source == "task_input":
                    if binding.source_name not in known:
                        return PlanExecutionReport(plan.plan_id, plan.task_id, ExecutionStatus.BLOCKED,
                            tuple(ordered_results), f"missing task input: {binding.source_name}")
                    values[binding.input_name] = known[binding.source_name]
                elif binding.source == "task_context":
                    if binding.source_name not in context_values:
                        return PlanExecutionReport(plan.plan_id, plan.task_id, ExecutionStatus.BLOCKED,
                            tuple(ordered_results), f"missing task context: {binding.source_name}")
                    values[binding.input_name] = context_values[binding.source_name]
                elif binding.source == "step_output":
                    source_step = next((item for item in plan.steps if item.step_id == binding.source_step_id), None)
                    source_capability = capability_map.get(source_step.capability_id) if source_step else None
                    source_port = next(
                        (port for port in source_capability.outputs if port.name == binding.source_name),
                        None,
                    ) if source_capability else None
                    target_port = next(
                        (port for port in capability.inputs if port.name == binding.input_name),
                        None,
                    )
                    if source_port is None or target_port is None or not target_port.accepts_output(source_port):
                        return PlanExecutionReport(plan.plan_id, plan.task_id, ExecutionStatus.BLOCKED,
                            tuple(ordered_results), f"incompatible dependency port binding: {binding.input_name}")
                    source = step_results.get(binding.source_step_id)
                    if source is None or not source.success:
                        return PlanExecutionReport(plan.plan_id, plan.task_id, ExecutionStatus.BLOCKED,
                            tuple(ordered_results), f"unavailable dependency output: {binding.source_step_id}")
                    if binding.source_name not in source.outputs:
                        return PlanExecutionReport(plan.plan_id, plan.task_id, ExecutionStatus.BLOCKED,
                            tuple(ordered_results), f"missing dependency output: {binding.source_name}")
                    values[binding.input_name] = source.outputs[binding.source_name]
                else:
                    return PlanExecutionReport(plan.plan_id, plan.task_id, ExecutionStatus.BLOCKED,
                        tuple(ordered_results), f"unknown binding source: {binding.source}")
            for port in capability.inputs:
                if port.required and port.name not in values:
                    return PlanExecutionReport(plan.plan_id, plan.task_id, ExecutionStatus.BLOCKED,
                        tuple(ordered_results), f"required input unbound: {port.name}")
                if port.name in values and not port.accepts_value(values[port.name]):
                    return PlanExecutionReport(plan.plan_id, plan.task_id, ExecutionStatus.BLOCKED,
                        tuple(ordered_results), f"input type mismatch: {port.name}")
            started = time.monotonic()
            try:
                outputs = self._run_handler(handler, values)
                self._validate_outputs(capability, outputs)
                item = StepExecutionResult(
                    step.step_id, step.subtask_id, step.capability_id, True,
                    values, outputs, duration_ms=(time.monotonic() - started) * 1000.0,
                )
            except Exception as exc:
                item = StepExecutionResult(
                    step.step_id, step.subtask_id, step.capability_id, False,
                    values, {}, f"{type(exc).__name__}: {str(exc)[:1000]}",
                    duration_ms=(time.monotonic() - started) * 1000.0,
                )
                ordered_results.append(item)
                return PlanExecutionReport(
                    plan.plan_id, plan.task_id, ExecutionStatus.FAILED,
                    tuple(ordered_results), f"execution stopped at {step.step_id}: {item.error}",
                )
            ordered_results.append(item)
            step_results[step.step_id] = item
            completed.add(step.step_id)

        return PlanExecutionReport(
            plan.plan_id, plan.task_id, ExecutionStatus.SUCCEEDED,
            tuple(ordered_results),
        )
