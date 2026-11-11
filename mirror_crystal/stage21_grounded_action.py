"""Stage 21: grounded, approval-gated action proposals and feedback.

Actions are pure bounded computations executed in the existing isolated container.
They cannot access the network, host filesystem or hardware. Consequential actions
require a separate human approval token. A host integration must separately enforce
that token before performing any real-world side effect.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping

from .container_sandbox import DEFAULT_IMAGE, run_python_regression_in_container
from .stage20_multimodal_grounding import GroundingReport, GroundingStatus


class GroundedActionError(ValueError):
    """Invalid action, untrusted evidence, verifier, approval or execution record."""


class ActionRisk(str, Enum):
    READ_ONLY = "read_only"
    REVERSIBLE = "reversible"
    CONSEQUENTIAL = "consequential"


class ActionCheckDecision(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class ActionExecutionStatus(str, Enum):
    SUCCEEDED = "succeeded"
    POSTCONDITION_MISMATCH = "postcondition_mismatch"
    FAILED = "failed"


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SLUG = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise GroundedActionError(f"value must be finite JSON data: {exc}") from exc


def _digest(value: Any) -> str:
    data = value if isinstance(value, bytes) else str(value).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _freeze(value: Any) -> Any:
    normalized = json.loads(_canonical(value))
    def visit(item: Any) -> Any:
        if isinstance(item, dict):
            return MappingProxyType({key: visit(child) for key, child in item.items()})
        if isinstance(item, list):
            return tuple(visit(child) for child in item)
        return item
    return visit(normalized)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value


def _text(value: Any, field: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GroundedActionError(f"{field} must be non-empty text")
    value = value.strip()
    if len(value) > maximum:
        raise GroundedActionError(f"{field} exceeds {maximum} characters")
    return value


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise GroundedActionError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _utc(value: str | None, field: str) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if not isinstance(value, str):
        raise GroundedActionError(f"{field} must be a timezone-aware ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GroundedActionError(f"{field} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise GroundedActionError(f"{field} must include an explicit timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def _validate_schema_value(value: Any, schema: Mapping[str, Any], path: str) -> None:
    if not isinstance(schema, Mapping):
        raise GroundedActionError(f"schema at {path} must be an object")
    expected = schema.get("type")
    match = {
        "object": lambda x: isinstance(x, dict),
        "array": lambda x: isinstance(x, list),
        "string": lambda x: isinstance(x, str),
        "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
        "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)),
        "boolean": lambda x: isinstance(x, bool),
        "null": lambda x: x is None,
    }
    if expected is not None:
        if expected not in match:
            raise GroundedActionError(f"unsupported schema type at {path}: {expected}")
        if not match[expected](value):
            raise GroundedActionError(f"{path} must have schema type {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise GroundedActionError(f"{path} is outside its allowed enum")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise GroundedActionError(f"{path} is below its minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise GroundedActionError(f"{path} exceeds its maximum")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise GroundedActionError(f"{path} is shorter than minLength")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise GroundedActionError(f"{path} exceeds maxLength")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise GroundedActionError(f"{path} has too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise GroundedActionError(f"{path} has too many items")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                _validate_schema_value(item, item_schema, f"{path}[{index}]")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, Mapping) or not isinstance(required, (list, tuple)):
            raise GroundedActionError(f"object schema at {path} is malformed")
        if len(set(required)) != len(required) or any(not isinstance(x, str) for x in required):
            raise GroundedActionError(f"required fields at {path} must be unique strings")
        missing = sorted(set(required) - set(value))
        if missing:
            raise GroundedActionError(f"{path} is missing required fields: {', '.join(missing)}")
        additional = schema.get("additionalProperties", False)
        for key, child in value.items():
            if key in properties:
                _validate_schema_value(child, properties[key], f"{path}.{key}")
            elif additional is False:
                raise GroundedActionError(f"unexpected field at {path}: {key}")
            elif isinstance(additional, Mapping):
                _validate_schema_value(child, additional, f"{path}.{key}")


def _validate_schema_definition(schema: Mapping[str, Any], path: str = "root") -> None:
    if not isinstance(schema, Mapping) or schema.get("type", "object") != "object":
        raise GroundedActionError("action input schema root must be an object")
    if len(_canonical(_thaw(schema)).encode("utf-8")) > 64_000:
        raise GroundedActionError("action input schema exceeds 64 KB")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, Mapping) or not isinstance(required, (list, tuple)):
        raise GroundedActionError(f"schema at {path} has malformed properties or required")
    allowed_types = {"object", "array", "string", "integer", "number", "boolean", "null"}
    def walk(node: Any, node_path: str) -> None:
        if not isinstance(node, Mapping):
            raise GroundedActionError(f"schema at {node_path} must be an object")
        kind = node.get("type")
        if kind is not None and kind not in allowed_types:
            raise GroundedActionError(f"unsupported schema type at {node_path}: {kind}")
        if "properties" in node:
            if not isinstance(node["properties"], Mapping):
                raise GroundedActionError(f"properties at {node_path} must be an object")
            for key, child in node["properties"].items():
                _text(key, f"property name at {node_path}", 128)
                walk(child, f"{node_path}.{key}")
        if "items" in node:
            walk(node["items"], f"{node_path}[]")
        if isinstance(node.get("additionalProperties"), Mapping):
            walk(node["additionalProperties"], f"{node_path}.*")
    walk(schema, path)


@dataclass(frozen=True)
class GroundedActionDefinition:
    action_id: str
    description: str
    source: str
    input_schema: Mapping[str, Any]
    risk: ActionRisk
    timeout_seconds: float = 8.0
    memory_limit_mb: int = 256
    max_pids: int = 16
    image: str = DEFAULT_IMAGE
    runtime: str = "auto"

    def __post_init__(self) -> None:
        action_id = _text(self.action_id, "action_id", 64)
        if not _SLUG.fullmatch(action_id):
            raise GroundedActionError("action_id must be a lowercase safe identifier")
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "description", _text(self.description, "description", 1024))
        if not isinstance(self.source, str) or not self.source.strip() or len(self.source.encode("utf-8")) > 256_000:
            raise GroundedActionError("action source must be non-empty and no larger than 256 KB")
        try:
            tree = ast.parse(self.source)
        except SyntaxError as exc:
            raise GroundedActionError(f"action source is not valid Python: {exc}") from exc
        entrypoints = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"]
        if len(entrypoints) != 1 or isinstance(entrypoints[0], ast.AsyncFunctionDef):
            raise GroundedActionError("action source must define exactly one synchronous run(inputs) function")
        args = entrypoints[0].args
        if len(args.args) != 1 or args.vararg or args.kwarg or args.kwonlyargs:
            raise GroundedActionError("action entrypoint must accept exactly one positional inputs argument")
        if not isinstance(self.input_schema, Mapping):
            raise GroundedActionError("input_schema must be a JSON Schema object")
        frozen_schema = _freeze(dict(self.input_schema))
        _validate_schema_definition(frozen_schema)
        object.__setattr__(self, "input_schema", frozen_schema)
        object.__setattr__(self, "risk", ActionRisk(self.risk))
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)) or not math.isfinite(float(self.timeout_seconds)) or not 1 <= self.timeout_seconds <= 30:
            raise GroundedActionError("timeout_seconds must be between 1 and 30")
        if isinstance(self.memory_limit_mb, bool) or not isinstance(self.memory_limit_mb, int) or not 64 <= self.memory_limit_mb <= 2048:
            raise GroundedActionError("memory_limit_mb must be between 64 and 2048")
        if isinstance(self.max_pids, bool) or not isinstance(self.max_pids, int) or not 8 <= self.max_pids <= 64:
            raise GroundedActionError("max_pids must be between 8 and 64")
        if not isinstance(self.image, str) or not re.search(r"@sha256:[0-9a-f]{64}\Z", self.image):
            raise GroundedActionError("action container image must be pinned by SHA-256 digest")
        if self.runtime not in {"auto", "docker", "podman"}:
            raise GroundedActionError("runtime must be auto, docker or podman")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id, "description": self.description, "source": self.source,
            "input_schema": _thaw(self.input_schema), "risk": self.risk.value,
            "timeout_seconds": float(self.timeout_seconds), "memory_limit_mb": self.memory_limit_mb,
            "max_pids": self.max_pids, "image": self.image, "runtime": self.runtime,
        }

    @property
    def fingerprint(self) -> str:
        return _digest(_canonical(self.to_dict()))


class GroundedActionRegistry:
    """Host-controlled allow-list. Definitions cannot be silently replaced."""

    def __init__(self) -> None:
        self._definitions: dict[str, GroundedActionDefinition] = {}

    def register(self, definition: GroundedActionDefinition) -> None:
        if not isinstance(definition, GroundedActionDefinition):
            raise TypeError("definition must be GroundedActionDefinition")
        previous = self._definitions.get(definition.action_id)
        if previous is not None and previous.fingerprint != definition.fingerprint:
            raise GroundedActionError(f"action definition already registered with different content: {definition.action_id}")
        self._definitions[definition.action_id] = definition

    def get(self, action_id: str) -> GroundedActionDefinition | None:
        return self._definitions.get(action_id)

    def list(self) -> tuple[GroundedActionDefinition, ...]:
        return tuple(self._definitions[key] for key in sorted(self._definitions))


@dataclass(frozen=True)
class GroundedActionProposal:
    proposal_id: str
    action_id: str
    definition_fingerprint: str
    inputs: Mapping[str, Any]
    input_evidence_bindings: Mapping[str, tuple[str, ...]]
    evidence_report_fingerprint: str
    evidence_observation_ids: tuple[str, ...]
    expected_postconditions: Mapping[str, Any]
    created_by: str
    created_at: str

    def __post_init__(self) -> None:
        _sha(self.proposal_id, "proposal_id")
        _text(self.action_id, "action_id", 64)
        _sha(self.definition_fingerprint, "definition_fingerprint")
        _sha(self.evidence_report_fingerprint, "evidence_report_fingerprint")
        if not isinstance(self.inputs, Mapping):
            raise GroundedActionError("action inputs must be an object")
        frozen_inputs = _freeze(dict(self.inputs))
        if len(_canonical(_thaw(frozen_inputs)).encode("utf-8")) > 64_000:
            raise GroundedActionError("action inputs exceed 64 KB")
        object.__setattr__(self, "inputs", frozen_inputs)
        ids = tuple(_text(value, "evidence_observation_id", 128) for value in self.evidence_observation_ids)
        if not ids or len(ids) > 32 or len(set(ids)) != len(ids):
            raise GroundedActionError("proposal requires 1-32 unique evidence observation IDs")
        object.__setattr__(self, "evidence_observation_ids", ids)
        if not isinstance(self.input_evidence_bindings, Mapping) or set(self.input_evidence_bindings) != set(frozen_inputs):
            raise GroundedActionError("every action input field must have an explicit evidence binding")
        bindings: dict[str, tuple[str, ...]] = {}
        for field, bound_ids in self.input_evidence_bindings.items():
            if not isinstance(bound_ids, (list, tuple)) or not bound_ids or len(bound_ids) > 32:
                raise GroundedActionError(f"input evidence binding for {field} must contain 1-32 observation IDs")
            normalized_ids = tuple(_text(value, "input evidence observation ID", 128) for value in bound_ids)
            if len(set(normalized_ids)) != len(normalized_ids) or not set(normalized_ids).issubset(set(ids)):
                raise GroundedActionError(f"input evidence binding for {field} must use unique selected observation IDs")
            bindings[field] = normalized_ids
        object.__setattr__(self, "input_evidence_bindings", MappingProxyType(bindings))
        if not isinstance(self.expected_postconditions, Mapping) or not self.expected_postconditions:
            raise GroundedActionError("expected_postconditions must be a non-empty JSON object")
        frozen_expected = _freeze(dict(self.expected_postconditions))
        if len(_canonical(_thaw(frozen_expected)).encode("utf-8")) > 16_384:
            raise GroundedActionError("expected postconditions exceed 16 KB")
        object.__setattr__(self, "expected_postconditions", frozen_expected)
        object.__setattr__(self, "created_by", _text(self.created_by, "created_by", 128))
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))
        if self.proposal_id != self._calculate_id():
            raise GroundedActionError("proposal ID does not fingerprint its exact contents")

    def _payload(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "definition_fingerprint": self.definition_fingerprint,
            "inputs": _thaw(self.inputs),
            "input_evidence_bindings": {key: list(value) for key, value in self.input_evidence_bindings.items()},
            "evidence_report_fingerprint": self.evidence_report_fingerprint,
            "evidence_observation_ids": list(self.evidence_observation_ids),
            "expected_postconditions": _thaw(self.expected_postconditions),
            "created_by": self.created_by,
            "created_at": self.created_at,
        }

    def _calculate_id(self) -> str:
        return _digest(_canonical(self._payload()))

    @classmethod
    def create(
        cls, definition: GroundedActionDefinition, *, inputs: Mapping[str, Any],
        evidence_report_fingerprint: str, evidence_observation_ids: tuple[str, ...] | list[str],
        input_evidence_bindings: Mapping[str, tuple[str, ...] | list[str]],
        expected_postconditions: Mapping[str, Any], created_by: str,
        created_at: str | None = None,
    ) -> "GroundedActionProposal":
        if not isinstance(definition, GroundedActionDefinition):
            raise TypeError("definition must be GroundedActionDefinition")
        if not isinstance(inputs, Mapping) or not isinstance(expected_postconditions, Mapping) or not isinstance(input_evidence_bindings, Mapping):
            raise GroundedActionError("inputs, input_evidence_bindings and expected_postconditions must be objects")
        _validate_schema_value(dict(inputs), definition.input_schema, definition.action_id)
        if set(input_evidence_bindings) != set(inputs):
            raise GroundedActionError("every action input field must have an explicit evidence binding")
        normalized_bindings = {
            key: list(value) if isinstance(value, (tuple, list)) else value
            for key, value in input_evidence_bindings.items()
        }
        stamp = _utc(created_at, "created_at")
        payload = {
            "action_id": definition.action_id,
            "definition_fingerprint": definition.fingerprint,
            "inputs": dict(inputs),
            "input_evidence_bindings": normalized_bindings,
            "evidence_report_fingerprint": _sha(evidence_report_fingerprint, "evidence_report_fingerprint"),
            "evidence_observation_ids": list(evidence_observation_ids),
            "expected_postconditions": dict(expected_postconditions),
            "created_by": _text(created_by, "created_by", 128),
            "created_at": stamp,
        }
        return cls(
            _digest(_canonical(payload)), definition.action_id, definition.fingerprint,
            dict(inputs), normalized_bindings, evidence_report_fingerprint, tuple(evidence_observation_ids),
            dict(expected_postconditions), created_by, stamp,
        )

    @property
    def fingerprint(self) -> str:
        return self._calculate_id()


@dataclass(frozen=True)
class ActionCheckResult:
    check_id: str
    verifier_id: str
    decision: ActionCheckDecision
    detail: str
    proposal_id: str
    evidence_report_fingerprint: str
    checked_at: str

    def __post_init__(self) -> None:
        if not _SLUG.fullmatch(_text(self.check_id, "check_id", 64)):
            raise GroundedActionError("check_id must be a safe identifier")
        if not _SLUG.fullmatch(_text(self.verifier_id, "verifier_id", 64)):
            raise GroundedActionError("verifier_id must be a safe identifier")
        object.__setattr__(self, "decision", ActionCheckDecision(self.decision))
        object.__setattr__(self, "detail", _text(self.detail, "detail", 2000))
        _sha(self.proposal_id, "proposal_id")
        _sha(self.evidence_report_fingerprint, "evidence_report_fingerprint")
        object.__setattr__(self, "checked_at", _utc(self.checked_at, "checked_at"))

    @property
    def fingerprint(self) -> str:
        return _digest(_canonical({
            "check_id": self.check_id, "verifier_id": self.verifier_id,
            "decision": self.decision.value, "detail": self.detail,
            "proposal_id": self.proposal_id,
            "evidence_report_fingerprint": self.evidence_report_fingerprint,
            "checked_at": self.checked_at,
        }))


@dataclass(frozen=True)
class ActionVerificationReport:
    proposal_id: str
    definition_fingerprint: str
    evidence_report_fingerprint: str
    checks: tuple[ActionCheckResult, ...]
    verified_at: str

    def __post_init__(self) -> None:
        _sha(self.proposal_id, "proposal_id")
        _sha(self.definition_fingerprint, "definition_fingerprint")
        _sha(self.evidence_report_fingerprint, "evidence_report_fingerprint")
        checks = tuple(self.checks)
        if len(checks) != 2 or any(not isinstance(item, ActionCheckResult) for item in checks):
            raise GroundedActionError("verification requires exactly two typed independent check results")
        if len({item.check_id for item in checks}) != 2 or len({item.verifier_id for item in checks}) != 2:
            raise GroundedActionError("verification requires exactly two distinct independent check identities")
        if any(item.proposal_id != self.proposal_id or item.evidence_report_fingerprint != self.evidence_report_fingerprint for item in checks):
            raise GroundedActionError("independent check result is not bound to this proposal and evidence report")
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "verified_at", _utc(self.verified_at, "verified_at"))

    @property
    def passed(self) -> bool:
        return all(item.decision is ActionCheckDecision.PASS for item in self.checks)

    @property
    def fingerprint(self) -> str:
        return _digest(_canonical({
            "proposal_id": self.proposal_id,
            "definition_fingerprint": self.definition_fingerprint,
            "evidence_report_fingerprint": self.evidence_report_fingerprint,
            "checks": [item.fingerprint for item in self.checks],
            "verified_at": self.verified_at,
        }))


@dataclass(frozen=True)
class HumanActionApproval:
    approval_id: str
    proposal_id: str
    verification_fingerprint: str
    reviewer_id: str
    reason: str
    approved_at: str

    def __post_init__(self) -> None:
        _sha(self.approval_id, "approval_id")
        _sha(self.proposal_id, "proposal_id")
        _sha(self.verification_fingerprint, "verification_fingerprint")
        object.__setattr__(self, "reviewer_id", _text(self.reviewer_id, "reviewer_id", 128))
        object.__setattr__(self, "reason", _text(self.reason, "reason", 2000))
        object.__setattr__(self, "approved_at", _utc(self.approved_at, "approved_at"))

    @property
    def fingerprint(self) -> str:
        return _digest(_canonical({
            "proposal_id": self.proposal_id,
            "verification_fingerprint": self.verification_fingerprint,
            "reviewer_id": self.reviewer_id,
            "reason": self.reason,
            "approved_at": self.approved_at,
        }))


@dataclass(frozen=True)
class ActionSandboxResult:
    outcome: Mapping[str, Any]
    runtime: str
    pid: int
    returncode: int

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, Mapping):
            raise GroundedActionError("sandbox action outcome must be a JSON object")
        frozen = _freeze(dict(self.outcome))
        if len(_canonical(_thaw(frozen)).encode("utf-8")) > 16_384:
            raise GroundedActionError("sandbox action outcome exceeds 16 KB")
        object.__setattr__(self, "outcome", frozen)
        if self.runtime not in {"docker", "podman"}:
            raise GroundedActionError("sandbox action runtime must be docker or podman")
        if isinstance(self.pid, bool) or self.pid != 1:
            raise GroundedActionError("sandbox action evidence must prove container PID 1")
        if isinstance(self.returncode, bool) or self.returncode != 0:
            raise GroundedActionError("sandbox action must return a zero exit code")


class ContainerActionExecutor:
    """One-shot Stage 21 runner using the Stage 19 hardened per-case child harness."""

    def execute(self, definition: GroundedActionDefinition, inputs: Mapping[str, Any]) -> ActionSandboxResult:
        if not isinstance(definition, GroundedActionDefinition):
            raise TypeError("definition must be GroundedActionDefinition")
        _validate_schema_value(dict(inputs), definition.input_schema, definition.action_id)
        selected_runtime = None if definition.runtime == "auto" else definition.runtime
        cases = [{
            "case_id": "stage21-action",
            "inputs": dict(inputs),
            "candidate_expected": None,
        }]
        raw = run_python_regression_in_container(
            definition.source, cases, expected_key="candidate_expected",
            timeout_seconds=float(definition.timeout_seconds),
            memory_limit_mb=definition.memory_limit_mb,
            max_pids=definition.max_pids,
            max_loop_steps=50_000,
            image=definition.image,
            runtime=selected_runtime,
        )
        if not raw.ok:
            raise GroundedActionError(f"container action failed closed: {raw.error}")
        if raw.pid != 1 or raw.returncode != 0:
            raise GroundedActionError("container action lacks valid PID-1/zero-exit evidence")
        prefix = "__MIRROR_STAGE19_RESULTS__"
        matching = [line[len(prefix):] for line in raw.output.splitlines() if line.startswith(prefix)]
        if len(matching) != 1:
            raise GroundedActionError("sandbox did not return exactly one trusted action result manifest")
        try:
            rows = json.loads(matching[0])["results"]
            if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("case_id") != "stage21-action":
                raise GroundedActionError("sandbox action manifest has unexpected case identity")
            row = rows[0]
            if not isinstance(row.get("actual_json"), str) or not row["actual_json"]:
                raise GroundedActionError(f"action candidate failed: {str(row.get('error', 'no result'))[:1000]}")
            outcome = json.loads(row["actual_json"])
            if not isinstance(outcome, dict):
                raise GroundedActionError("action entrypoint must return a JSON object")
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, GroundedActionError):
                raise
            raise GroundedActionError(f"sandbox returned malformed action evidence: {exc}") from exc
        return ActionSandboxResult(outcome, raw.runtime, raw.pid, raw.returncode)


@dataclass(frozen=True)
class ActionFeedback:
    feedback_id: str
    proposal_id: str
    execution_id: str
    status: ActionExecutionStatus
    matched_conditions: tuple[str, ...]
    failed_conditions: tuple[str, ...]
    explanation: str
    recorded_at: str

    def __post_init__(self) -> None:
        _sha(self.feedback_id, "feedback_id")
        _sha(self.proposal_id, "proposal_id")
        _sha(self.execution_id, "execution_id")
        object.__setattr__(self, "status", ActionExecutionStatus(self.status))
        object.__setattr__(self, "matched_conditions", tuple(self.matched_conditions))
        object.__setattr__(self, "failed_conditions", tuple(self.failed_conditions))
        object.__setattr__(self, "explanation", _text(self.explanation, "explanation", 2000))
        object.__setattr__(self, "recorded_at", _utc(self.recorded_at, "recorded_at"))


@dataclass(frozen=True)
class ActionExecutionReport:
    execution_id: str
    proposal_id: str
    verification_fingerprint: str
    status: ActionExecutionStatus
    observed_outcome: Mapping[str, Any]
    expected_postconditions: Mapping[str, Any]
    matched_conditions: tuple[str, ...]
    failed_conditions: tuple[str, ...]
    runtime: str
    pid: int
    returncode: int
    error: str
    completed_at: str
    feedback: ActionFeedback

    def __post_init__(self) -> None:
        _sha(self.execution_id, "execution_id")
        _sha(self.proposal_id, "proposal_id")
        _sha(self.verification_fingerprint, "verification_fingerprint")
        object.__setattr__(self, "status", ActionExecutionStatus(self.status))
        object.__setattr__(self, "observed_outcome", _freeze(dict(self.observed_outcome)))
        object.__setattr__(self, "expected_postconditions", _freeze(dict(self.expected_postconditions)))
        object.__setattr__(self, "matched_conditions", tuple(self.matched_conditions))
        object.__setattr__(self, "failed_conditions", tuple(self.failed_conditions))
        object.__setattr__(self, "error", str(self.error)[:2000])
        object.__setattr__(self, "completed_at", _utc(self.completed_at, "completed_at"))
        if self.feedback.execution_id != self.execution_id or self.feedback.proposal_id != self.proposal_id:
            raise GroundedActionError("feedback is not bound to its execution report")

    @property
    def fingerprint(self) -> str:
        return _digest(_canonical({
            "execution_id": self.execution_id, "proposal_id": self.proposal_id,
            "verification_fingerprint": self.verification_fingerprint,
            "status": self.status.value, "observed_outcome": _thaw(self.observed_outcome),
            "expected_postconditions": _thaw(self.expected_postconditions),
            "matched_conditions": list(self.matched_conditions),
            "failed_conditions": list(self.failed_conditions),
            "runtime": self.runtime, "pid": self.pid, "returncode": self.returncode,
            "error": self.error, "completed_at": self.completed_at,
            "feedback_id": self.feedback.feedback_id,
        }))


class GroundedActionEngine:
    """Evidence gate, independent checks, human approval and bounded sandbox execution."""

    VERSION = "stage21-grounded-action-v1"

    def __init__(
        self,
        registry: GroundedActionRegistry,
        executor: ContainerActionExecutor | Any | None = None,
        *,
        allow_test_executor: bool = False,
    ) -> None:
        if not isinstance(registry, GroundedActionRegistry):
            raise TypeError("registry must be GroundedActionRegistry")
        if executor is not None and type(executor) is not ContainerActionExecutor and not allow_test_executor:
            raise GroundedActionError("custom action executors are test-only")
        if allow_test_executor and executor is None:
            raise GroundedActionError("allow_test_executor requires an injected test executor")
        self.registry = registry
        self.executor = executor if executor is not None else ContainerActionExecutor()
        self._verifications: dict[str, ActionVerificationReport] = {}
        self._approvals: dict[str, HumanActionApproval] = {}
        self._executed: dict[str, ActionExecutionReport] = {}
        self._history: list[ActionExecutionReport] = []

    @staticmethod
    def _report_fingerprint(report: GroundingReport) -> str:
        if not isinstance(report, GroundingReport):
            raise GroundedActionError("evidence must be a Stage 20 GroundingReport")
        payload = {
            "assets": [item.to_dict() for item in report.assets],
            "segments": [item.to_dict() for item in report.segments],
            "observations": [item.to_dict() for item in report.observations],
            "conclusions": [item.to_dict() for item in report.conclusions],
            "unknowns": list(report.unknowns),
        }
        actual = _digest(_canonical(payload))
        if actual != report.report_fingerprint:
            raise GroundedActionError("grounding report fingerprint does not match its evidence contents")
        return actual

    @classmethod
    def _check_evidence(cls, proposal: GroundedActionProposal, report: GroundingReport) -> str:
        cls._report_fingerprint(report)
        observations = {item.observation_id: item for item in report.observations}
        assets = {item.source_id: item for item in report.assets}
        valid_conclusions = tuple(
            conclusion for conclusion in report.conclusions
            if conclusion.status is GroundingStatus.VERIFIED_AGREEMENT
        )
        selected = []
        for observation_id in proposal.evidence_observation_ids:
            item = observations.get(observation_id)
            if item is None:
                raise GroundedActionError(f"proposal references missing observation: {observation_id}")
            asset = assets.get(item.source_id)
            if asset is None or asset.sha256 != item.source_sha256 or not asset.integrity_verified:
                raise GroundedActionError("evidence source/observation integrity binding failed")
            if not item.origin_verified:
                raise GroundedActionError(f"observation origin is not authenticated: {observation_id}")
            selected.append(item)
        selected_ids = {item.observation_id for item in selected}
        if not any(selected_ids.issubset(set(conclusion.observation_ids)) for conclusion in valid_conclusions):
            raise GroundedActionError("all selected action evidence must belong to one Stage 20 verified_agreement conclusion")
        if len({item.modality.value for item in selected}) < 2:
            raise GroundedActionError("selected evidence must span at least two independent modalities")
        if len({item.source_id for item in selected}) < 2:
            raise GroundedActionError("selected evidence must reference at least two distinct sources")
        for input_field, observation_ids in proposal.input_evidence_bindings.items():
            bound = [observations[item_id] for item_id in observation_ids]
            if len({item.modality.value for item in bound}) < 2 or len({item.source_id for item in bound}) < 2:
                raise GroundedActionError(f"input {input_field} requires at least two independent modality/source observations")
            if any(item.value != proposal.inputs[input_field] for item in bound):
                raise GroundedActionError(f"input {input_field} does not exactly match its cited grounded observation values")
            if len({(item.dimension, item.canonical_unit) for item in bound}) != 1:
                raise GroundedActionError(f"input {input_field} is bound to incompatible units or value dimensions")
        return "each action input is exactly bound to authenticated observations from distinct modalities and sources"

    @staticmethod
    def _check_action_policy(proposal: GroundedActionProposal, definition: GroundedActionDefinition) -> str:
        if proposal.action_id != definition.action_id or proposal.definition_fingerprint != definition.fingerprint:
            raise GroundedActionError("proposal is not bound to the registered action definition")
        _validate_schema_value(_thaw(proposal.inputs), definition.input_schema, definition.action_id)
        if not proposal.expected_postconditions:
            raise GroundedActionError("at least one explicit postcondition is required")
        if definition.risk is ActionRisk.CONSEQUENTIAL and not proposal.created_by:
            raise GroundedActionError("consequential actions require an attributable proposer")
        return f"action contract, typed input schema, risk={definition.risk.value}, resource limits and postconditions pass"

    def verify(self, proposal: GroundedActionProposal, report: GroundingReport) -> ActionVerificationReport:
        if not isinstance(proposal, GroundedActionProposal):
            raise TypeError("proposal must be GroundedActionProposal")
        definition = self.registry.get(proposal.action_id)
        stamp = datetime.now(timezone.utc).isoformat()
        checks: list[ActionCheckResult] = []
        report_fingerprint = proposal.evidence_report_fingerprint
        try:
            observed_fp = self._report_fingerprint(report)
            if observed_fp != proposal.evidence_report_fingerprint:
                raise GroundedActionError("proposal references a different grounding report")
            detail = self._check_evidence(proposal, report)
            decision = ActionCheckDecision.PASS
        except Exception as exc:
            detail = f"{type(exc).__name__}: {str(exc)[:1500]}"
            decision = ActionCheckDecision.FAIL
        checks.append(ActionCheckResult(
            "evidence_provenance", "mirror.evidence_provenance", decision, detail,
            proposal.proposal_id, report_fingerprint, stamp,
        ))
        try:
            if definition is None:
                raise GroundedActionError("action is not registered in the host allow-list")
            detail = self._check_action_policy(proposal, definition)
            decision = ActionCheckDecision.PASS
        except Exception as exc:
            detail = f"{type(exc).__name__}: {str(exc)[:1500]}"
            decision = ActionCheckDecision.FAIL
        checks.append(ActionCheckResult(
            "action_contract", "mirror.action_contract", decision, detail,
            proposal.proposal_id, report_fingerprint, stamp,
        ))
        result = ActionVerificationReport(
            proposal.proposal_id,
            proposal.definition_fingerprint,
            report_fingerprint,
            tuple(checks),
            stamp,
        )
        self._verifications[result.fingerprint] = result
        return result

    def approve(
        self, proposal: GroundedActionProposal, verification: ActionVerificationReport,
        *, reviewer_id: str, reason: str,
    ) -> HumanActionApproval:
        if not isinstance(proposal, GroundedActionProposal) or not isinstance(verification, ActionVerificationReport):
            raise TypeError("proposal and verification have invalid types")
        definition = self.registry.get(proposal.action_id)
        if definition is None or definition.risk is not ActionRisk.CONSEQUENTIAL:
            raise GroundedActionError("human approval is reserved for consequential actions")
        registered = self._verifications.get(verification.fingerprint)
        if registered != verification or verification.proposal_id != proposal.proposal_id or not verification.passed:
            raise GroundedActionError("approval requires this engine's passing verification for the exact proposal")
        if reviewer_id == proposal.created_by:
            raise GroundedActionError("proposal author cannot approve a consequential action")
        if proposal.proposal_id in self._approvals:
            raise GroundedActionError("proposal already has a human approval decision")
        reviewer = _text(reviewer_id, "reviewer_id", 128)
        rationale = _text(reason, "reason", 2000)
        stamp = datetime.now(timezone.utc).isoformat()
        body = {
            "proposal_id": proposal.proposal_id,
            "verification_fingerprint": verification.fingerprint,
            "reviewer_id": reviewer,
            "reason": rationale,
            "approved_at": stamp,
        }
        approval = HumanActionApproval(
            _digest(_canonical(body)), proposal.proposal_id, verification.fingerprint,
            reviewer, rationale, stamp,
        )
        self._approvals[proposal.proposal_id] = approval
        return approval

    def execute(
        self, proposal: GroundedActionProposal, verification: ActionVerificationReport,
        *, approval: HumanActionApproval | None = None,
    ) -> ActionExecutionReport:
        if not isinstance(proposal, GroundedActionProposal) or not isinstance(verification, ActionVerificationReport):
            raise TypeError("proposal and verification have invalid types")
        definition = self.registry.get(proposal.action_id)
        if definition is None:
            raise GroundedActionError("action definition is no longer registered")
        registered = self._verifications.get(verification.fingerprint)
        if registered != verification or verification.proposal_id != proposal.proposal_id or not verification.passed:
            raise GroundedActionError("execution requires this engine's passing independent verification")
        if verification.definition_fingerprint != definition.fingerprint or definition.fingerprint != proposal.definition_fingerprint:
            raise GroundedActionError("action definition changed after independent verification")
        if proposal.proposal_id in self._executed:
            raise GroundedActionError("proposal execution is single-use")
        if definition.risk is ActionRisk.CONSEQUENTIAL:
            trusted_approval = self._approvals.get(proposal.proposal_id)
            if approval is None or trusted_approval != approval or approval.verification_fingerprint != verification.fingerprint:
                raise GroundedActionError("consequential action requires prior independent human approval")
        elif approval is not None:
            raise GroundedActionError("unexpected human approval token for non-consequential action")

        stamp = datetime.now(timezone.utc).isoformat()
        runtime, pid, returncode, error = "", 0, -1, ""
        outcome: Mapping[str, Any] = {}
        try:
            raw = self.executor.execute(definition, _thaw(proposal.inputs))
            if not isinstance(raw, ActionSandboxResult):
                raise GroundedActionError("executor must return ActionSandboxResult")
            runtime, pid, returncode = raw.runtime, raw.pid, raw.returncode
            outcome = raw.outcome
            matched = tuple(
                key for key, value in proposal.expected_postconditions.items()
                if key in outcome and outcome[key] == value
            )
            failed = tuple(sorted(set(proposal.expected_postconditions) - set(matched)))
            status = ActionExecutionStatus.SUCCEEDED if not failed else ActionExecutionStatus.POSTCONDITION_MISMATCH
            explanation = (
                "all declared postconditions matched the sandbox observation"
                if not failed else "observed outcome did not satisfy every predicted postcondition; trusted knowledge was not modified"
            )
        except Exception as exc:
            status = ActionExecutionStatus.FAILED
            explanation = f"fail-closed sandbox execution: {type(exc).__name__}: {str(exc)[:1500]}"
            error = explanation
            matched, failed = (), tuple(sorted(proposal.expected_postconditions))
        payload = {
            "proposal_id": proposal.proposal_id,
            "verification_fingerprint": verification.fingerprint,
            "status": status.value,
            "outcome": _thaw(outcome),
            "matched": list(matched), "failed": list(failed),
            "runtime": runtime, "pid": pid, "returncode": returncode,
            "error": error, "completed_at": stamp,
        }
        execution_id = _digest(_canonical(payload))
        feedback_body = {
            "proposal_id": proposal.proposal_id, "execution_id": execution_id,
            "status": status.value, "matched": list(matched), "failed": list(failed),
            "explanation": explanation, "recorded_at": stamp,
        }
        feedback = ActionFeedback(
            _digest(_canonical(feedback_body)), proposal.proposal_id, execution_id,
            status, matched, failed, explanation, stamp,
        )
        result = ActionExecutionReport(
            execution_id, proposal.proposal_id, verification.fingerprint, status,
            outcome, _thaw(proposal.expected_postconditions), matched, failed,
            runtime, pid, returncode, error, stamp, feedback,
        )
        self._executed[proposal.proposal_id] = result
        self._history.append(result)
        return result

    @property
    def history(self) -> tuple[ActionExecutionReport, ...]:
        return tuple(self._history)

    def get_feedback(self, proposal_id: str) -> ActionFeedback | None:
        report = self._executed.get(proposal_id)
        return report.feedback if report is not None else None
