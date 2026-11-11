"""Stage 19: controlled, container-isolated self-improvement with human approval.

Candidates are immutable source artifacts evaluated against a frozen baseline and
regression/improvement suite. Baseline and candidate run in separate fresh Docker /
Podman containers. Passing yields a proposal for human review only; it never changes
the production registry or repository automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .container_sandbox import DEFAULT_IMAGE, run_python_regression_in_container


class SelfImprovementError(ValueError):
    """Invalid proposal, frozen suite, execution evidence, or approval request."""


class ChangeKind(str, Enum):
    CODE = "code"
    RULE = "rule"


class RegressionCaseKind(str, Enum):
    REGRESSION = "regression"
    IMPROVEMENT = "improvement"


class EvaluationStatus(str, Enum):
    REJECTED = "rejected"
    READY_FOR_HUMAN_REVIEW = "ready_for_human_review"


class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise SelfImprovementError(f"value must be finite JSON data: {exc}") from exc


def _freeze_json(value: Any) -> Any:
    encoded = _canonical(value)
    normalized = json.loads(encoded)
    def freeze(item: Any) -> Any:
        if isinstance(item, dict):
            return MappingProxyType({key: freeze(child) for key, child in item.items()})
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        return item
    return freeze(normalized)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


def _required_text(value: str, field: str, maximum: int = 4_000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SelfImprovementError(f"{field} must be a non-empty string")
    value = value.strip()
    if len(value) > maximum:
        raise SelfImprovementError(f"{field} exceeds {maximum} characters")
    return value


def _require_sha256(value: str, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise SelfImprovementError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class RegressionCase:
    case_id: str
    inputs: Mapping[str, Any]
    baseline_expected: Any
    candidate_expected: Any
    kind: RegressionCaseKind = RegressionCaseKind.REGRESSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _required_text(self.case_id, "case_id", 128))
        object.__setattr__(self, "kind", RegressionCaseKind(self.kind))
        if not isinstance(self.inputs, Mapping):
            raise SelfImprovementError("regression inputs must be a JSON object")
        object.__setattr__(self, "inputs", _freeze_json(dict(self.inputs)))
        baseline_json = _canonical(self.baseline_expected)
        candidate_json = _canonical(self.candidate_expected)
        object.__setattr__(self, "baseline_expected", _freeze_json(self.baseline_expected))
        object.__setattr__(self, "candidate_expected", _freeze_json(self.candidate_expected))
        if self.kind is RegressionCaseKind.REGRESSION and baseline_json != candidate_json:
            raise SelfImprovementError("regression cases must preserve the baseline expected output")
        if self.kind is RegressionCaseKind.IMPROVEMENT and baseline_json == candidate_json:
            raise SelfImprovementError("improvement cases must define a changed expected output")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "inputs": _thaw_json(self.inputs),
            "baseline_expected": _thaw_json(self.baseline_expected),
            "candidate_expected": _thaw_json(self.candidate_expected),
            "kind": self.kind.value,
        }


@dataclass(frozen=True)
class FrozenRegressionSuite:
    suite_id: str
    baseline_revision: str
    environment_sha256: str
    cases: tuple[RegressionCase, ...]
    container_runtime: str = "docker"
    container_image: str = DEFAULT_IMAGE
    timeout_seconds: float = 8.0
    memory_limit_mb: int = 256
    max_pids: int = 32
    version: str = "stage19-regression-suite-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "suite_id", _required_text(self.suite_id, "suite_id", 128))
        if not isinstance(self.baseline_revision, str) or not _REVISION.fullmatch(self.baseline_revision):
            raise SelfImprovementError("baseline_revision must be a 40- or 64-character lowercase Git revision")
        _require_sha256(self.environment_sha256, "environment_sha256")
        if self.container_runtime not in {"docker", "podman"}:
            raise SelfImprovementError("container_runtime must be docker or podman")
        if not isinstance(self.container_image, str) or not re.search(r"@sha256:[0-9a-f]{64}\Z", self.container_image):
            raise SelfImprovementError("container_image must use an immutable SHA-256 image digest")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)) or not math.isfinite(float(self.timeout_seconds)) or not 1 <= self.timeout_seconds <= 60:
            raise SelfImprovementError("suite timeout_seconds must be between 1 and 60")
        if isinstance(self.memory_limit_mb, bool) or not isinstance(self.memory_limit_mb, int) or not 64 <= self.memory_limit_mb <= 8192:
            raise SelfImprovementError("suite memory_limit_mb must be between 64 and 8192")
        if isinstance(self.max_pids, bool) or not isinstance(self.max_pids, int) or not 8 <= self.max_pids <= 256:
            raise SelfImprovementError("suite max_pids must be between 8 and 256")
        expected_environment = ContainerRegressionExecutor.environment_fingerprint(
            runtime=self.container_runtime,
            image=self.container_image,
            timeout_seconds=float(self.timeout_seconds),
            memory_limit_mb=self.memory_limit_mb,
            max_pids=self.max_pids,
        )
        if self.environment_sha256 != expected_environment:
            raise SelfImprovementError("environment fingerprint does not match the pinned runtime, image and resource limits")
        bounded_cases = []
        for item in self.cases:
            if len(bounded_cases) >= 250:
                raise SelfImprovementError("frozen regression suite exceeds 250 cases")
            bounded_cases.append(item)
        cases = tuple(bounded_cases)
        if len(cases) < 4 or any(not isinstance(item, RegressionCase) for item in cases):
            raise SelfImprovementError("suite requires at least four typed cases")
        ids = tuple(item.case_id for item in cases)
        if len(set(ids)) != len(ids):
            raise SelfImprovementError("regression case IDs must be unique")
        regression_count = sum(item.kind is RegressionCaseKind.REGRESSION for item in cases)
        improvement_count = sum(item.kind is RegressionCaseKind.IMPROVEMENT for item in cases)
        if regression_count < 3 or improvement_count < 1:
            raise SelfImprovementError("suite requires at least three regression cases and one improvement case")
        if len(_canonical([item.to_dict() for item in cases]).encode("utf-8")) > 512_000:
            raise SelfImprovementError("frozen regression suite exceeds 512 KB")
        object.__setattr__(self, "cases", cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "baseline_revision": self.baseline_revision,
            "environment_sha256": self.environment_sha256,
            "container_runtime": self.container_runtime,
            "container_image": self.container_image,
            "timeout_seconds": float(self.timeout_seconds),
            "memory_limit_mb": self.memory_limit_mb,
            "max_pids": self.max_pids,
            "cases": [item.to_dict() for item in self.cases],
            "version": self.version,
        }

    @property
    def fingerprint(self) -> str:
        return _digest_text(_canonical(self.to_dict()))

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(item.case_id for item in self.cases)


@dataclass(frozen=True)
class ImprovementProposal:
    proposal_id: str
    title: str
    rationale: str
    artifact_path: str
    change_kind: ChangeKind
    created_by: str
    baseline_revision: str
    baseline_source: str
    candidate_source: str
    baseline_artifact_sha256: str
    candidate_artifact_sha256: str
    regression_suite_fingerprint: str

    def __post_init__(self) -> None:
        _require_sha256(self.proposal_id, "proposal_id")
        object.__setattr__(self, "title", _required_text(self.title, "title", 256))
        object.__setattr__(self, "rationale", _required_text(self.rationale, "rationale", 4_000))
        object.__setattr__(self, "created_by", _required_text(self.created_by, "created_by", 128))
        object.__setattr__(self, "change_kind", ChangeKind(self.change_kind))
        if not isinstance(self.artifact_path, str) or not self.artifact_path or self.artifact_path.startswith("/"):
            raise SelfImprovementError("artifact_path must be a normalized relative path")
        if any(part in {"", ".", ".."} for part in self.artifact_path.split("/")) or re.match(r"^[A-Za-z]:", self.artifact_path):
            raise SelfImprovementError("artifact_path must not contain traversal or drive prefixes")
        if not isinstance(self.baseline_revision, str) or not _REVISION.fullmatch(self.baseline_revision):
            raise SelfImprovementError("baseline_revision must be a 40- or 64-character lowercase Git revision")
        if not isinstance(self.baseline_source, str) or not self.baseline_source.strip():
            raise SelfImprovementError("baseline_source must be non-empty source code")
        if not isinstance(self.candidate_source, str) or not self.candidate_source.strip():
            raise SelfImprovementError("candidate_source must be non-empty source code")
        if len(self.baseline_source.encode("utf-8")) > 512_000 or len(self.candidate_source.encode("utf-8")) > 512_000:
            raise SelfImprovementError("baseline and candidate source are each limited to 512 KB")
        if self.baseline_artifact_sha256 != _digest_text(self.baseline_source):
            raise SelfImprovementError("baseline artifact digest does not match its source")
        if self.candidate_artifact_sha256 != _digest_text(self.candidate_source):
            raise SelfImprovementError("candidate artifact digest does not match its source")
        _require_sha256(self.regression_suite_fingerprint, "regression_suite_fingerprint")
        expected_id = _digest_text(_canonical({
            "title": self.title,
            "rationale": self.rationale,
            "artifact_path": self.artifact_path,
            "change_kind": self.change_kind.value,
            "created_by": self.created_by,
            "baseline_revision": self.baseline_revision,
            "baseline_artifact_sha256": self.baseline_artifact_sha256,
            "candidate_artifact_sha256": self.candidate_artifact_sha256,
            "regression_suite_fingerprint": self.regression_suite_fingerprint,
        }))
        if expected_id != self.proposal_id:
            raise SelfImprovementError("proposal ID does not fingerprint the exact proposal contents")

    @classmethod
    def create(
        cls,
        *,
        title: str,
        rationale: str,
        artifact_path: str,
        change_kind: ChangeKind,
        created_by: str,
        baseline_revision: str,
        baseline_source: str,
        candidate_source: str,
        suite: FrozenRegressionSuite,
    ) -> "ImprovementProposal":
        if not isinstance(suite, FrozenRegressionSuite):
            raise TypeError("suite must be FrozenRegressionSuite")
        title = _required_text(title, "title", 256)
        rationale = _required_text(rationale, "rationale", 4_000)
        artifact_path = _required_text(artifact_path, "artifact_path", 512).replace("\\", "/")
        if artifact_path.startswith("/") or re.match(r"^[A-Za-z]:", artifact_path) or any(
            part in {"", ".", ".."} for part in artifact_path.split("/")
        ):
            raise SelfImprovementError("artifact_path must be a normalized relative path without traversal")
        created_by = _required_text(created_by, "created_by", 128)
        if not isinstance(baseline_revision, str) or not _REVISION.fullmatch(baseline_revision):
            raise SelfImprovementError("baseline_revision must be a 40- or 64-character lowercase Git revision")
        if baseline_revision != suite.baseline_revision:
            raise SelfImprovementError("proposal baseline revision does not match the frozen suite")
        if not isinstance(baseline_source, str) or not baseline_source.strip():
            raise SelfImprovementError("baseline_source must be non-empty source code")
        if not isinstance(candidate_source, str) or not candidate_source.strip():
            raise SelfImprovementError("candidate_source must be non-empty source code")
        if len(baseline_source.encode("utf-8")) > 512_000 or len(candidate_source.encode("utf-8")) > 512_000:
            raise SelfImprovementError("baseline and candidate source are each limited to 512 KB")
        kind = ChangeKind(change_kind)
        base_hash = _digest_text(baseline_source)
        candidate_hash = _digest_text(candidate_source)
        payload = {
            "title": title,
            "rationale": rationale,
            "artifact_path": artifact_path,
            "change_kind": kind.value,
            "created_by": created_by,
            "baseline_revision": baseline_revision,
            "baseline_artifact_sha256": base_hash,
            "candidate_artifact_sha256": candidate_hash,
            "regression_suite_fingerprint": suite.fingerprint,
        }
        proposal_id = _digest_text(_canonical(payload))
        return cls(
            proposal_id, title, rationale, artifact_path, kind, created_by,
            baseline_revision, baseline_source, candidate_source, base_hash,
            candidate_hash, suite.fingerprint,
        )


@dataclass(frozen=True)
class CaseExecutionResult:
    case_id: str
    passed: bool
    actual_json: str = ""
    error: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _required_text(self.case_id, "case_id", 128))
        if not isinstance(self.passed, bool):
            raise SelfImprovementError("case passed must be boolean")
        if not isinstance(self.actual_json, str) or len(self.actual_json) > 2_000:
            raise SelfImprovementError("actual_json must be text no longer than 2000 characters")
        if not isinstance(self.error, str) or len(self.error) > 1_000:
            raise SelfImprovementError("case error must be text no longer than 1000 characters")


@dataclass(frozen=True)
class ExecutionEvidence:
    artifact_sha256: str
    suite_fingerprint: str
    environment_sha256: str
    expected_key: str
    runtime: str
    returncode: int
    pid: int
    results: tuple[CaseExecutionResult, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.artifact_sha256, "artifact_sha256")
        _require_sha256(self.suite_fingerprint, "suite_fingerprint")
        _require_sha256(self.environment_sha256, "environment_sha256")
        if self.expected_key not in {"baseline_expected", "candidate_expected"}:
            raise SelfImprovementError("execution expected_key is invalid")
        if self.runtime not in {"docker", "podman"}:
            raise SelfImprovementError("execution runtime must be docker or podman")
        if isinstance(self.returncode, bool) or not isinstance(self.returncode, int):
            raise SelfImprovementError("execution returncode must be an integer")
        if isinstance(self.pid, bool) or not isinstance(self.pid, int) or self.pid < 1:
            raise SelfImprovementError("execution PID must be a positive integer")
        results = tuple(self.results)
        if any(not isinstance(item, CaseExecutionResult) for item in results):
            raise TypeError("results must contain CaseExecutionResult values")
        if len({item.case_id for item in results}) != len(results):
            raise SelfImprovementError("execution contains duplicate case IDs")
        object.__setattr__(self, "results", results)

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and bool(self.results) and all(item.passed for item in self.results)

    @property
    def result_fingerprint(self) -> str:
        rows = [
            {"case_id": item.case_id, "passed": item.passed, "actual_json": item.actual_json, "error": item.error}
            for item in self.results
        ]
        return _digest_text(_canonical(rows))


class ContainerRegressionExecutor:
    """Runs a candidate in a fresh Docker/Podman sandbox; never downgrades to a process runner."""

    @staticmethod
    def environment_fingerprint(
        *,
        runtime: str = "docker",
        image: str = DEFAULT_IMAGE,
        timeout_seconds: float = 8.0,
        memory_limit_mb: int = 256,
        max_pids: int = 32,
    ) -> str:
        if runtime not in {"docker", "podman"}:
            raise SelfImprovementError("runtime must be docker or podman")
        return _digest_text(_canonical({
            "runtime": runtime,
            "image": image,
            "timeout_seconds": float(timeout_seconds),
            "memory_limit_mb": memory_limit_mb,
            "max_pids": max_pids,
            "container_policy": "network-none-readonly-root-nonroot-drop-all-caps-no-new-privileges-v1",
            "stage19_runner": "frozen-regression-harness-v1",
            "python": "3.11",
        }))

    def __init__(
        self,
        *,
        timeout_seconds: float = 8.0,
        memory_limit_mb: int = 256,
        max_pids: int = 32,
        image: str | None = None,
        runtime: str | None = None,
    ) -> None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not 1 <= timeout_seconds <= 60:
            raise SelfImprovementError("timeout_seconds must be between 1 and 60")
        if isinstance(memory_limit_mb, bool) or not isinstance(memory_limit_mb, int) or not 64 <= memory_limit_mb <= 8192:
            raise SelfImprovementError("memory_limit_mb must be between 64 and 8192")
        if isinstance(max_pids, bool) or not isinstance(max_pids, int) or not 8 <= max_pids <= 256:
            raise SelfImprovementError("max_pids must be between 8 and 256")
        if runtime is not None and runtime not in {"docker", "podman"}:
            raise SelfImprovementError("runtime must be docker or podman")
        self.timeout_seconds = float(timeout_seconds)
        self.memory_limit_mb = memory_limit_mb
        self.max_pids = max_pids
        self.image = image
        self.runtime = runtime

    def execute(
        self,
        source: str,
        suite: FrozenRegressionSuite,
        *,
        expected_key: str,
    ) -> ExecutionEvidence:
        if not isinstance(suite, FrozenRegressionSuite):
            raise TypeError("suite must be FrozenRegressionSuite")
        if expected_key not in {"baseline_expected", "candidate_expected"}:
            raise SelfImprovementError("expected_key is invalid")
        selected_runtime = self.runtime or suite.container_runtime
        selected_image = self.image or suite.container_image
        if (
            selected_runtime != suite.container_runtime
            or selected_image != suite.container_image
            or self.timeout_seconds != float(suite.timeout_seconds)
            or self.memory_limit_mb != suite.memory_limit_mb
            or self.max_pids != suite.max_pids
        ):
            raise SelfImprovementError("executor settings differ from the frozen suite environment")
        raw = run_python_regression_in_container(
            source,
            [item.to_dict() for item in suite.cases],
            expected_key=expected_key,
            timeout_seconds=self.timeout_seconds,
            memory_limit_mb=self.memory_limit_mb,
            max_pids=self.max_pids,
            image=selected_image,
            runtime=selected_runtime,
        )
        if not raw.ok:
            raise SelfImprovementError(f"container execution failed closed: {raw.error}")
        prefix = "__MIRROR_STAGE19_RESULTS__"
        matching = [line[len(prefix):] for line in raw.output.splitlines() if line.startswith(prefix)]
        if len(matching) != 1:
            raise SelfImprovementError("container did not return exactly one Stage 19 result manifest")
        try:
            payload = json.loads(matching[0])
            raw_results = payload["results"]
            results = tuple(
                CaseExecutionResult(
                    case_id=item["case_id"],
                    passed=item["passed"],
                    actual_json=item.get("actual_json", ""),
                    error=item.get("error", ""),
                )
                for item in raw_results
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SelfImprovementError(f"container returned malformed Stage 19 results: {exc}") from exc
        return ExecutionEvidence(
            artifact_sha256=_digest_text(source),
            suite_fingerprint=suite.fingerprint,
            environment_sha256=suite.environment_sha256,
            expected_key=expected_key,
            runtime=raw.runtime,
            returncode=raw.returncode if raw.returncode is not None else -1,
            pid=raw.pid if raw.pid is not None else 1,
            results=results,
        )


@dataclass(frozen=True)
class SelfImprovementEvaluation:
    evaluation_id: str
    proposal_id: str
    status: EvaluationStatus
    baseline_evidence: ExecutionEvidence | None
    candidate_evidence: ExecutionEvidence | None
    regression_case_ids: tuple[str, ...]
    improvement_case_ids: tuple[str, ...]
    failed_case_ids: tuple[str, ...]
    explanation: str
    evaluated_at: str

    def to_dict(self) -> dict[str, Any]:
        def evidence_dict(value: ExecutionEvidence | None) -> Any:
            if value is None:
                return None
            return {
                "artifact_sha256": value.artifact_sha256,
                "suite_fingerprint": value.suite_fingerprint,
                "environment_sha256": value.environment_sha256,
                "expected_key": value.expected_key,
                "runtime": value.runtime,
                "returncode": value.returncode,
                "pid": value.pid,
                "passed": value.passed,
                "result_fingerprint": value.result_fingerprint,
                "results": [
                    {"case_id": item.case_id, "passed": item.passed, "actual_json": item.actual_json, "error": item.error}
                    for item in value.results
                ],
            }
        return {
            "evaluation_id": self.evaluation_id,
            "proposal_id": self.proposal_id,
            "status": self.status.value,
            "baseline_evidence": evidence_dict(self.baseline_evidence),
            "candidate_evidence": evidence_dict(self.candidate_evidence),
            "regression_case_ids": list(self.regression_case_ids),
            "improvement_case_ids": list(self.improvement_case_ids),
            "failed_case_ids": list(self.failed_case_ids),
            "explanation": self.explanation,
            "evaluated_at": self.evaluated_at,
        }


class ControlledSelfImprovementEngine:
    """Fail-closed candidate evaluator and explicit human approval gate."""

    def __init__(self, executor: ContainerRegressionExecutor | Any | None = None) -> None:
        self.executor = executor or ContainerRegressionExecutor()
        self._verified_evaluations: dict[str, SelfImprovementEvaluation] = {}
        self._decisions_by_evaluation: dict[str, HumanApprovalRecord] = {}

    def _remember(self, evaluation: SelfImprovementEvaluation) -> SelfImprovementEvaluation:
        self._verified_evaluations[evaluation.evaluation_id] = evaluation
        return evaluation

    @staticmethod
    def _validate_evidence(
        evidence: ExecutionEvidence,
        *,
        source: str,
        suite: FrozenRegressionSuite,
        expected_key: str,
    ) -> tuple[str, ...]:
        if not isinstance(evidence, ExecutionEvidence):
            raise SelfImprovementError("executor must return ExecutionEvidence")
        if evidence.artifact_sha256 != _digest_text(source):
            raise SelfImprovementError("execution evidence is not bound to the exact artifact")
        if evidence.suite_fingerprint != suite.fingerprint:
            raise SelfImprovementError("execution evidence uses a different frozen regression suite")
        if evidence.environment_sha256 != suite.environment_sha256:
            raise SelfImprovementError("execution evidence environment does not match the frozen suite")
        if evidence.expected_key != expected_key:
            raise SelfImprovementError("execution evidence is bound to the wrong baseline/candidate expectation")
        if set(item.case_id for item in evidence.results) != set(suite.case_ids):
            raise SelfImprovementError("execution evidence must contain every frozen case exactly once")
        if evidence.returncode != 0:
            raise SelfImprovementError("sandbox returned a non-zero exit code")
        return tuple(item.case_id for item in evidence.results if not item.passed)

    def evaluate(
        self,
        proposal: ImprovementProposal,
        suite: FrozenRegressionSuite,
    ) -> SelfImprovementEvaluation:
        if not isinstance(proposal, ImprovementProposal) or not isinstance(suite, FrozenRegressionSuite):
            raise TypeError("proposal and suite have invalid types")
        started = datetime.now(timezone.utc).isoformat()
        if proposal.regression_suite_fingerprint != suite.fingerprint:
            return self._remember(self._rejected(proposal, None, None, (), "proposal is not bound to this frozen regression suite", started))
        if proposal.baseline_revision != suite.baseline_revision:
            return self._remember(self._rejected(proposal, None, None, (), "proposal baseline revision drifted from the frozen suite", started))
        if proposal.baseline_artifact_sha256 != _digest_text(proposal.baseline_source):
            return self._remember(self._rejected(proposal, None, None, (), "baseline source digest mismatch", started))
        if proposal.candidate_artifact_sha256 != _digest_text(proposal.candidate_source):
            return self._remember(self._rejected(proposal, None, None, (), "candidate source digest mismatch", started))

        baseline_evidence = None
        candidate_evidence = None
        try:
            baseline_evidence = self.executor.execute(
                proposal.baseline_source, suite, expected_key="baseline_expected",
            )
            baseline_failures = self._validate_evidence(
                baseline_evidence, source=proposal.baseline_source,
                suite=suite, expected_key="baseline_expected",
            )
            if baseline_failures or not baseline_evidence.passed:
                return self._remember(self._rejected(
                    proposal, baseline_evidence, None, baseline_failures,
                    "frozen baseline did not pass every regression and improvement control; candidate comparison is invalid",
                    started,
                ))
            candidate_evidence = self.executor.execute(
                proposal.candidate_source, suite, expected_key="candidate_expected",
            )
            failed = self._validate_evidence(
                candidate_evidence, source=proposal.candidate_source,
                suite=suite, expected_key="candidate_expected",
            )
            if failed or not candidate_evidence.passed:
                return self._remember(self._rejected(
                    proposal, baseline_evidence, candidate_evidence, failed,
                    "candidate failed one or more frozen regression/improvement cases; deployment is blocked",
                    started,
                ))
            if baseline_evidence.runtime != candidate_evidence.runtime:
                raise SelfImprovementError("baseline and candidate were not run with the same container runtime")
            status = EvaluationStatus.READY_FOR_HUMAN_REVIEW
            explanation = (
                "Baseline and candidate passed the same frozen suite in separate containers. "
                "All regression invariants held and every predeclared improvement case passed. "
                "This is review eligibility, not automatic promotion."
            )
            payload = {
                "proposal_id": proposal.proposal_id,
                "status": status.value,
                "baseline": baseline_evidence.result_fingerprint,
                "candidate": candidate_evidence.result_fingerprint,
                "suite": suite.fingerprint,
                "environment": suite.environment_sha256,
                "evaluated_at": started,
            }
            evaluation_id = _digest_text(_canonical(payload))
            return self._remember(SelfImprovementEvaluation(
                evaluation_id, proposal.proposal_id, status, baseline_evidence,
                candidate_evidence,
                tuple(x.case_id for x in suite.cases if x.kind is RegressionCaseKind.REGRESSION),
                tuple(x.case_id for x in suite.cases if x.kind is RegressionCaseKind.IMPROVEMENT),
                (), explanation, started,
            ))
        except Exception as exc:
            return self._remember(self._rejected(
                proposal, baseline_evidence, candidate_evidence, (),
                f"fail-closed evaluation error: {type(exc).__name__}: {str(exc)[:500]}",
                started,
            ))

    @staticmethod
    def _rejected(
        proposal: ImprovementProposal,
        baseline: ExecutionEvidence | None,
        candidate: ExecutionEvidence | None,
        failed: tuple[str, ...],
        explanation: str,
        started: str,
    ) -> SelfImprovementEvaluation:
        payload = {
            "proposal_id": proposal.proposal_id,
            "status": EvaluationStatus.REJECTED.value,
            "baseline": baseline.result_fingerprint if baseline else None,
            "candidate": candidate.result_fingerprint if candidate else None,
            "failed": list(failed),
            "explanation": explanation,
            "evaluated_at": started,
        }
        return SelfImprovementEvaluation(
            _digest_text(_canonical(payload)), proposal.proposal_id,
            EvaluationStatus.REJECTED, baseline, candidate, (), (), tuple(failed),
            explanation, started,
        )

    def record_human_decision(
        self,
        proposal: ImprovementProposal,
        evaluation: SelfImprovementEvaluation,
        *,
        reviewer_id: str,
        approved: bool,
        reason: str,
    ) -> "HumanApprovalRecord":
        if not isinstance(proposal, ImprovementProposal) or not isinstance(evaluation, SelfImprovementEvaluation):
            raise TypeError("proposal and evaluation have invalid types")
        reviewer = _required_text(reviewer_id, "reviewer_id", 128)
        reason = _required_text(reason, "reason", 2_000)
        if not isinstance(approved, bool):
            raise SelfImprovementError("approved must be boolean")
        if evaluation.proposal_id != proposal.proposal_id:
            raise SelfImprovementError("evaluation does not belong to this proposal")
        registered = self._verified_evaluations.get(evaluation.evaluation_id)
        if registered is None or registered != evaluation:
            raise SelfImprovementError("evaluation is not a verified result recorded by this engine instance")
        if evaluation.evaluation_id in self._decisions_by_evaluation:
            raise SelfImprovementError("a human decision has already been recorded for this evaluation")
        if reviewer == proposal.created_by:
            raise SelfImprovementError("proposal authors cannot approve their own changes")
        if approved and evaluation.status is not EvaluationStatus.READY_FOR_HUMAN_REVIEW:
            raise SelfImprovementError("only passing evaluations can receive approval")
        decision = ApprovalDecision.APPROVED if approved else ApprovalDecision.REJECTED
        created_at = datetime.now(timezone.utc).isoformat()
        authorization = None
        if approved:
            if evaluation.candidate_evidence is None:
                raise SelfImprovementError("passing evaluation lacks candidate execution evidence")
            authorization = PromotionAuthorization(
                proposal_id=proposal.proposal_id,
                evaluation_id=evaluation.evaluation_id,
                candidate_artifact_sha256=proposal.candidate_artifact_sha256,
                suite_fingerprint=evaluation.candidate_evidence.suite_fingerprint,
                environment_sha256=evaluation.candidate_evidence.environment_sha256,
                approved_by=reviewer,
                approved_at=created_at,
            )
        payload = {
            "proposal_id": proposal.proposal_id,
            "evaluation_id": evaluation.evaluation_id,
            "decision": decision.value,
            "reviewer_id": reviewer,
            "reason": reason,
            "approved_at": created_at,
            "authorization": authorization.fingerprint if authorization else None,
        }
        record = HumanApprovalRecord(
            approval_id=_digest_text(_canonical(payload)),
            proposal_id=proposal.proposal_id,
            evaluation_id=evaluation.evaluation_id,
            decision=decision,
            reviewer_id=reviewer,
            reason=reason,
            approved_at=created_at,
            authorization=authorization,
        )
        self._decisions_by_evaluation[evaluation.evaluation_id] = record
        return record


@dataclass(frozen=True)
class PromotionAuthorization:
    proposal_id: str
    evaluation_id: str
    candidate_artifact_sha256: str
    suite_fingerprint: str
    environment_sha256: str
    approved_by: str
    approved_at: str

    def __post_init__(self) -> None:
        for name in ("proposal_id", "evaluation_id"):
            _require_sha256(getattr(self, name), name)
        for name in ("candidate_artifact_sha256", "suite_fingerprint", "environment_sha256"):
            _require_sha256(getattr(self, name), name)
        object.__setattr__(self, "approved_by", _required_text(self.approved_by, "approved_by", 128))
        object.__setattr__(self, "approved_at", _required_text(self.approved_at, "approved_at", 64))

    @property
    def fingerprint(self) -> str:
        return _digest_text(_canonical({
            "proposal_id": self.proposal_id,
            "evaluation_id": self.evaluation_id,
            "candidate_artifact_sha256": self.candidate_artifact_sha256,
            "suite_fingerprint": self.suite_fingerprint,
            "environment_sha256": self.environment_sha256,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
        }))

    def authorizes(self, proposal: ImprovementProposal, evaluation: SelfImprovementEvaluation) -> bool:
        return (
            isinstance(proposal, ImprovementProposal)
            and isinstance(evaluation, SelfImprovementEvaluation)
            and evaluation.status is EvaluationStatus.READY_FOR_HUMAN_REVIEW
            and evaluation.candidate_evidence is not None
            and self.proposal_id == proposal.proposal_id == evaluation.proposal_id
            and self.evaluation_id == evaluation.evaluation_id
            and self.candidate_artifact_sha256 == proposal.candidate_artifact_sha256
            and self.candidate_artifact_sha256 == evaluation.candidate_evidence.artifact_sha256
            and self.suite_fingerprint == evaluation.candidate_evidence.suite_fingerprint
            and self.environment_sha256 == evaluation.candidate_evidence.environment_sha256
        )


@dataclass(frozen=True)
class HumanApprovalRecord:
    approval_id: str
    proposal_id: str
    evaluation_id: str
    decision: ApprovalDecision
    reviewer_id: str
    reason: str
    approved_at: str
    authorization: PromotionAuthorization | None = None

    def __post_init__(self) -> None:
        _require_sha256(self.approval_id, "approval_id")
        _require_sha256(self.proposal_id, "proposal_id")
        _require_sha256(self.evaluation_id, "evaluation_id")
        object.__setattr__(self, "decision", ApprovalDecision(self.decision))
        object.__setattr__(self, "reviewer_id", _required_text(self.reviewer_id, "reviewer_id", 128))
        object.__setattr__(self, "reason", _required_text(self.reason, "reason", 2_000))
        if self.decision is ApprovalDecision.APPROVED and not isinstance(self.authorization, PromotionAuthorization):
            raise SelfImprovementError("approved decisions require an exact-artifact promotion authorization")
        if self.decision is ApprovalDecision.REJECTED and self.authorization is not None:
            raise SelfImprovementError("rejected decisions cannot carry promotion authorization")
