"""Container-isolated execution for untrusted code.

Docker/Podman runs with no network, a read-only root filesystem, dropped Linux
capabilities, no-new-privileges, non-root UID, and CPU/memory/PID/file limits.
If no runtime/image is ready, execution fails closed; it never downgrades silently.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import shutil
import subprocess
from typing import Any

from .os_process_sandbox import _CHILD_RUNNER


@dataclass(frozen=True)
class ContainerExecutionResult:
    ok: bool
    output: str
    error: str
    runtime: str
    pid: int | None
    returncode: int | None


DEFAULT_IMAGE = "python:3.11-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39"

_STAGE19_HARNESS = r'''
import json

results = []
entrypoint = candidate_namespace.get("run")
for case in regression_cases:
    case_id = case["case_id"]
    try:
        if not callable(entrypoint):
            raise TypeError("candidate must expose callable run(inputs)")
        actual = entrypoint(case["inputs"])
        actual_json = json.dumps(actual, allow_nan=False, sort_keys=True, separators=(",", ":"))
        expected_json = json.dumps(case[expected_key], allow_nan=False, sort_keys=True, separators=(",", ":"))
        results.append({
            "case_id": case_id,
            "passed": actual_json == expected_json,
            "actual_json": actual_json[:2_000],
            "error": "",
        })
    except Exception as exc:
        results.append({
            "case_id": case_id,
            "passed": False,
            "actual_json": "",
            "error": str(exc)[:1_000],
        })

print("__MIRROR_STAGE19_RESULTS__" + json.dumps(
    {"results": results}, allow_nan=False, sort_keys=True, separators=(",", ":")
))
'''


def find_container_runtime() -> tuple[str, str] | None:
    requested = os.environ.get("MIRROR_SANDBOX_CONTAINER_RUNTIME", "").strip().lower()
    candidates = [requested] if requested else ["docker", "podman"]
    for candidate in candidates:
        if candidate not in {"docker", "podman"}:
            continue
        path = shutil.which(candidate)
        if path:
            return candidate, path
    return None


def build_container_command(
    runtime: str,
    image: str,
    *,
    timeout_seconds: float,
    memory_limit_mb: int,
    max_pids: int,
) -> list[str]:
    if runtime not in {"docker", "podman"}:
        raise ValueError("runtime must be docker or podman")
    security_opt = "no-new-privileges:true" if runtime == "docker" else "no-new-privileges"
    pull_policy = "missing" if os.environ.get("MIRROR_SANDBOX_PULL_IMAGE") == "1" else "never"
    cpu_seconds = max(1, int(timeout_seconds))
    memory = max(64, int(memory_limit_mb))
    return [
        runtime, "run", "--rm", "--interactive",
        "--pull", pull_policy,
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt", security_opt,
        "--pids-limit", str(max(8, int(max_pids))),
        "--memory", f"{memory}m",
        "--memory-swap", f"{memory}m",
        "--cpus", "1",
        "--user", "65534:65534",
        "--ulimit", f"cpu={cpu_seconds}:{cpu_seconds + 1}",
        "--ulimit", "fsize=1048576:1048576",
        "--ulimit", "nofile=64:64",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=16m",
        "--workdir", "/tmp",
        "--env", "PYTHONHASHSEED=0",
        "--entrypoint", "python",
        image, "-I", "-S", "-c", _CHILD_RUNNER,
    ]


def run_python_in_container(
    source: str,
    *,
    timeout_seconds: float = 8.0,
    memory_limit_mb: int = 256,
    max_pids: int = 32,
    max_loop_steps: int = 50000,
    image: str | None = None,
) -> ContainerExecutionResult:
    """Run candidate code in a fresh, network-isolated Docker/Podman container."""
    if not isinstance(source, str) or not source.strip():
        return ContainerExecutionResult(False, "", "empty source", "", None, None)
    if len(source.encode("utf-8", errors="replace")) > 512_000:
        return ContainerExecutionResult(False, "", "source exceeds 512 KB limit", "", None, None)
    if not 1 <= float(timeout_seconds) <= 60:
        return ContainerExecutionResult(False, "", "timeout must be between 1 and 60 seconds", "", None, None)

    found = find_container_runtime()
    if found is None:
        return ContainerExecutionResult(
            False, "", "Docker or Podman is required; process fallback is disabled by default.", "", None, None
        )
    runtime, _runtime_path = found
    image_name = image or os.environ.get("MIRROR_SANDBOX_CONTAINER_IMAGE", DEFAULT_IMAGE)
    command = build_container_command(
        runtime,
        image_name,
        timeout_seconds=timeout_seconds,
        memory_limit_mb=memory_limit_mb,
        max_pids=max_pids,
    )
    payload = json.dumps({"source": source, "max_loop_steps": int(max_loop_steps)})
    env = os.environ.copy()
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    env.pop("ALL_PROXY", None)
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)
    env.pop("all_proxy", None)

    try:
        completed = subprocess.run(
            command,
            input=payload,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(timeout_seconds) + 3.0,
            check=False,
            close_fds=True,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return ContainerExecutionResult(False, "", "container execution timeout", runtime, None, None)
    except OSError as exc:
        return ContainerExecutionResult(False, "", f"container runtime error: {exc}", runtime, None, None)

    try:
        data: dict[str, Any] = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        detail = (completed.stderr.strip() or completed.stdout.strip())[-1200:]
        return ContainerExecutionResult(
            False, "", f"container returned invalid output: {detail}", runtime, None, completed.returncode
        )

    ok = bool(data.get("ok")) and completed.returncode == 0
    error = str(data.get("error") or "")
    if completed.returncode != 0 and not error:
        error = completed.stderr.strip()[-1200:] or f"container exited with code {completed.returncode}"
    return ContainerExecutionResult(
        ok=ok,
        output=str(data.get("output") or ""),
        error=error,
        runtime=runtime,
        pid=int(data["pid"]) if data.get("pid") is not None else None,
        returncode=completed.returncode,
    )


def run_python_regression_in_container(
    candidate_source: str,
    regression_cases: list[dict[str, Any]],
    *,
    expected_key: str,
    timeout_seconds: float = 8.0,
    memory_limit_mb: int = 256,
    max_pids: int = 32,
    max_loop_steps: int = 50000,
    image: str | None = None,
    runtime: str | None = None,
) -> ContainerExecutionResult:
    """Run candidate code and a trusted regression harness in a fresh container."""
    if not isinstance(candidate_source, str) or not candidate_source.strip():
        return ContainerExecutionResult(False, "", "empty candidate source", "", None, None)
    if len(candidate_source.encode("utf-8", errors="replace")) > 512_000:
        return ContainerExecutionResult(False, "", "candidate source exceeds 512 KB limit", "", None, None)
    if not isinstance(regression_cases, list) or not regression_cases:
        return ContainerExecutionResult(False, "", "regression cases must be a non-empty list", "", None, None)
    if expected_key not in {"baseline_expected", "candidate_expected"}:
        return ContainerExecutionResult(False, "", "unsupported expected output key", "", None, None)
    if not 1 <= float(timeout_seconds) <= 60:
        return ContainerExecutionResult(False, "", "timeout must be between 1 and 60 seconds", "", None, None)
    try:
        payload = json.dumps({
            "candidate_source": candidate_source,
            "regression_cases": regression_cases,
            "expected_key": expected_key,
            "max_loop_steps": int(max_loop_steps),
            "candidate_timeout_seconds": float(timeout_seconds),
        }, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        return ContainerExecutionResult(False, "", f"regression payload is not JSON-safe: {exc}", "", None, None)
    if len(payload.encode("utf-8")) > 1_000_000:
        return ContainerExecutionResult(False, "", "regression payload exceeds 1 MB limit", "", None, None)

    if runtime is not None:
        if runtime not in {"docker", "podman"}:
            return ContainerExecutionResult(False, "", "runtime must be docker or podman", "", None, None)
        runtime_path = shutil.which(runtime)
        if runtime_path is None:
            return ContainerExecutionResult(False, "", f"requested container runtime '{runtime}' is unavailable", "", None, None)
        found = (runtime, runtime_path)
    else:
        found = find_container_runtime()
    if found is None:
        return ContainerExecutionResult(
            False, "", "Docker or Podman is required; process fallback is disabled by default.", "", None, None
        )
    runtime, _runtime_path = found
    image_name = image or os.environ.get("MIRROR_SANDBOX_CONTAINER_IMAGE", DEFAULT_IMAGE)
    command = build_container_command(
        runtime,
        image_name,
        timeout_seconds=timeout_seconds,
        memory_limit_mb=memory_limit_mb,
        max_pids=max_pids,
    )
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    try:
        completed = subprocess.run(
            command,
            input=payload,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(timeout_seconds) + 3.0,
            check=False,
            close_fds=True,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return ContainerExecutionResult(False, "", "container execution timeout", runtime, None, None)
    except OSError as exc:
        return ContainerExecutionResult(False, "", f"container runtime error: {exc}", runtime, None, None)

    try:
        data: dict[str, Any] = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        detail = (completed.stderr.strip() or completed.stdout.strip())[-1_200:]
        return ContainerExecutionResult(
            False, "", f"container returned invalid output: {detail}", runtime, None, completed.returncode
        )
    ok = bool(data.get("ok")) and completed.returncode == 0
    error = str(data.get("error") or "")
    if completed.returncode != 0 and not error:
        error = completed.stderr.strip()[-1_200:] or f"container exited with code {completed.returncode}"
    return ContainerExecutionResult(
        ok=ok,
        output=str(data.get("output") or ""),
        error=error,
        runtime=runtime,
        pid=int(data["pid"]) if data.get("pid") is not None else None,
        returncode=completed.returncode,
    )
