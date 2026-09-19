from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import subprocess
import tempfile
import textwrap
import uuid
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: float = 5.0
    memory_mb: int = 256
    cpu_count: float = 1.0
    pids_limit: int = 32
    output_bytes: int = 64 * 1024
    file_size_bytes: int = 1 * 1024 * 1024
    tmpfs_mb: int = 16
    max_source_bytes: int = 64 * 1024
    max_cases: int = 128

    def validate(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.memory_mb < 64:
            raise ValueError("memory_mb must be at least 64")
        if self.cpu_count <= 0:
            raise ValueError("cpu_count must be positive")
        if self.pids_limit < 4:
            raise ValueError("pids_limit must be at least 4")
        if self.output_bytes < 1024:
            raise ValueError("output_bytes must be at least 1024")
        if self.file_size_bytes < 4096:
            raise ValueError("file_size_bytes must be at least 4096")
        if self.tmpfs_mb < 4:
            raise ValueError("tmpfs_mb must be at least 4")
        if self.max_source_bytes < 1024:
            raise ValueError("max_source_bytes must be at least 1024")
        if self.max_cases < 1:
            raise ValueError("max_cases must be positive")


@dataclass(frozen=True)
class SandboxResult:
    accepted: bool
    returncode: int | None
    stdout: str
    stderr: str
    reason: str
    timed_out: bool = False
    resource_exceeded: bool = False
    backend: str = "docker"

    @property
    def lines(self) -> tuple[str, ...]:
        return tuple(line for line in self.stdout.splitlines() if line.strip())


class ProductionSandbox:
    """Fail-closed Docker sandbox for untrusted self-modification candidates.

    The sandbox never mounts the repository, Docker socket, host root, or a
    writable host directory. Candidate inputs are exposed only through stdin.
    """

    HARNESS = textwrap.dedent(
        """
        import json
        import os
        import runpy
        import sys

        candidate_path = sys.argv[1]
        max_output = int(os.environ["MIRROR7_MAX_OUTPUT_BYTES"])
        env = runpy.run_path(candidate_path, run_name="__candidate__")
        solve = env.get("solve")
        if not callable(solve):
            raise SystemExit(3)

        for line in sys.stdin:
            payload = json.loads(line)
            answer = solve(payload["values"])
            raw = json.dumps(answer, separators=(",", ":"))
            if len(raw.encode("utf-8")) > max_output:
                raise SystemExit(22)
            print(raw, flush=True)
        """
    ).strip()

    def __init__(
        self,
        limits: SandboxLimits | None = None,
        image: str | None = None,
        allow_pull: bool = False,
    ) -> None:
        self.limits = limits or SandboxLimits()
        self.limits.validate()
        self.image = image or os.environ.get(
            "MIRROR7_SANDBOX_IMAGE", "python:3.11.16-alpine3.24"
        )
        self.allow_pull = bool(allow_pull)

    @staticmethod
    def docker_available() -> bool:
        docker = shutil.which("docker")
        if docker is None:
            return False
        try:
            proc = subprocess.run(
                [docker, "info", "--format", "{{.ServerVersion}}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return proc.returncode == 0

    def _ensure_image(self) -> tuple[bool, str]:
        docker = shutil.which("docker")
        if docker is None:
            return False, "docker executable unavailable"

        inspect = subprocess.run(
            [docker, "image", "inspect", self.image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
        if inspect.returncode == 0:
            return True, "image available"

        if not self.allow_pull:
            return False, "sandbox image unavailable and image pulling is disabled"

        pulled = subprocess.run(
            [docker, "pull", self.image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
            check=False,
        )
        if pulled.returncode != 0:
            return False, "sandbox image pull failed"
        return True, "image pulled"

    def _command(self, workspace: str, container_name: str, script: str) -> list[str]:
        docker = shutil.which("docker")
        if docker is None:
            raise RuntimeError("docker executable unavailable")
        limit = self.limits
        return [
            docker,
            "run",
            "--rm",
            "--init",
            "--name",
            container_name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges=true",
            "--user",
            "65534:65534",
            "--pids-limit",
            str(limit.pids_limit),
            "--memory",
            f"{limit.memory_mb}m",
            "--memory-swap",
            f"{limit.memory_mb}m",
            "--cpus",
            str(limit.cpu_count),
            "--ulimit",
            f"nofile=64:64",
            "--ulimit",
            f"fsize={limit.file_size_bytes}:{limit.file_size_bytes}",
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,size={limit.tmpfs_mb}m",
            "--mount",
            f"type=bind,source={workspace},target=/workspace,readonly",
            "--workdir",
            "/workspace",
            "--env",
            f"MIRROR7_MAX_OUTPUT_BYTES={limit.output_bytes}",
            self.image,
            "python",
            "-I",
            script,
        ]

    def _run(self, command: Sequence[str], payload: str = "") -> SandboxResult:
        container_name = next(
            part[1:] for part in command[command.index("--name") + 1 : command.index("--network")]
        )
        try:
            proc = subprocess.run(
                list(command),
                input=payload,
                text=True,
                capture_output=True,
                timeout=self.limits.timeout_seconds,
                check=False,
            )
            resource_exceeded = proc.returncode in {137, 139, 143}
            accepted = proc.returncode == 0 and not resource_exceeded
            reason = "completed" if accepted else f"container exit {proc.returncode}"
            return SandboxResult(
                accepted,
                proc.returncode,
                proc.stdout[: self.limits.output_bytes],
                proc.stderr[: self.limits.output_bytes],
                reason,
                resource_exceeded=resource_exceeded,
            )
        except subprocess.TimeoutExpired as exc:
            docker = shutil.which("docker")
            if docker:
                subprocess.run(
                    [docker, "rm", "-f", container_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=3,
                    check=False,
                )
            return SandboxResult(
                False,
                None,
                (exc.stdout or "")[: self.limits.output_bytes],
                (exc.stderr or "")[: self.limits.output_bytes],
                "sandbox timeout",
                timed_out=True,
            )
        except OSError as exc:
            return SandboxResult(
                False,
                None,
                "",
                str(exc)[: self.limits.output_bytes],
                "sandbox launch failed",
            )

    def run_script(self, source: str) -> SandboxResult:
        """Run an arbitrary probe script only inside the production sandbox."""
        if not self.docker_available():
            return SandboxResult(False, None, "", "", "docker unavailable")
        ok, reason = self._ensure_image()
        if not ok:
            return SandboxResult(False, None, "", "", reason)
        if len(source.encode("utf-8")) > self.limits.max_source_bytes:
            return SandboxResult(False, None, "", "", "source exceeds sandbox limit")

        with tempfile.TemporaryDirectory(prefix="mirror7-sbx-") as td:
            path = os.path.join(td, "probe.py")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(source)
            name = f"mirror7-sbx-{uuid.uuid4().hex[:12]}"
            command = self._command(td, name, "/workspace/probe.py")
            return self._run(command)

    def run_candidate(
        self,
        source: str,
        cases: Sequence[Mapping[str, object]],
        validator: object | None = None,
    ) -> SandboxResult:
        """Run a validated solve(values) source with strict production limits."""
        if validator is not None:
            report = validator.validate_source(source)
            if not report.accepted:
                return SandboxResult(
                    False, None, "", "", "candidate rejected before sandbox"
                )

        if len(source.encode("utf-8")) > self.limits.max_source_bytes:
            return SandboxResult(False, None, "", "", "source exceeds sandbox limit")
        if len(cases) > self.limits.max_cases:
            return SandboxResult(False, None, "", "", "case count exceeds sandbox limit")
        if not self.docker_available():
            return SandboxResult(False, None, "", "", "docker unavailable")
        ok, reason = self._ensure_image()
        if not ok:
            return SandboxResult(False, None, "", "", reason)

        import json

        payload = "".join(
            json.dumps({"values": case["public"]["values"]}, separators=(",", ":"))
            + "\n"
            for case in cases
        )

        with tempfile.TemporaryDirectory(prefix="mirror7-sbx-") as td:
            candidate = os.path.join(td, "candidate.py")
            harness = os.path.join(td, "harness.py")
            with open(candidate, "w", encoding="utf-8") as handle:
                handle.write(source)
            with open(harness, "w", encoding="utf-8") as handle:
                handle.write(self.HARNESS)
            name = f"mirror7-sbx-{uuid.uuid4().hex[:12]}"
            command = self._command(td, name, "/workspace/harness.py")
            command.append("/workspace/candidate.py")
            result = self._run(command, payload)
            if result.accepted and len(result.lines) != len(cases):
                return SandboxResult(
                    False,
                    result.returncode,
                    result.stdout,
                    result.stderr,
                    "candidate output count mismatch",
                    resource_exceeded=result.resource_exceeded,
                )
            return result
