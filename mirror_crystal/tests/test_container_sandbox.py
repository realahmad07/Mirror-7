from __future__ import annotations

import os
import shutil

import pytest

from mirror_crystal.container_sandbox import (
    DEFAULT_IMAGE, build_container_command, find_container_runtime, run_python_in_container,
)


def test_container_command_has_required_isolation_controls():
    command = build_container_command("docker", DEFAULT_IMAGE, timeout_seconds=6, memory_limit_mb=192, max_pids=16)
    joined = " ".join(command)
    for expected in (
        "--network=none", "--read-only", "--cap-drop=ALL", "no-new-privileges",
        "--pids-limit 16", "--memory 192m", "--memory-swap 192m", "--cpus 1",
        "--user 65534:65534", "--tmpfs", "--pull never",
    ):
        assert expected in joined
    assert "-v " not in joined and "--volume" not in joined


def test_container_execution_smoke_and_pid_namespace():
    if find_container_runtime() is None:
        if os.environ.get("MIRROR_REQUIRE_CONTAINER_SANDBOX") == "1":
            pytest.fail("CI requires Docker/Podman but no container runtime was found")
        pytest.skip("Docker/Podman is not installed")
    result = run_python_in_container("print(6 * 7)")
    if not result.ok and os.environ.get("MIRROR_REQUIRE_CONTAINER_SANDBOX") != "1":
        pytest.skip(result.error)
    assert result.ok, result.error
    assert result.output == "42"
    assert result.pid == 1


def test_container_runner_fails_closed_when_runtime_is_unavailable(monkeypatch):
    import mirror_crystal.container_sandbox as sandbox
    monkeypatch.setattr(sandbox, "find_container_runtime", lambda: None)
    result = sandbox.run_python_in_container("print('do not run')")
    assert not result.ok
    assert "required" in result.error
