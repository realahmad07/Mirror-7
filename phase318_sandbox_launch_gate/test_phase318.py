import pytest

from phase315_production_sandbox import ProductionSandbox, SandboxLimits

pytestmark = pytest.mark.skipif(
    not ProductionSandbox.docker_available(),
    reason="Docker daemon unavailable; production-sandbox CI runs with Docker enabled",
)

def test_hardened_defaults():
    limits = SandboxLimits()
    assert limits.memory_mb <= 256
    assert limits.cpu_count <= 1
    assert limits.pids_limit <= 32
    assert limits.output_bytes <= 64 * 1024
    assert limits.file_size_bytes <= 1024 * 1024

def test_image_is_explicit():
    sandbox = ProductionSandbox()
    assert sandbox.image

def test_pull_is_not_enabled_by_default():
    assert not ProductionSandbox().allow_pull

def test_sandbox_result_is_fail_closed():
    result = ProductionSandbox().run_candidate(
        "def solve(values):\n    return values[-1]\n",
        [{"public": {"values": [1, 2]}, "target": 2}],
    )
    assert result.accepted
