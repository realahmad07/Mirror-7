import pytest

from phase315_production_sandbox import ProductionSandbox, SandboxLimits
from phase292_patch_validation import PatchValidator

pytestmark = pytest.mark.skipif(
    not ProductionSandbox.docker_available(),
    reason="Docker daemon unavailable; production-sandbox CI runs with Docker enabled",
)

def test_limits_are_bounded():
    limits = SandboxLimits()
    limits.validate()
    assert limits.timeout_seconds <= 10
    assert limits.memory_mb <= 512
    assert limits.pids_limit <= 64

def test_non_root_execution():
    result = ProductionSandbox().run_script("import os\nprint(os.getuid())\n")
    assert result.accepted
    assert result.lines[-1] == "65534"

def test_network_is_disabled():
    source = """
import socket
try:
    socket.create_connection(("1.1.1.1", 80), 1)
except OSError:
    print("blocked")
else:
    raise SystemExit(9)
"""
    result = ProductionSandbox().run_script(source)
    assert result.accepted
    assert result.lines == ("blocked",)

def test_workspace_is_read_only():
    source = """
try:
    with open("/workspace/escape.txt", "w", encoding="utf-8") as f:
        f.write("escape")
except OSError:
    print("blocked")
else:
    raise SystemExit(9)
"""
    result = ProductionSandbox().run_script(source)
    assert result.accepted
    assert result.lines == ("blocked",)

def test_timeout_is_enforced():
    result = ProductionSandbox(SandboxLimits(timeout_seconds=1.0)).run_script(
        "while True: pass\n"
    )
    assert not result.accepted
    assert result.timed_out

def test_candidate_executes_under_sandbox():
    source = "def solve(values):\n    return values[-1] + (values[-1] - values[-2])\n"
    cases = [
        {"public": {"values": [1, 3]}, "target": 5},
        {"public": {"values": [5, 8]}, "target": 11},
    ]
    result = ProductionSandbox().run_candidate(
        source, cases, validator=PatchValidator()
    )
    assert result.accepted
    assert result.lines == ("5", "11")

def test_candidate_output_cap_fails_closed():
    source = "def solve(values):\n    return [0] * 1000000\n"
    cases = [{"public": {"values": [1, 2]}, "target": []}]
    result = ProductionSandbox().run_candidate(
        source, cases, validator=PatchValidator()
    )
    assert not result.accepted

def test_candidate_case_limit():
    sandbox = ProductionSandbox(SandboxLimits(max_cases=1))
    cases = [
        {"public": {"values": [1, 2]}, "target": 3},
        {"public": {"values": [2, 4]}, "target": 6},
    ]
    result = sandbox.run_candidate(
        "def solve(values):\n    return values[-1]\n",
        cases,
        validator=PatchValidator(),
    )
    assert not result.accepted

def test_validator_rejection_happens_before_container():
    result = ProductionSandbox().run_candidate(
        "import os\ndef solve(values):\n    return 0\n",
        [{"public": {"values": [1]}, "target": 0}],
        validator=PatchValidator(),
    )
    assert not result.accepted
    assert result.reason == "candidate rejected before sandbox"
