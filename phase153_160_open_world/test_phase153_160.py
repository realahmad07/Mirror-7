import json
import subprocess
import sys
import pytest

from phase153_160_open_world.phase153_160 import run_all_seeds

RESULTS = run_all_seeds((0, 1, 2))

@pytest.mark.parametrize("phase", sorted(RESULTS))
def test_phase_passes_all_three_seeds(phase):
    assert RESULTS[phase] == {0: True, 1: True, 2: True}

def test_phase158_external_evaluator_contract():
    payload = {"prediction_accuracy": 1.0, "unknown_confidence": 0.1, "max_memory": 64, "max_workspace": 64}
    proc = subprocess.run(
        [sys.executable, "phase153_160_open_world/independent_evaluator.py"],
        input=json.dumps(payload), text=True, capture_output=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["passed"] is True


def test_phase158_external_evaluator_rejects_bad_metrics():
    payload = {"prediction_accuracy": 0.5, "unknown_confidence": 0.5, "max_memory": 2000, "max_workspace": 200}
    proc = subprocess.run(
        [sys.executable, "phase153_160_open_world/independent_evaluator.py"],
        input=json.dumps(payload), text=True, capture_output=True, check=False,
    )
    assert proc.returncode != 0
    assert json.loads(proc.stdout)["passed"] is False
