import json
import subprocess
import sys
import pytest
from phase161_170_blackbox.phase161_170 import run_all_seeds

RESULTS = run_all_seeds((0, 1, 2))

@pytest.mark.parametrize("phase", sorted(RESULTS))
def test_phase_passes_all_three_seeds(phase):
    assert RESULTS[phase] == {0: True, 1: True, 2: True}

def test_final_gate_is_strict():
    assert all(all(v for v in seed_map.values()) for seed_map in RESULTS.values())

def test_independent_evaluator_positive_control():
    payload = {"raw_representation_accuracy":1.0,"opaque_action_accuracy":1.0,"goal_success":1.0,"contamination_detected":True,"max_memory":512}
    proc = subprocess.run([sys.executable,"phase161_170_blackbox/independent_evaluator.py"],input=json.dumps(payload),text=True,capture_output=True,check=False)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["passed"]

def test_independent_evaluator_negative_control():
    payload = {"raw_representation_accuracy":0.4,"opaque_action_accuracy":0.3,"goal_success":0.2,"contamination_detected":False,"max_memory":5000}
    proc = subprocess.run([sys.executable,"phase161_170_blackbox/independent_evaluator.py"],input=json.dumps(payload),text=True,capture_output=True,check=False)
    assert proc.returncode != 0
    assert not json.loads(proc.stdout)["passed"]
