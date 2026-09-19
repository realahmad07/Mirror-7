import pytest

from phase280_sealed_candidate_eval import EvaluationPack
from phase315_production_sandbox import ProductionSandbox
from phase317_production_self_redesign import ProductionSelfRedesign

pytestmark = pytest.mark.skipif(
    not ProductionSandbox.docker_available(),
    reason="Docker daemon unavailable; production-sandbox CI runs with Docker enabled",
)

BASE = "def solve(values):\n    return values[-1]\n"

def pack():
    return EvaluationPack(
        [{"public": {"values": [0, 2]}, "target": 4}],
        [{"public": {"values": [3, 6]}, "target": 9}],
        [{"public": {"values": [1, 1]}, "target": 1}],
    )

def test_production_self_redesign_uses_sandbox():
    system = ProductionSelfRedesign(BASE)
    assert system.evaluator.sandbox is system.sandbox

def test_production_self_redesign_can_promote_bounded_change():
    system = ProductionSelfRedesign(BASE)
    report = system.run_round(
        pack(), "solver.py", ["sequence extrapolation failure"], 1, 1
    )
    assert report.promoted
    assert "values[-2]" in system.source

def test_production_self_redesign_rollback_restores_exact_source():
    system = ProductionSelfRedesign(BASE)
    report = system.run_round(
        pack(), "solver.py", ["sequence extrapolation failure"], 1, 1
    )
    assert report.promoted
    rollback = system.rollback()
    assert rollback.accepted
    assert system.source == BASE
