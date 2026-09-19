import pytest

from phase315_production_sandbox import ProductionSandbox
from phase316_sandboxed_evaluator import ProductionSealedSourceEvaluator

pytestmark = pytest.mark.skipif(
    not ProductionSandbox.docker_available(),
    reason="Docker daemon unavailable; production-sandbox CI runs with Docker enabled",
)

def test_production_evaluator_requires_docker():
    evaluator = ProductionSealedSourceEvaluator()
    assert evaluator.sandbox.docker_available()

def test_production_evaluator_has_no_legacy_backend():
    evaluator = ProductionSealedSourceEvaluator()
    assert evaluator.sandbox is not None
