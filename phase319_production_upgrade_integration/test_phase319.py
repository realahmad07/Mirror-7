import pytest

from phase315_production_sandbox import ProductionSandbox
from .mirror7_phase319 import (
    ProductionMultiCapabilityUpgrade,
    ProductionSequenceFrontierAdapter,
)

pytestmark = pytest.mark.skipif(
    not ProductionSandbox.docker_available(),
    reason="Docker daemon unavailable; production integration CI runs with Docker enabled",
)


def test_production_registry_contains_four_capabilities():
    system = ProductionMultiCapabilityUpgrade()
    assert system.capability_names == (
        "compositional_language",
        "planning",
        "sequence_extrapolation",
        "symbolic_reasoning",
    )


def test_sequence_adapter_uses_the_mandatory_production_sandbox():
    system = ProductionMultiCapabilityUpgrade()
    adapter = system.adapters.get("sequence_extrapolation")
    assert isinstance(adapter, ProductionSequenceFrontierAdapter)
    assert adapter.sandbox is system.sandbox
    assert adapter.system.sandbox is system.sandbox
    assert adapter.system.evaluator.sandbox is system.sandbox


def test_frontier_starts_with_a_real_source_redesign_path():
    system = ProductionMultiCapabilityUpgrade()
    assert system.frontier.top_gap() is not None
    assert system.frontier.top_gap().name == "planning"
    assert system.source.strip().endswith("return values[-1]")


def test_single_step_is_bounded_and_accepts_a_real_upgrade():
    system = ProductionMultiCapabilityUpgrade()
    event = system.step(seed=7, rounds=4, max_candidates=1)
    assert event is not None
    assert event.accepted
    assert 0.0 <= event.score <= 1.0


def test_end_to_end_campaign_reaches_production_sequence_adapter():
    system = ProductionMultiCapabilityUpgrade()
    events = system.run(steps=4, rounds=4, max_candidates=1)
    assert len(events) == 4
    assert all(event.accepted for event in events)
    assert any(event.sandboxed for event in events)
    assert "values[-2]" in system.source


def test_campaign_scores_remain_bounded():
    system = ProductionMultiCapabilityUpgrade()
    system.run(steps=4, rounds=4, max_candidates=1)
    assert all(0.0 <= score <= 1.0 for score in system.scores.values())


def test_run_bound_is_enforced():
    system = ProductionMultiCapabilityUpgrade()
    with pytest.raises(ValueError):
        system.run(0)


def test_invalid_step_bounds_are_rejected():
    system = ProductionMultiCapabilityUpgrade()
    with pytest.raises(ValueError):
        system.step(rounds=0)
    with pytest.raises(ValueError):
        system.step(max_candidates=0)
