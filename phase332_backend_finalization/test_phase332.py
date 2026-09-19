from __future__ import annotations

from mirror7_backend.service import BackendService
from mirror7_backend.observability import BackendMetrics


class Engine:
    def __init__(self, fail=False):
        self.fail = fail

    def step(self, observations, goal=None, research_tasks=(), views=()):
        if self.fail:
            raise RuntimeError("engine failure")
        return {"ok": True, "goal": goal}


def test_metrics_are_thread_safe_and_snapshot_is_copy():
    metrics = BackendMetrics()
    metrics.increment("sessions_created")
    snap = metrics.snapshot()
    snap["sessions_created"] = 999
    assert metrics.snapshot()["sessions_created"] == 1


def test_service_reports_operational_metrics():
    service = BackendService(engine_factory=lambda: Engine())
    service.create_session("m")
    service.step("m", {"x": 1})
    service.close_session("m")
    metrics = service.status()["metrics"]
    assert metrics == {
        "sessions_created": 1,
        "sessions_closed": 1,
        "steps_started": 1,
        "steps_succeeded": 1,
        "steps_failed": 0,
    }


def test_failed_step_is_counted_and_does_not_fake_success():
    service = BackendService(engine_factory=lambda: Engine(fail=True))
    service.create_session("bad")
    try:
        service.step("bad", {"x": 1})
    except RuntimeError as exc:
        assert str(exc) == "engine failure"
    else:
        raise AssertionError("expected engine failure")
    metrics = service.status()["metrics"]
    assert metrics["steps_started"] == 1
    assert metrics["steps_succeeded"] == 0
    assert metrics["steps_failed"] == 1
