import pytest

from mirror7_backend.actions import ActionGateway, ActionPolicy
from mirror7_backend.persistence import CheckpointStore
from mirror7_backend.runtime import BackendSession
from mirror7_backend.service import BackendService


class Engine:
    def __init__(self):
        self.n = 0

    def step(self, observation, *, goal=None, research_tasks=(), views=()):
        self.n += 1
        return {"n": self.n, "observation": observation, "goal": goal}


def factory():
    return Engine()


def test_service_lifecycle_and_step():
    service = BackendService(engine_factory=factory)
    session = service.create_session("s")
    assert isinstance(session, BackendSession)
    result = service.step("s", {"x": 1}, goal="g")
    assert result.sequence == 1
    assert result.engine_result["n"] == 1
    assert service.status()["sessions"] == 1


def test_duplicate_and_unknown_sessions_rejected():
    service = BackendService(engine_factory=factory)
    service.create_session("s")
    with pytest.raises(ValueError):
        service.create_session("s")
    with pytest.raises(KeyError):
        service.get_session("missing")
    with pytest.raises(KeyError):
        service.close_session("missing")


def test_capacity_is_enforced():
    service = BackendService(engine_factory=factory, max_sessions=1)
    service.create_session("a")
    with pytest.raises(RuntimeError):
        service.create_session("b")


def test_checkpoint_restore_rehydrates_session(tmp_path):
    service = BackendService(
        engine_factory=factory,
        checkpoint_store=CheckpointStore(tmp_path),
    )
    service.create_session("saved")
    service.step("saved", {"x": 7})
    service.save("saved")
    service.close_session("saved")
    restored = service.restore("saved")
    assert restored.sequence == 1
    assert restored.history[-1]["observation"] == {"x": 7}


def test_restore_rejects_existing_session(tmp_path):
    service = BackendService(
        engine_factory=factory,
        checkpoint_store=CheckpointStore(tmp_path),
    )
    service.create_session("saved")
    service.save("saved")
    with pytest.raises(ValueError):
        service.restore("saved")


def test_action_gateway_is_composed_without_bypass():
    gateway = ActionGateway(
        {"add": lambda a, b: a + b},
        ActionPolicy(frozenset({"add"})),
    )
    service = BackendService(action_gateway=gateway)
    assert service.execute_action("add", 2, 5) == 7
    with pytest.raises(PermissionError):
        service.execute_action("delete", 1)


def test_missing_optional_components_fail_closed():
    service = BackendService(engine_factory=factory)
    with pytest.raises(RuntimeError):
        service.save("missing")
    with pytest.raises(RuntimeError):
        service.execute_action("anything")


def test_close_releases_capacity():
    service = BackendService(engine_factory=factory, max_sessions=1)
    service.create_session("a")
    service.close_session("a")
    service.create_session("b")
    assert service.status()["session_ids"] == ("b",)
