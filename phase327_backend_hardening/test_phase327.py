import pytest

from mirror7_backend.persistence import CheckpointStore
from mirror7_backend.service import BackendService
from phase327_backend_hardening import HardenedBackend, canonical_digest


class Engine:
    def step(self, observation, *, goal=None, research_tasks=(), views=()):
        return {"observation": observation, "goal": goal}


def test_session_id_validation():
    service = BackendService(engine_factory=Engine)
    hard = HardenedBackend(service)
    for value in ("", "x\n", "x\r", "x\t"):
        with pytest.raises(ValueError):
            hard.validate_session_id(value)
    assert hard.validate_session_id("normal-1") == "normal-1"


def test_status_and_snapshot_are_defensive():
    service = BackendService(engine_factory=Engine)
    service.create_session("s")
    service.step("s", {"nested": {"x": 1}})
    hard = HardenedBackend(service)
    status = hard.status()
    snapshot = hard.snapshot("s")
    snapshot["state"]["injected"] = True
    status["sessions"] = 999
    assert "injected" not in service.get_session("s").state
    assert service.status()["sessions"] == 1


def test_snapshot_digest_is_stable():
    service = BackendService(engine_factory=Engine)
    service.create_session("s")
    service.step("s", {"b": 2, "a": 1})
    hard = HardenedBackend(service)
    assert hard.snapshot_digest("s") == canonical_digest(hard.snapshot("s"))


def test_checkpoint_corruption_fails_closed(tmp_path):
    service = BackendService(engine_factory=Engine, checkpoint_store=CheckpointStore(tmp_path))
    service.create_session("s")
    service.step("s", {"x": 1})
    service.save("s")
    path = service.checkpoint_store.path_for("s")
    text = path.read_text()
    path.write_text(text.replace('"x": 1', '"x": 2'))
    hard = HardenedBackend(service)
    with pytest.raises(Exception):
        hard.load_checkpoint("s")


def test_overlong_id_rejected():
    service = BackendService(engine_factory=Engine)
    hard = HardenedBackend(service)
    with pytest.raises(ValueError):
        hard.validate_session_id("x" * 129)
