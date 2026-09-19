from __future__ import annotations

import threading
from pathlib import Path

from mirror7_backend.persistence import CheckpointStore, CheckpointError
from mirror7_backend.service import BackendService


class Engine:
    def step(self, observations, goal=None, research_tasks=(), views=()):
        return {"observation": observations, "goal": goal}


def test_checkpoint_store_rejects_traversal_and_corruption(tmp_path: Path):
    store = CheckpointStore(tmp_path)
    try:
        store.path_for("../escape")
    except ValueError:
        pass
    else:
        raise AssertionError("path traversal was accepted")

    store.save("safe", {"value": 1})
    path = store.path_for("safe")
    raw = path.read_text()
    path.write_text(raw.replace('"value": 1', '"value": 2'))
    try:
        store.load("safe")
    except CheckpointError:
        pass
    else:
        raise AssertionError("corrupted checkpoint was accepted")


def test_concurrent_session_steps_preserve_exact_sequence():
    service = BackendService(engine_factory=Engine)
    service.create_session("race")
    errors = []

    def worker():
        try:
            for i in range(10):
                service.step("race", {"i": i})
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert service.get_session("race").sequence == 40
    assert len(service.get_session("race").history) == 40


def test_close_waits_for_active_step_and_removes_session():
    service = BackendService(engine_factory=Engine)
    service.create_session("close")
    service.step("close", {"x": 1})
    service.close_session("close")
    assert "close" not in service.list_sessions()
