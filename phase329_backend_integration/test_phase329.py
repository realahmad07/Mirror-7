from __future__ import annotations

import json
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from mirror7_backend.persistence import CheckpointStore
from mirror7_backend.service import BackendService
from mirror7_backend.http_server import create_server


class FakeEngine:
    def __init__(self):
        self.state = {"steps": 0}

    def step(self, observation, *, goal=None, research_tasks=(), views=()):
        self.state["steps"] += 1
        return {"state": dict(self.state), "observation": observation, "goal": goal}


def http_request(server, method, path, payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_service_close_waits_for_active_step():
    entered = threading.Event()
    release = threading.Event()

    class BlockingEngine(FakeEngine):
        def step(self, observation, *, goal=None, research_tasks=(), views=()):
            entered.set()
            assert release.wait(timeout=3)
            return super().step(observation, goal=goal, research_tasks=research_tasks, views=views)

    service = BackendService(engine_factory=BlockingEngine)
    service.create_session("race")

    step_result = {}
    step_thread = threading.Thread(
        target=lambda: step_result.setdefault("value", service.step("race", {"x": 1}))
    )
    step_thread.start()
    assert entered.wait(timeout=3)

    close_done = threading.Event()
    close_thread = threading.Thread(
        target=lambda: (service.close_session("race"), close_done.set())
    )
    close_thread.start()
    time.sleep(0.05)
    assert not close_done.is_set()

    release.set()
    step_thread.join(timeout=3)
    close_thread.join(timeout=3)
    assert close_done.is_set()
    assert step_result["value"].sequence == 1


def test_http_validation_and_lifecycle():
    with tempfile.TemporaryDirectory() as tmp:
        service = BackendService(
            engine_factory=FakeEngine,
            checkpoint_store=CheckpointStore(Path(tmp)),
        )
        server = create_server(port=0, service=service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            status, body = http_request(server, "POST", "/api/sessions", {"session_id": "api-test"})
            assert status == 201
            assert body["session_id"] == "api-test"

            status, body = http_request(
                server,
                "POST",
                "/api/sessions/api-test/step",
                {"observation": {"value": 9}, "goal": "remember"},
            )
            assert status == 200
            assert body["sequence"] == 1

            status, body = http_request(server, "POST", "/api/sessions/api-test/save", {})
            assert status == 200
            assert body["ok"] is True

            status, body = http_request(server, "POST", "/api/sessions/api-test/step", {"observation": 10})
            assert status == 200
            assert body["sequence"] == 2

            status, body = http_request(server, "DELETE", "/api/sessions/api-test")
            assert status == 200
            assert body["closed"] is True

            status, body = http_request(server, "POST", "/api/sessions/api-test/restore", {})
            assert status == 200
            assert body["session_id"] == "api-test"

            status, body = http_request(server, "POST", "/api/sessions/api-test/step", {"observation": 11})
            assert status == 200
            assert body["sequence"] == 2

            status, body = http_request(server, "POST", "/api/sessions/../step", {})
            assert status in {400, 404}
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def test_http_rejects_oversized_body():
    service = BackendService(engine_factory=FakeEngine)
    server = create_server(port=0, service=service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        oversized = "x" * (1_048_576 + 1)
        status, body = http_request(server, "POST", "/api/sessions", {"session_id": oversized})
        assert status == 400
        assert status == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
