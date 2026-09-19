from __future__ import annotations

import json
import threading
import urllib.request

from .http_server import create_server


class FakeEngine:
    def __init__(self):
        self.state = {"steps": 0}

    def step(self, observation, *, goal=None, research_tasks=(), views=()):
        self.state["steps"] += 1
        return {
            "state": dict(self.state),
            "observation": observation,
            "goal": goal,
        }


def request(server, method, path, payload=None):
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(req, timeout=3) as response:
        return response.status, json.loads(response.read())


def test_health_create_and_step():
    from .service import BackendService

    service = BackendService(engine_factory=FakeEngine)
    server = create_server(port=0, service=service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, health = request(server, "GET", "/api/health")
        assert status == 200
        assert health["ok"] is True

        status, created = request(server, "POST", "/api/sessions", {"session_id": "ui-test"})
        assert status == 201
        assert created["session_id"] == "ui-test"

        status, result = request(
            server,
            "POST",
            "/api/sessions/ui-test/step",
            {"observation": {"value": 7}, "goal": "test"},
        )
        assert status == 200
        assert result["sequence"] == 1
        assert result["observation"] == {"value": 7}
        assert result["goal"] == "test"
        assert result["state_digest"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
