from __future__ import annotations

import json
import threading
import urllib.request

from mirror7_backend.http_server import create_server
from mirror7_backend.service import BackendService


class EchoEngine:
    def step(self, observation, *, goal=None, research_tasks=(), views=()):
        return {
            "grounded": str(observation),
            "goal": goal,
            "response": f"processed:{observation}",
        }


def _request(server, method, path, payload=None, headers=None):
    data = None
    merged = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        merged["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}{path}",
        data=data,
        headers=merged,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        body = response.read()
        return response.status, response.headers, json.loads(body) if body else None


def test_phase342_browser_contract_and_security_headers():
    service = BackendService(engine_factory=EchoEngine)
    server = create_server(port=0, service=service, cors_origin="https://example.com")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, headers, body = _request(server, "OPTIONS", "/api/sessions")
        assert status == 204
        assert body is None
        assert headers["Access-Control-Allow-Origin"] == "https://example.com"
        assert "POST" in headers["Access-Control-Allow-Methods"]

        status, headers, body = _request(
            server, "POST", "/api/sessions", {"session_id": "release-gate"}
        )
        assert status == 201
        assert body["session_id"] == "release-gate"
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["Cache-Control"] == "no-store"

        status, _, body = _request(
            server,
            "POST",
            "/api/sessions/release-gate/step",
            {"observation": {"hello": "world"}, "goal": "respond"},
        )
        assert status == 200
        assert body["sequence"] == 1
        assert body["engine_result"]["goal"] == "respond"

        status, _, body = _request(server, "GET", "/api/sessions/release-gate")
        assert status == 200
        assert body["snapshot"]["sequence"] == 1

        status, _, body = _request(server, "DELETE", "/api/sessions/release-gate")
        assert status == 200
        assert body["closed"] is True

        status, _, body = _request(server, "GET", "/api/status")
        assert status == 200
        assert body["sessions"] == 0
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
