from __future__ import annotations

import json
import threading
import urllib.request

from mirror7_backend.service import BackendService
from mirror7_backend.http_server import create_server
from phase140_unified_cognitive_runtime.mirror7_phase140 import UnifiedCognitiveRuntime


def request(server, method, path, payload=None):
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
    with urllib.request.urlopen(req, timeout=10) as response:
        return response.status, json.loads(response.read())


def test_real_mirror_runtime_survives_multiple_http_turns():
    service = BackendService(engine_factory=UnifiedCognitiveRuntime)
    server = create_server(port=0, service=service, cors_origin="http://localhost:3000")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = request(server, "POST", "/api/sessions", {"session_id": "e2e"})
        assert status == 201
        assert body["session_id"] == "e2e"

        status, first = request(
            server, "POST", "/api/sessions/e2e/step",
            {"observation": {"temperature": 20}, "goal": "track temperature"},
        )
        assert status == 200
        assert first["sequence"] == 1
        assert first["engine_result"]["goal"] == "track temperature"

        status, second = request(
            server, "POST", "/api/sessions/e2e/step",
            {"observation": {"temperature": 21}},
        )
        assert status == 200
        assert second["sequence"] == 2
        assert second["state_digest"] != ""

        status, snapshot = request(server, "GET", "/api/sessions/e2e")
        assert status == 200
        assert snapshot["snapshot"]["sequence"] == 2
        assert len(snapshot["snapshot"]["history"]) == 2

        status, health = request(server, "GET", "/api/health")
        assert status == 200
        assert health["ok"] is True
        assert health["status"]["sessions"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_runtime_is_injected_not_replaced_by_backend():
    service = BackendService(engine_factory=UnifiedCognitiveRuntime)
    session = service.create_session("direct")
    assert isinstance(session.engine, UnifiedCognitiveRuntime)
    result = service.step("direct", {"value": 1})
    assert result.sequence == 1
    assert result.engine_result.grounded != ""


def test_concurrent_sessions_are_isolated():
    service = BackendService(engine_factory=UnifiedCognitiveRuntime)
    service.create_session("a")
    service.create_session("b")

    errors = []

    def worker(session_id, value):
        try:
            for _ in range(4):
                result = service.step(session_id, {"value": value})
                assert result.session_id == session_id
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=("a", 1)),
        threading.Thread(target=worker, args=("b", 2)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert service.get_session("a").sequence == 4
    assert service.get_session("b").sequence == 4
