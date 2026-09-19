from __future__ import annotations

import json
import tempfile
import threading
import urllib.error
import urllib.request

from mirror7_backend.actions import ActionGateway, ActionPolicy
from mirror7_backend.http_server import create_server
from mirror7_backend.persistence import CheckpointStore
from mirror7_backend.service import BackendService


class Engine:
    def step(self, observation, goal=None, research_tasks=(), views=()):
        return {"observation": observation, "goal": goal}


def request(server, method, path, payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}{path}",
        data=data, headers=headers, method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_snapshot_checkpoint_delete_and_action_api():
    with tempfile.TemporaryDirectory() as root:
        gateway = ActionGateway(
            {"echo": lambda value: value},
            ActionPolicy(frozenset({"echo"}), max_calls=2),
        )
        service = BackendService(
            engine_factory=Engine,
            checkpoint_store=CheckpointStore(root),
            action_gateway=gateway,
        )
        server = create_server(port=0, service=service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            assert request(server, "POST", "/api/sessions", {"session_id": "launch"})[0] == 201
            assert request(server, "POST", "/api/sessions/launch/step",
                           {"observation": {"x": 1}})[0] == 200

            status, snapshot = request(server, "GET", "/api/sessions/launch")
            assert status == 200
            assert snapshot["snapshot"]["sequence"] == 1

            assert request(server, "POST", "/api/sessions/launch/save", {})[0] == 200
            status, deleted = request(server, "DELETE", "/api/sessions/launch/checkpoint")
            assert status == 200 and deleted["deleted"] is True

            status, action = request(
                server, "POST", "/api/sessions/launch/action",
                {"name": "echo", "args": ["ok"]},
            )
            assert status == 200 and action["result"] == "ok"

            status, denied = request(
                server, "POST", "/api/sessions/launch/action",
                {"name": "blocked"},
            )
            assert status == 403 and "not allowed" in denied["error"]
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


def test_action_payload_shape_is_rejected():
    service = BackendService(
        engine_factory=Engine,
        action_gateway=ActionGateway(
            {"echo": lambda value: value},
            ActionPolicy(frozenset({"echo"})),
        ),
    )
    server = create_server(port=0, service=service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request(server, "POST", "/api/sessions", {"session_id": "shape"})
        status, body = request(
            server, "POST", "/api/sessions/shape/action",
            {"name": "echo", "args": {"bad": True}},
        )
        assert status == 400
        assert "array" in body["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
