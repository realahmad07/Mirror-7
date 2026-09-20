from __future__ import annotations

import json
import subprocess
import sys
import urllib.request

from mirror7_backend.service import BackendService
from mirror7_backend.http_server import create_server
from phase140_unified_cognitive_runtime.mirror7_phase140 import UnifiedCognitiveRuntime


def _request(server, method, path, payload=None):
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


def test_phase341_real_runtime_contract():
    service = BackendService(engine_factory=UnifiedCognitiveRuntime)
    server = create_server(port=0, service=service)
    try:
        status, created = _request(
            server, "POST", "/api/sessions", {"session_id": "pre-gpu"}
        )
        assert status == 201
        assert created["session_id"] == "pre-gpu"

        status, result = _request(
            server,
            "POST",
            "/api/sessions/pre-gpu/step",
            {"observation": {"signal": 1}, "goal": "respond"},
        )
        assert status == 200
        assert result["sequence"] == 1
        assert result["engine_result"]["goal"] == "respond"
        assert result["engine_result"]["grounded"] != ""
        assert result["state_digest"]

        status, status_body = _request(server, "GET", "/api/status")
        assert status == 200
        assert status_body["status"]["sessions"] == 1
        assert status_body["status"]["metrics"]["steps_succeeded"] == 1
    finally:
        server.shutdown()
        server.server_close()


def test_phase341_source_compiles():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "mirror7_backend",
            "phase140_unified_cognitive_runtime",
            "phase341_pre_gpu_readiness",
        ],
        check=True,
    )
