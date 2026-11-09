from __future__ import annotations

import json
import threading
import urllib.error
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


class SemanticEngine:
    def step(self, observation, *, goal=None, research_tasks=(), views=()):
        from phase343_semantic_state import SemanticStateInducer

        return type(
            "Result",
            (),
            {"semantic_state": SemanticStateInducer().discover(observation)},
        )()


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
        assert result["realization_contract"] is None
        assert result["response"] is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_step_returns_realization_contract_and_generated_response():
    from .service import BackendService

    def generator(request):
        return {"mode": request.contract.mode, "text": "Mirror response"}

    service = BackendService(
        engine_factory=SemanticEngine,
        response_generator=generator,
    )
    server = create_server(port=0, service=service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, created = request(server, "POST", "/api/sessions", {"session_id": "generation-test"})
        assert status == 201
        assert created["session_id"] == "generation-test"

        status, result = request(
            server,
            "POST",
            "/api/sessions/generation-test/step",
            {"observation": "Explain Python"},
        )
        assert status == 200
        assert result["realization_contract"]["mode"] == "explain"
        assert result["response"] == {"mode": "explain", "text": "Mirror response"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_bounded_server_backpressure_and_pool():
    from .service import BackendService
    service = BackendService(engine_factory=FakeEngine)
    server = create_server(port=0, service=service, max_workers=2, request_queue_size=2)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert server.max_workers == 2
        assert server.request_queue_size == 2
        status, health = request(server, "GET", "/api/health")
        assert status == 200
        assert health["ok"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_non_loopback_bind_requires_token_and_specific_cors():
    from .service import BackendService

    try:
        create_server(host="0.0.0.0", port=0, service=BackendService())
    except ValueError as exc:
        assert "MIRROR7_API_TOKEN" in str(exc)
    else:
        raise AssertionError("public bind without token must fail closed")

    try:
        create_server(
            host="0.0.0.0",
            port=0,
            service=BackendService(),
            api_token="x" * 40,
        )
    except ValueError as exc:
        assert "CORS" in str(exc)
    else:
        raise AssertionError("public wildcard CORS must fail closed")


def test_api_token_protects_sensitive_routes_but_not_liveness():
    from .service import BackendService

    token = "mirror7-test-token-" + ("x" * 40)
    server = create_server(port=0, service=BackendService(), api_token=token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, health = request(server, "GET", "/api/health")
        assert status == 200
        assert health["ok"] is True

        unauthenticated = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/api/sessions",
            method="GET",
        )
        try:
            urllib.request.urlopen(unauthenticated, timeout=3)
        except urllib.error.HTTPError as exc:
            assert exc.code == 401
        else:
            raise AssertionError("sensitive API route must require authentication")

        authenticated = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/api/sessions",
            headers={"Authorization": f"Bearer {token}"},
            method="GET",
        )
        with urllib.request.urlopen(authenticated, timeout=3) as response:
            payload = json.loads(response.read())
            assert response.status == 200
            assert payload["sessions"] == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_tdd_generate_route_uses_native_builder_and_rejects_unsupported_requests():
    from .service import BackendService

    service = BackendService(enable_crystal_auto_load=False)
    server = create_server(port=0, service=service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, unsupported = request(
            server,
            "POST",
            "/api/tdd/generate",
            {"prompt": "Write a sorting function"},
        )
        assert status == 200
        assert unsupported["ok"] is False
        assert unsupported["verified"] is False
        assert "does not yet support" in unsupported["error"]

        status, generated = request(
            server,
            "POST",
            "/api/tdd/generate",
            {"prompt": "Write Python code to add two numbers"},
        )
        assert status == 200
        assert generated["ok"] is True
        assert generated["verified"] is True
        assert "def add(a, b)" in generated["final_code"]
        assert generated["verification"]["tests_generated"] == 3
        assert generated["verification"]["tests_passed"] == 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_normal_qna_returns_native_code_only_after_tdd_passes():
    from .service import BackendService

    service = BackendService(enable_crystal_auto_load=False)
    server = create_server(port=0, service=service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, result = request(
            server,
            "POST",
            "/api/crystal/ask",
            {"prompt": "Write Python code to add two numbers"},
        )
        assert status == 200
        assert result["answer_qualification"] == "verified_task_result"
        assert result["task_result_verified"] is True
        assert result["task_code_verified"] is True
        assert "def add(a, b)" in result["answer"]
        assert result["code_verification"]["tests_generated"] == 3
        assert result["code_verification"]["tests_passed"] == 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
