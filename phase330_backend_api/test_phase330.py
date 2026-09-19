from __future__ import annotations
import json, threading, urllib.request
from mirror7_backend.service import BackendService
from mirror7_backend.http_server import create_server

class FakeEngine:
    def __init__(self): self.n = 0
    def step(self, observation, *, goal=None, research_tasks=(), views=()):
        self.n += 1
        return {"n": self.n, "observation": observation, "goal": goal}

def req(server, method, path, payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(f"http://127.0.0.1:{server.server_port}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(r, timeout=3) as x:
        return x.status, json.loads(x.read())

def test_phase330_api_surface():
    service = BackendService(engine_factory=FakeEngine)
    server = create_server(port=0, service=service, cors_origin="http://localhost:3000")
    t = threading.Thread(target=server.serve_forever, daemon=True); t.start()
    try:
        s,b=req(server,"POST","/api/sessions",{"session_id":"p330"}); assert s==201
        s,b=req(server,"GET","/api/sessions"); assert s==200 and b["sessions"]==["p330"]
        s,b=req(server,"POST","/api/sessions/p330/step",{"observation":{"x":1}}); assert s==200 and b["sequence"]==1
        s,b=req(server,"GET","/api/sessions"); assert b["sessions"]==["p330"]
        s,b=req(server,"DELETE","/api/sessions/p330"); assert s==200 and b["closed"] is True
    finally:
        server.shutdown(); server.server_close(); t.join(timeout=2)

def test_service_listing_and_checkpoint_delete():
    service = BackendService(engine_factory=FakeEngine)
    service.create_session("a"); service.create_session("b")
    assert service.list_sessions() == ("a","b")
