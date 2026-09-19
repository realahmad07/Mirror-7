from __future__ import annotations

import json
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

from mirror7_backend.http_server import create_server
from mirror7_backend.persistence import CheckpointStore
from mirror7_backend.service import BackendService

class Engine:
    def step(self, observations, goal=None, research_tasks=(), views=()):
        return {'goal': goal, 'observation': observations}

def call(server, method, path, payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(f'http://127.0.0.1:{server.server_port}{path}', data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

def test_complete_session_and_checkpoint_lifecycle():
    with tempfile.TemporaryDirectory() as root:
        service = BackendService(engine_factory=Engine, checkpoint_store=CheckpointStore(root))
        server = create_server(port=0, service=service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            assert call(server, 'POST', '/api/sessions', {'session_id': 'api'})[0] == 201
            assert call(server, 'POST', '/api/sessions/api/step', {'observation': {'x': 1}})[1]['sequence'] == 1
            status, snap = call(server, 'GET', '/api/sessions/api')
            assert status == 200 and snap['snapshot']['sequence'] == 1
            status, saved = call(server, 'POST', '/api/sessions/api/save', {})
            assert status == 200 and Path(saved['checkpoint']).is_file()
            status, deleted = call(server, 'DELETE', '/api/sessions/api/checkpoint')
            assert status == 200 and deleted['deleted'] is True
            status, deleted_again = call(server, 'DELETE', '/api/sessions/api/checkpoint')
            assert status == 200 and deleted_again['deleted'] is False
            status, restored = call(server, 'POST', '/api/sessions/api/restore', {})
            assert status == 409
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=3)

def test_session_capacity_is_enforced():
    service = BackendService(engine_factory=Engine, max_sessions=1)
    service.create_session('one')
    try:
        service.create_session('two')
    except RuntimeError as exc:
        assert 'capacity' in str(exc)
    else:
        raise AssertionError('capacity limit not enforced')
