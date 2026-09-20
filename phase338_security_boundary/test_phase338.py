from __future__ import annotations
import http.client
import json
import threading
import urllib.error
import urllib.request
from mirror7_backend.http_server import create_server
from mirror7_backend.service import BackendService

class Engine:
    def step(self, observation, goal=None, research_tasks=(), views=()):
        return {'observation': observation}

def req(server, method, path, body=None, content_type='application/json'):
    data = None if body is None else json.dumps(body).encode()
    headers = {} if body is None else {'Content-Type': content_type}
    request = urllib.request.Request(
        f'http://127.0.0.1:{server.server_port}{path}',
        data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

def test_http_requires_json_for_body_requests():
    server=create_server(port=0,service=BackendService(engine_factory=Engine)); t=threading.Thread(target=server.serve_forever,daemon=True); t.start()
    try:
        status,body=req(server,'POST','/api/sessions',{'session_id':'x'},'text/plain')
        assert status==400 and 'Content-Type' in body['error']
    finally: server.shutdown(); server.server_close(); t.join(timeout=3)

def test_http_rejects_oversized_body():
    server=create_server(port=0,service=BackendService(engine_factory=Engine)); t=threading.Thread(target=server.serve_forever,daemon=True); t.start()
    try:
        conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=3)
        conn.request('POST','/api/sessions',body=None,headers={
            'Content-Type':'application/json',
            'Content-Length':str(1024*1024+1),
        })
        response=conn.getresponse()
        response.read()
        assert response.status==400
        conn.close()
    finally: server.shutdown(); server.server_close(); t.join(timeout=3)
