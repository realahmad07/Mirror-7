from __future__ import annotations
import os
import tempfile
from mirror7_backend.http_server import create_server
from mirror7_backend.persistence import CheckpointStore
from mirror7_backend.service import BackendService

def test_server_environment_defaults_are_parseable(monkeypatch):
    monkeypatch.setenv("MIRROR7_HOST", "127.0.0.1")
    monkeypatch.setenv("MIRROR7_PORT", "0")
    server = create_server(host=os.getenv("MIRROR7_HOST"), port=int(os.getenv("MIRROR7_PORT")))
    try:
        assert server.server_port > 0
    finally:
        server.server_close()

def test_checkpoint_existence_is_explicit():
    with tempfile.TemporaryDirectory() as root:
        service = BackendService(checkpoint_store=CheckpointStore(root))
        assert service.has_checkpoint("missing") is False
        service.checkpoint_store.save("x", {"version": 1})
        assert service.has_checkpoint("x") is True
        service.delete_checkpoint("x")
        assert service.has_checkpoint("x") is False
