from __future__ import annotations

from typing import Any, Mapping
import copy
import hashlib
import json

from mirror7_backend.persistence import CheckpointError, CheckpointStore
from mirror7_backend.service import BackendService


def canonical_digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(data).hexdigest()


class HardenedBackend:
    """Defensive adapter around BackendService.

    It validates externally supplied identifiers/snapshots and returns copies
    so callers cannot mutate backend-owned structures through returned values.
    """

    def __init__(self, service: BackendService):
        self.service = service

    @staticmethod
    def validate_session_id(session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if len(session_id) > 128:
            raise ValueError("session_id too long")
        if any(ord(ch) < 32 for ch in session_id):
            raise ValueError("session_id contains control characters")
        return session_id

    def status(self) -> Mapping[str, Any]:
        return copy.deepcopy(dict(self.service.status()))

    def snapshot(self, session_id: str) -> dict[str, Any]:
        self.validate_session_id(session_id)
        return copy.deepcopy(self.service.get_session(session_id).snapshot())

    def snapshot_digest(self, session_id: str) -> str:
        return canonical_digest(self.snapshot(session_id))

    def restore(self, session_id: str):
        self.validate_session_id(session_id)
        try:
            return self.service.restore(session_id)
        except CheckpointError:
            raise
        except (ValueError, RuntimeError):
            raise

    def load_checkpoint(self, session_id: str) -> dict[str, Any]:
        self.validate_session_id(session_id)
        store = self.service.checkpoint_store
        if store is None:
            raise RuntimeError("checkpoint store is not configured")
        payload = store.load(session_id)
        return copy.deepcopy(payload)
