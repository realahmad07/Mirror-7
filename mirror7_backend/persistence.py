from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
import hashlib
import json
import os
from typing import Any


class CheckpointError(RuntimeError):
    pass


class CheckpointStore:
    """Atomic, integrity-checked JSON checkpoint store."""

    FORMAT = "mirror7-checkpoint-v1"

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, session_id: str) -> Path:
        if not session_id or "/" in session_id or "\\" in session_id or session_id in {".", ".."}:
            raise ValueError("invalid session id")
        return self.root / f"{session_id}.json"

    def save(self, session_id: str, snapshot: dict[str, Any]) -> Path:
        path = self.path_for(session_id)
        body = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str)
        envelope = {
            "format": self.FORMAT,
            "payload": snapshot,
            "sha256": hashlib.sha256(body.encode()).hexdigest(),
        }
        encoded = json.dumps(envelope, sort_keys=True, indent=2).encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("wb", dir=path.parent, delete=False) as tmp:
            tmp.write(encoded)
            tmp.flush()
            os.fsync(tmp.fileno())
            temp_name = tmp.name
        try:
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return path

    def load(self, session_id: str) -> dict[str, Any]:
        path = self.path_for(session_id)
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CheckpointError(f"cannot read checkpoint: {path}") from exc
        if envelope.get("format") != self.FORMAT:
            raise CheckpointError("unsupported checkpoint format")
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise CheckpointError("checkpoint payload must be an object")
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        expected = hashlib.sha256(body.encode()).hexdigest()
        if envelope.get("sha256") != expected:
            raise CheckpointError("checkpoint integrity check failed")
        return payload

    def delete(self, session_id: str) -> None:
        self.path_for(session_id).unlink(missing_ok=True)
