from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlparse

from .actions import ActionDenied
from .persistence import CheckpointError
from .service import BackendService


class MirrorAPIHandler(BaseHTTPRequestHandler):
    service: BackendService | None = None
    server_version = "Mirror7HTTP/1.1"
    max_body_bytes = 1_048_576
    cors_origin = "*"

    def _json_safe(self, value: Any) -> Any:
        if is_dataclass(value):
            return {k: self._json_safe(v) for k, v in asdict(value).items()}
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        return value

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        safe_payload = self._json_safe(payload)
        body = json.dumps(safe_payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", self.cors_origin)
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 0 or length > self.max_body_bytes:
            raise ValueError("request body exceeds 1 MiB limit")
        if length == 0:
            return {}
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise ValueError("Content-Type must be application/json")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _session_id(self, raw: str) -> str:
        session_id = unquote(raw).strip()
        if not session_id or len(session_id) > 128:
            raise ValueError("invalid session_id")
        if any(ord(ch) < 32 for ch in session_id):
            raise ValueError("invalid session_id")
        return session_id

    def do_OPTIONS(self) -> None:
        self._send(HTTPStatus.NO_CONTENT, {})

    def do_GET(self) -> None:
        service = self.service
        if service is None:
            self._send(500, {"ok": False, "error": "service unavailable"})
            return
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send(200, {"ok": True, "service": "mirror7", "status": service.status()})
            return
        if path == "/api/status":
            self._send(200, dict(service.status()))
            return
        if path == "/api/sessions":
            self._send(200, {"ok": True, "sessions": service.list_sessions()})
            return
        parts = [p for p in path.split("/") if p]
        if len(parts) == 3 and parts[:2] == ["api", "sessions"]:
            session_id = self._session_id(parts[2])
            self._send(200, {"ok": True, "snapshot": service.session_snapshot(session_id)})
            return
        self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        service = self.service
        if service is None:
            self._send(500, {"ok": False, "error": "service unavailable"})
            return
        try:
            payload = self._read_json()
            path = urlparse(self.path).path
            parts = [p for p in path.split("/") if p]

            if parts == ["api", "sessions"]:
                session_id = self._session_id(str(payload.get("session_id", "")))
                session = service.create_session(session_id)
                self._send(201, {"ok": True, "session_id": session.session_id})
                return

            if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "step":
                session_id = self._session_id(parts[2])
                result = service.step(session_id, payload.get("observation"), goal=payload.get("goal"))
                self._send(200, {
                    "ok": True,
                    "session_id": result.session_id,
                    "sequence": result.sequence,
                    "observation": result.observation,
                    "goal": result.goal,
                    "engine_result": result.engine_result,
                    "state_digest": result.state_digest,
                })
                return

            if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "save":
                session_id = self._session_id(parts[2])
                path = service.save(session_id)
                self._send(200, {"ok": True, "session_id": session_id, "checkpoint": str(path)})
                return

            if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "action":
                session_id = self._session_id(parts[2])
                name = payload.get("name")
                args = payload.get("args", [])
                kwargs = payload.get("kwargs", {})
                if not isinstance(name, str) or not name:
                    raise ValueError("action name must be a non-empty string")
                if not isinstance(args, list):
                    raise ValueError("action args must be a JSON array")
                if not isinstance(kwargs, dict):
                    raise ValueError("action kwargs must be a JSON object")
                result = service.execute_action(name, *args, **kwargs)
                self._send(200, {"ok": True, "session_id": session_id, "result": result})
                return

            if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "restore":
                session_id = self._session_id(parts[2])
                session = service.restore(session_id)
                self._send(200, {"ok": True, "session_id": session.session_id})
                return

            self._send(404, {"ok": False, "error": "not found"})
        except KeyError as exc:
            self._send(404, {"ok": False, "error": str(exc)})
        except ValueError as exc:
            self._send(400, {"ok": False, "error": str(exc)})
        except ActionDenied as exc:
            self._send(403, {"ok": False, "error": str(exc)})
        except CheckpointError as exc:
            self._send(409, {"ok": False, "error": str(exc)})
        except RuntimeError as exc:
            self._send(409, {"ok": False, "error": str(exc)})
        except Exception:
            self._send(500, {"ok": False, "error": "internal server error"})

    def do_DELETE(self) -> None:
        service = self.service
        if service is None:
            self._send(500, {"ok": False, "error": "service unavailable"})
            return
        try:
            path = urlparse(self.path).path
            parts = [p for p in path.split("/") if p]
            if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "checkpoint":
                session_id = self._session_id(parts[2])
                deleted = service.has_checkpoint(session_id)
                service.delete_checkpoint(session_id)
                self._send(200, {"ok": True, "session_id": session_id, "deleted": deleted})
                return
            if len(parts) == 3 and parts[:2] == ["api", "sessions"]:
                session_id = self._session_id(parts[2])
                service.close_session(session_id)
                self._send(200, {"ok": True, "session_id": session_id, "closed": True})
                return
            self._send(404, {"ok": False, "error": "not found"})
        except KeyError as exc:
            self._send(404, {"ok": False, "error": str(exc)})
        except ValueError as exc:
            self._send(400, {"ok": False, "error": str(exc)})
        except Exception:
            self._send(500, {"ok": False, "error": "internal server error"})

    def log_message(self, format: str, *args: Any) -> None:
        return


def create_server(host: str = "127.0.0.1", port: int = 8787, service: BackendService | None = None, cors_origin: str = "*"):
    if not cors_origin or "\n" in cors_origin or "\r" in cors_origin:
        raise ValueError("invalid cors_origin")
    MirrorAPIHandler.service = service or BackendService()
    MirrorAPIHandler.cors_origin = cors_origin
    return ThreadingHTTPServer((host, port), MirrorAPIHandler)


def serve(host: str | None = None, port: int | None = None) -> None:
    resolved_host = host if host is not None else os.getenv("MIRROR7_HOST", "127.0.0.1")
    resolved_port = port if port is not None else int(os.getenv("MIRROR7_PORT", "8787"))
    server = create_server(resolved_host, resolved_port)
    print(f"Mirror 7 API listening on http://{resolved_host}:{resolved_port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    serve()
