from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .service import BackendService


class MirrorAPIHandler(BaseHTTPRequestHandler):
    service: BackendService | None = None
    server_version = "Mirror7HTTP/1.0"

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_OPTIONS(self) -> None:
        self._send(204, {})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send(200, {"ok": True, "service": "mirror7", "status": self.service.status()})
            return
        if path == "/api/status":
            self._send(200, dict(self.service.status()))
            return
        self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        service = self.service
        try:
            payload = self._read_json()
            path = urlparse(self.path).path
            parts = [p for p in path.split("/") if p]

            if parts == ["api", "sessions"]:
                session_id = str(payload.get("session_id", "")).strip()
                if not session_id:
                    raise ValueError("session_id is required")
                session = service.create_session(session_id)
                self._send(201, {"ok": True, "session_id": session.session_id})
                return

            if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "step":
                session_id = parts[2]
                observation = payload.get("observation")
                result = service.step(
                    session_id,
                    observation,
                    goal=payload.get("goal"),
                )
                self._send(
                    200,
                    {
                        "ok": True,
                        "session_id": result.session_id,
                        "sequence": result.sequence,
                        "observation": result.observation,
                        "goal": result.goal,
                        "engine_result": result.engine_result,
                        "state_digest": result.state_digest,
                    },
                )
                return

            self._send(404, {"ok": False, "error": "not found"})
        except (ValueError, KeyError) as exc:
            self._send(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send(500, {"ok": False, "error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        return


def create_server(host: str = "127.0.0.1", port: int = 8787, service: BackendService | None = None):
    MirrorAPIHandler.service = service or BackendService()
    return ThreadingHTTPServer((host, port), MirrorAPIHandler)


def serve(host: str = "127.0.0.1", port: int = 8787) -> None:
    server = create_server(host, port)
    print(f"Mirror 7 API listening on http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    serve()
