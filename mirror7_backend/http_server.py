from __future__ import annotations

import hmac
import ipaddress
import json
import os
from dataclasses import asdict, is_dataclass
from http import HTTPStatus
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
import threading
import time
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
import uuid

from .actions import ActionDenied
from .persistence import CheckpointError
from .service import BackendService


class MirrorAPIHandler(BaseHTTPRequestHandler):
    service: BackendService | None = None
    server_version = "Mirror7HTTP/1.1"
    max_body_bytes = 1_048_576
    cors_origin = "*"
    api_token: str | None = None

    def _get_request_id(self) -> str:
        if not hasattr(self, "_cached_request_id"):
            self._cached_request_id = (
                self.headers.get("X-Request-ID")
                or self.headers.get("X-Trace-ID")
                or str(uuid.uuid4())
            )
        return self._cached_request_id

    def _authorize_request(self, path: str) -> bool:
        # Liveness, readiness, healthz, and metrics probes are unauthenticated
        if self.command == "GET" and path in {
            "/api/health", "/api/livez", "/api/readyz", "/api/healthz", "/metrics", "/api/metrics"
        }:
            return True
        if not path.startswith("/api/") or self.api_token is None:
            return True
        supplied = self.headers.get("Authorization", "")
        expected = "Bearer " + self.api_token
        if not hmac.compare_digest(supplied, expected):
            self._send(HTTPStatus.UNAUTHORIZED, {
                "ok": False,
                "error": "authentication required",
            })
            return False
        return True

    def _json_safe(self, value: Any) -> Any:
        if is_dataclass(value):
            return {k: self._json_safe(v) for k, v in asdict(value).items()}
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        return value

    def _send_bytes(self, status: int, data: bytes, content_type: str = "text/html; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", self.cors_origin)
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Request-ID, X-Trace-ID")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Request-ID", self._get_request_id())
        self.end_headers()
        self.wfile.write(data)

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        safe_payload = self._json_safe(payload)
        body = b"" if status == HTTPStatus.NO_CONTENT else json.dumps(
            safe_payload, ensure_ascii=False, default=str
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", self.cors_origin)
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Request-ID, X-Trace-ID")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Request-ID", self._get_request_id())
        self.end_headers()
        if body:
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

    def _handle_sse_stream(self, prompt: str, session_id: str = "studio_stream") -> None:
        service = self.service
        if service is None:
            self._send(500, {"ok": False, "error": "service unavailable"})
            return
        res = service.crystal_ask(prompt, session_id=session_id)
        full_text = res.get("answer", "")

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", self.cors_origin)
        self.end_headers()

        tokens = full_text.split(" ")
        for i, tok in enumerate(tokens):
            chunk = tok + (" " if i < len(tokens) - 1 else "")
            evt = json.dumps({"chunk": chunk, "done": False}, ensure_ascii=False)
            self.wfile.write(f"data: {evt}\n\n".encode("utf-8"))
            self.wfile.flush()

        final_meta = {
            "done": True,
            "confidence": res.get("confidence", 0.0),
            "matched_rule_id": res.get("matched_rule_id"),
            "mode": res.get("mode"),
            "latency_ms": res.get("latency_ms", 0.0),
            "epistemic_tier": res.get("epistemic_tier"),
            "verified": bool(res.get("verified", False)),
            "abstained": bool(res.get("abstained", True)),
            "answer_qualification": res.get("answer_qualification", "unknown"),
        }
        self.wfile.write(f"data: {json.dumps(final_meta, ensure_ascii=False)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_OPTIONS(self) -> None:
        self._send(HTTPStatus.NO_CONTENT, {})

    def do_GET(self) -> None:
        service = self.service
        if service is None:
            self._send(500, {"ok": False, "error": "service unavailable"})
            return
        path = urlparse(self.path).path
        if not self._authorize_request(path):
            return
        if path in {"/", "/studio", "/studio.html"}:
            from pathlib import Path
            studio_p = Path(__file__).parent / "studio.html"
            if studio_p.exists():
                self._send_bytes(200, studio_p.read_bytes(), "text/html; charset=utf-8")
                return
            self._send(404, {"ok": False, "error": "studio.html not found"})
            return
        if path == "/api/crystal/wasm":
            from pathlib import Path
            from mirror_crystal import WasmPocketCompiler
            wasm_p = Path("artifacts/mirror_crystal_pocket_popcnt.wasm")
            wasm_bytes = wasm_p.read_bytes() if wasm_p.exists() else WasmPocketCompiler.compile_pocket_wasm_bytes()
            self._send_bytes(200, wasm_bytes, "application/wasm")
            return
        if path == "/api/crystal/status":
            st = dict(service.status())
            self._send(200, {"ok": True, "crystal": st.get("crystal", {})})
            return
        if path == "/api/crystal/coverage":
            self._send(200, service.crystal_coverage_map())
            return
        if path == "/api/crystal/lattice":
            self._send(200, {"ok": True, "nodes": service.get_visual_lattice_nodes()})
            return
        if path == "/api/memory/gdpr/export":
            parsed_qs = parse_qs(urlparse(self.path).query)
            sess = parsed_qs.get("session_id", ["studio_web"])[0]
            self._send(200, {"ok": True, "export": service.memory_export_gdpr(sess)})
            return
        if path == "/api/streamer/status":
            self._send(200, service.streamer_status())
            return
        if path == "/api/memory/entities":
            parsed_qs = parse_qs(urlparse(self.path).query)
            sess = parsed_qs.get("session_id", ["studio_web"])[0]
            etype = parsed_qs.get("entity_type", [None])[0]
            self._send(200, {"ok": True, "entities": service.memory_list(sess, entity_type=etype)})
            return
        if path == "/api/memory/profile":
            parsed_qs = parse_qs(urlparse(self.path).query)
            sess = parsed_qs.get("session_id", ["studio_web"])[0]
            self._send(200, {"ok": True, "profile": service.memory_get_profile(sess)})
            return
        if path == "/api/crystal/gaps":
            self._send(200, {"ok": True, "gaps": service.crystal_get_gaps()})
            return
        if path == "/api/crystal/gap_prompt":
            parsed_qs = parse_qs(urlparse(self.path).query)
            t = parsed_qs.get("topic", [None])[0]
            self._send(200, {"ok": True, "prompt": service.crystal_gap_prompt(topic=t)})
            return
        if path == "/api/livez":
            self._send(200, {"ok": True, "status": "alive", "service": "mirror7"})
            return
        if path == "/api/readyz":
            ready = getattr(service, "is_ready", True)
            self._send(200 if ready else 503, {
                "ok": ready,
                "status": "ready" if ready else "not_ready",
                "service": "mirror7",
            })
            return
        if path == "/api/healthz":
            active_sessions = len(service.list_sessions()) if hasattr(service, "list_sessions") else 0
            st = dict(service.status()) if hasattr(service, "status") else {}
            srv_start = getattr(self.server, "start_time", time.time())
            self._send(200, {
                "ok": True,
                "service": "mirror7",
                "ready": getattr(service, "is_ready", True),
                "uptime_seconds": round(time.time() - srv_start, 2),
                "sessions": active_sessions,
                "crystal": st.get("crystal", {}),
            })
            return
        if path in {"/metrics", "/api/metrics"}:
            from mirror_crystal.production_readiness import MetricsCollector
            metrics = getattr(self.server, "metrics", None) or MetricsCollector()
            self._send_bytes(200, metrics.render_prometheus_text().encode("utf-8"), "text/plain; version=0.0.4; charset=utf-8")
            return
        if path == "/api/health":
            active_sessions = len(service.list_sessions()) if hasattr(service, "list_sessions") else 0
            self._send(200, {
                "ok": True,
                "service": "mirror7",
                "ready": getattr(service, "is_ready", True),
                "warmup_ms": getattr(service, "warmup_ms", 0.0),
                "status": {"sessions": active_sessions},
            })
            return
        if path == "/api/crystal/stream":
            qs = parse_qs(urlparse(self.path).query)
            prompt = qs.get("q", [""])[0] or qs.get("prompt", [""])[0]
            session_id = qs.get("session_id", ["studio_stream"])[0]
            self._handle_sse_stream(prompt, session_id=session_id)
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
        path = urlparse(self.path).path
        if not self._authorize_request(path):
            return
        try:
            payload = self._read_json()
            parts = [p for p in path.split("/") if p]

            if parts == ["api", "crystal", "stream"]:
                prompt = str(payload.get("prompt", "") or payload.get("query", ""))
                session_id = str(payload.get("session_id", "studio_stream"))
                self._handle_sse_stream(prompt, session_id=session_id)
                return

            if parts == ["api", "crystal", "ingest_path"]:
                path_to_ingest = str(payload.get("path", ""))
                dom = payload.get("domain")
                res = service.crystal_ingest_path(path_to_ingest, domain=dom)
                self._send(200, res)
                return

            if parts == ["api", "memory", "consolidate"]:
                min_conf = float(payload.get("min_confidence", 0.85))
                res = service.memory_dream_consolidation(min_confidence=min_conf)
                self._send(200, res)
                return

            if parts == ["api", "universal", "analyze"]:
                prompt = str(payload.get("prompt", ""))
                options = payload.get("options", {})
                if not isinstance(options, dict):
                    raise ValueError("options must be a JSON object")
                res = service.universal_analyze(prompt, **options)
                self._send(200, {"ok": True, "universal_intelligence": res})
                return

            if parts == ["api", "universal", "synthesize"]:
                goal = payload.get("goal")
                if not isinstance(goal, dict):
                    raise ValueError("goal must be a JSON object")
                res = service.universal_synthesize(goal)
                self._send(200, {"ok": True, "uol_synthesis": res})
                return

            if parts == ["api", "crystal", "ask"]:
                prompt = str(payload.get("prompt", ""))
                session_id = str(payload.get("session_id", "studio_web"))
                consent_log = bool(payload.get("consent_log", False))
                universal_options = payload.get("universal_options", {})
                if not isinstance(universal_options, dict):
                    raise ValueError("universal_options must be a JSON object")
                res = service.crystal_ask(
                    prompt, session_id=session_id, consent_log=consent_log,
                    universal_options=universal_options,
                )
                self._send(200, res)
                return

            if parts in (["api", "safety", "evaluate"], ["api", "v1", "safety", "evaluate"]):
                prompt = str(payload.get("prompt", "") or payload.get("text", ""))
                from mirror_crystal.safety_alignment import SafetyAlignmentManager
                mgr = SafetyAlignmentManager()
                result = mgr.evaluate_request(prompt)
                self._send(200, {
                    "ok": True,
                    "is_safe": result.is_safe,
                    "category": result.category.value,
                    "risk_score": result.risk_score,
                    "violations": result.detected_violations,
                    "tainted": result.tainted,
                    "refusal": result.refusal_contract.format_full_response() if result.refusal_contract else None,
                    "latency_ms": result.latency_ms,
                })
                return

            if parts in (["api", "safety", "sanitize"], ["api", "v1", "safety", "sanitize"]):
                prompt = str(payload.get("prompt", "") or payload.get("text", ""))
                from mirror_crystal.safety_alignment import PrivacyProtectionEngine
                priv = PrivacyProtectionEngine()
                sanitized, counts = priv.scrub_pii_and_credentials(prompt)
                self._send(200, {
                    "ok": True,
                    "sanitized_text": sanitized,
                    "redacted_counts": counts,
                    "total_redacted": sum(counts.values()),
                })
                return

            if parts in (["api", "evidence", "verify"], ["api", "v1", "evidence", "verify"]):
                query = str(payload.get("query", ""))
                raw_claims = payload.get("claims", [])
                raw_sources = payload.get("sources", [])
                raw_passages = payload.get("passages", [])
                threshold = float(payload.get("confidence_threshold", 0.55))

                from mirror_crystal.evidence_intelligence import (
                    EvidenceBasedIntelligenceEngine,
                    SourceProvenance,
                    EvidencePassage,
                    AtomicClaim,
                )

                claims_obj = []
                if isinstance(raw_claims, str):
                    claims_obj = EvidenceBasedIntelligenceEngine.decompose_statements(raw_claims)
                elif isinstance(raw_claims, list):
                    for idx, c in enumerate(raw_claims, 1):
                        if isinstance(c, str):
                            claims_obj.append(AtomicClaim(claim_id=f"c_{idx:03d}", statement=c))
                        elif isinstance(c, dict):
                            claims_obj.append(AtomicClaim(
                                claim_id=c.get("claim_id", f"c_{idx:03d}"),
                                statement=c.get("statement", ""),
                                subject=c.get("subject", ""),
                                predicate=c.get("predicate", ""),
                                object_val=c.get("object_val", ""),
                                is_assumption=bool(c.get("is_assumption", False)),
                            ))

                sources_obj = []
                for s in raw_sources:
                    if isinstance(s, dict):
                        sources_obj.append(SourceProvenance(
                            source_id=s.get("source_id", ""),
                            title=s.get("title", ""),
                            url=s.get("url", ""),
                            publisher=s.get("publisher", ""),
                            license_spdx=s.get("license_spdx", ""),
                            retrieved_at=s.get("retrieved_at", "2026-01-01T00:00:00+00:00"),
                            sha256_hash=s.get("sha256_hash", ""),
                            content=s.get("content", ""),
                        ))

                passages_obj = []
                for p in raw_passages:
                    if isinstance(p, dict):
                        passages_obj.append(EvidencePassage(
                            passage_id=p.get("passage_id", ""),
                            source_id=p.get("source_id", ""),
                            locator=p.get("locator", ""),
                            text=p.get("text", ""),
                            sha256_hash=p.get("sha256_hash", ""),
                            authoritativeness_weight=float(p.get("authoritativeness_weight", 1.0)),
                        ))

                engine = EvidenceBasedIntelligenceEngine()
                report = engine.process_and_synthesize(
                    query=query,
                    claims=claims_obj,
                    sources=sources_obj,
                    passages=passages_obj,
                    confidence_threshold=threshold,
                )
                self._send(200, {
                    "ok": True,
                    "report": report.to_dict(),
                })
                return

            if parts in (["api", "evidence", "contradictions"], ["api", "v1", "evidence", "contradictions"]):
                raw_claims = payload.get("claims", [])
                from mirror_crystal.evidence_intelligence import ContradictionDetector, AtomicClaim

                claims_obj = []
                for idx, c in enumerate(raw_claims, 1):
                    if isinstance(c, str):
                        claims_obj.append(AtomicClaim(claim_id=f"c_{idx:03d}", statement=c))
                    elif isinstance(c, dict):
                        claims_obj.append(AtomicClaim(
                            claim_id=c.get("claim_id", f"c_{idx:03d}"),
                            statement=c.get("statement", ""),
                            subject=c.get("subject", ""),
                            predicate=c.get("predicate", ""),
                        ))

                conflicts = ContradictionDetector.detect_conflicts(claims_obj)
                self._send(200, {
                    "ok": True,
                    "conflicts": [c.to_dict() for c in conflicts],
                    "total_conflicts": len(conflicts),
                })
                return

            if parts in (["api", "crystal", "feedback"], ["api", "crystal", "downvote"]):
                rid = int(payload.get("rule_id", 0))
                pos = bool(payload.get("is_positive", False if parts == ["api", "crystal", "downvote"] else True))
                res = service.crystal_feedback(rid, is_positive=pos)
                self._send(200, res)
                return

            if parts == ["api", "crystal", "purge_domain"]:
                dom = str(payload.get("domain", ""))
                res = service.crystal_purge_domain(dom)
                self._send(200, res)
                return

            if parts == ["api", "crystal", "compact"]:
                res = service.crystal_compact()
                self._send(200, res)
                return

            if parts == ["api", "crystal", "teach"]:
                q = str(payload.get("question", "") or payload.get("prompt", ""))
                a = str(payload.get("answer", "") or payload.get("response", ""))
                tc = str(payload.get("test_code", ""))
                dom = str(payload.get("domain", "studio_teach"))
                added = service.crystal_teach(q, a, test_code=tc, domain=dom)
                self._send(200, {"ok": True, "crystallized": added})
                return

            if parts == ["api", "crystal", "teleport"]:
                scrubbed = str(payload.get("scrubbed_prompt", "") or payload.get("prompt", ""))
                dev_id = str(payload.get("device_id", "pocket_studio"))
                pkt = service.crystal_teleport(scrubbed, device_id=dev_id)
                self._send(200, pkt)
                return

            if parts == ["api", "crystal", "crystallize_doc"]:
                txt = str(payload.get("text", "") or payload.get("content", ""))
                title = str(payload.get("title", "StudioDoc"))
                stats = service.crystal_auto_ingest_document(txt, title=title)
                self._send(200, {"ok": True, "stats": stats})
                return

            if parts == ["api", "cerebrum", "solve"]:
                prompt = str(payload.get("prompt", ""))
                res = service.cerebrum_solve(prompt)
                self._send(200, res)
                return

            if parts == ["api", "tdd", "synthesize"]:
                code = str(payload.get("code", "") or payload.get("candidate_code", ""))
                spec = str(payload.get("spec", "") or payload.get("spec_desc", ""))
                tests = str(payload.get("tests", "") or payload.get("explicit_tests", ""))
                res = service.tdd_synthesize(code, spec_desc=spec, explicit_tests=tests)
                self._send(200, res)
                return

            if parts == ["api", "tdd", "generate"]:
                prompt = str(payload.get("prompt", ""))
                spec = str(payload.get("spec", "") or payload.get("spec_desc", ""))
                tests = str(payload.get("tests", "") or payload.get("explicit_tests", ""))
                res = service.tdd_generate_and_verify(
                    prompt, spec_desc=spec, explicit_tests=tests
                )
                self._send(200, res)
                return

            if parts == ["api", "data", "six_gate_audit"]:
                text = str(payload.get("text", "") or payload.get("content", ""))
                spdx = str(payload.get("spdx", "MIT") or payload.get("license", "MIT"))
                res = service.six_gate_audit(text, spdx_license=spdx)
                self._send(200, res)
                return

            if parts == ["api", "streamer", "start"]:
                sec = float(payload.get("interval_seconds", 30.0))
                service.streamer_start(interval_seconds=sec)
                self._send(200, service.streamer_status())
                return

            if parts == ["api", "streamer", "stop"]:
                service.streamer_stop()
                self._send(200, service.streamer_status())
                return

            if parts == ["api", "streamer", "pulse"]:
                res = service.streamer_pulse()
                self._send(200, res)
                return

            if parts == ["api", "memory", "remember"]:
                sess = str(payload.get("session_id", "studio_web"))
                k = str(payload.get("key", ""))
                v = payload.get("value")
                etype = str(payload.get("entity_type", "general_preference"))
                conf = float(payload.get("confidence", 1.0))
                ent = service.memory_remember(sess, k, v, entity_type=etype, confidence=conf)
                self._send(200, {"ok": True, "entity": ent})
                return

            if parts == ["api", "memory", "forget"]:
                sess = str(payload.get("session_id", "studio_web"))
                k = str(payload.get("key", ""))
                deleted = service.memory_forget(sess, k)
                self._send(200, {"ok": True, "deleted": deleted})
                return

            if parts == ["api", "memory", "undo"]:
                sess = str(payload.get("session_id", "studio_web"))
                undone = service.memory_undo(sess)
                self._send(200, {"ok": True, "undone": undone})
                return

            if parts == ["api", "memory", "confirm"]:
                sess = str(payload.get("session_id", "studio_web"))
                k = str(payload.get("key", ""))
                confirmed = service.memory_confirm(sess, k)
                self._send(200, {"ok": True, "confirmed": confirmed})
                return

            if parts == ["api", "memory", "gdpr", "purge"]:
                sess = str(payload.get("session_id", "studio_web"))
                purged = service.memory_purge_gdpr(sess)
                self._send(200, {"ok": True, "purged": purged})
                return

            if parts == ["api", "memory", "compact"]:
                sess = payload.get("session_id")
                compacted = service.memory_compact(session_id=str(sess) if sess else None)
                self._send(200, {"ok": True, "compacted": compacted})
                return

            if parts == ["api", "memory", "profile"]:
                sess = str(payload.get("session_id", "studio_web"))
                name = str(payload.get("name", "DefaultProject"))
                root = str(payload.get("root_dir", ""))
                fw = str(payload.get("framework", ""))
                db = str(payload.get("database", ""))
                p_raw = payload.get("port")
                port = int(p_raw) if p_raw is not None else None
                constraints = payload.get("constraints")
                prof = service.memory_set_profile(
                    sess, name=name, root_dir=root, framework=fw, database=db, port=port, constraints=constraints
                )
                self._send(200, {"ok": True, "profile": prof})
                return

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
                    "realization_contract": result.realization_contract,
                    "response": result.response,
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
        path = urlparse(self.path).path
        if not self._authorize_request(path):
            return
        try:
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


class BoundedThreadPoolServer(HTTPServer):
    """
    Production-grade HTTP server utilizing a bounded worker pool and backpressure queue.
    Guarantees fixed thread boundaries to prevent thread-exhaustion and slowloris attacks.
    Immediately sheds excess traffic with HTTP 503 Service Unavailable when the queue is saturated.
    """

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass: type[BaseHTTPRequestHandler],
        max_workers: int = 32,
        request_queue_size: int = 256,
        socket_timeout: float = 30.0,
        bind_and_activate: bool = True,
    ) -> None:
        super().__init__(server_address, RequestHandlerClass, bind_and_activate=bind_and_activate)
        self.max_workers = int(max(1, max_workers))
        self.request_queue_size = int(max(1, request_queue_size))
        self.socket_timeout = float(max(0.5, socket_timeout))
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="mirror7_worker",
        )
        self._request_semaphore = threading.Semaphore(self.max_workers + self.request_queue_size)
        self._is_shutting_down = False
        self.start_time = time.time()
        from mirror_crystal.production_readiness import MetricsCollector
        self.metrics = MetricsCollector()

    def process_request(self, request: Any, client_address: Any) -> None:
        if self._is_shutting_down:
            self._reject_overloaded(request, 503, "Server is shutting down")
            return

        acquired = self._request_semaphore.acquire(blocking=False)
        if not acquired:
            self._reject_overloaded(request, 503, "Worker queue saturated - please retry shortly")
            return

        try:
            if hasattr(request, "settimeout"):
                request.settimeout(self.socket_timeout)
            self.executor.submit(self._worker_process, request, client_address)
        except Exception:
            self._request_semaphore.release()
            self._reject_overloaded(request, 503, "Failed to schedule worker thread")

    def _reject_overloaded(self, request: Any, status: int, message: str) -> None:
        try:
            if hasattr(request, "settimeout"):
                request.settimeout(2.0)
            body = f'{{"ok":false,"error":"{message}"}}'.encode("utf-8")
            header = (
                f"HTTP/1.1 {status} Service Unavailable\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"Retry-After: 2\r\n"
                f"Connection: close\r\n\r\n"
            ).encode("utf-8")
            request.sendall(header + body)
        except Exception:
            pass
        finally:
            try:
                self.shutdown_request(request)
            except Exception:
                pass

    def _worker_process(self, request: Any, client_address: Any) -> None:
        try:
            self.finish_request(request, client_address)
            self.shutdown_request(request)
        except Exception:
            self.handle_error(request, client_address)
            try:
                self.shutdown_request(request)
            except Exception:
                pass
        finally:
            self._request_semaphore.release()

    def server_close(self) -> None:
        self._is_shutting_down = True
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        super().server_close()


def create_server(
    host: str = "127.0.0.1",
    port: int = 8787,
    service: BackendService | None = None,
    cors_origin: str = "*",
    max_workers: int = 32,
    request_queue_size: int = 256,
    socket_timeout: float = 30.0,
    api_token: str | None = None,
):
    if not cors_origin or chr(13) in cors_origin or chr(10) in cors_origin:
        raise ValueError("invalid cors_origin")
    token = api_token if api_token is not None else os.getenv("MIRROR7_API_TOKEN")
    token = token.strip() if token else None
    try:
        is_loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = host.casefold() == "localhost"
    if not is_loopback and not token:
        raise ValueError("MIRROR7_API_TOKEN is required when binding beyond loopback")
    if token is not None and len(token) < 32:
        raise ValueError("API token must contain at least 32 characters")
    if not is_loopback and cors_origin == "*":
        raise ValueError("public deployments must configure a specific CORS origin")
    MirrorAPIHandler.service = service or BackendService()
    MirrorAPIHandler.cors_origin = cors_origin
    MirrorAPIHandler.api_token = token
    return BoundedThreadPoolServer(
        (host, port),
        MirrorAPIHandler,
        max_workers=max_workers,
        request_queue_size=request_queue_size,
        socket_timeout=socket_timeout,
    )


def serve(host: str | None = None, port: int | None = None) -> None:
    resolved_host = host if host is not None else os.getenv("MIRROR7_HOST", "127.0.0.1")
    resolved_port = port if port is not None else int(os.getenv("MIRROR7_PORT", "8787"))
    resolved_cors = os.getenv("MIRROR7_CORS_ORIGIN", "*")
    api_token = os.getenv("MIRROR7_API_TOKEN")
    server = create_server(
        resolved_host, resolved_port, cors_origin=resolved_cors, api_token=api_token
    )
    print(f"Mirror 7 API listening on http://{resolved_host}:{resolved_port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    serve()
