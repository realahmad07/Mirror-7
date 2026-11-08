from __future__ import annotations

import json
from pathlib import Path
import threading
import urllib.request

from mirror7_backend.http_server import create_server
from mirror7_backend.service import BackendService
from mirror_crystal import MirrorCrystalEngine


def make_request(server, method: str, path: str, payload: dict | None = None, is_bytes: bool = False):
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(req, timeout=5) as response:
        content = response.read()
        if is_bytes:
            return response.status, content, response.headers.get("Content-Type")
        return response.status, json.loads(content.decode("utf-8")), response.headers.get("Content-Type")


def test_crystal_http_server_full_integration(tmp_path: Path):
    eng = MirrorCrystalEngine()
    eng.ingest_pair("What is Grover search?", "Grover search achieves O(sqrt(N)) speedup.")

    service = BackendService(crystal_vault=eng, enable_crystal_auto_load=False)
    server = create_server(port=0, service=service)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        # 1. Test Studio HTML Serving
        code, html_bytes, ct = make_request(server, "GET", "/", is_bytes=True)
        assert code == 200
        assert b"Mirror Crystal Live Studio" in html_bytes
        assert "text/html" in ct

        code2, html_bytes2, _ = make_request(server, "GET", "/studio", is_bytes=True)
        assert code2 == 200
        assert b"Mirror Crystal Live Studio" in html_bytes2

        # 2. Test Crystal Status Endpoint
        code_st, data_st, _ = make_request(server, "GET", "/api/crystal/status")
        assert code_st == 200
        assert data_st["ok"] is True
        assert data_st["crystal"]["active"] is True
        assert data_st["crystal"]["rule_count"] >= 1

        # 3. Test WASM Binary Serving
        code_w, wasm_b, ct_w = make_request(server, "GET", "/api/crystal/wasm", is_bytes=True)
        assert code_w == 200
        assert wasm_b[:4] == b"\x00asm"
        assert "application/wasm" in ct_w

        # 4. Test Query Ask Endpoint (Direct Lock + CAS)
        code_q, data_q, _ = make_request(server, "POST", "/api/crystal/ask", {"prompt": "What is Grover search?"})
        assert code_q == 200
        assert "O(sqrt(N))" in data_q["answer"]
        assert data_q["epistemic_tier"] in {"verified_exact", "high_confidence"}

        code_cas, data_cas, _ = make_request(server, "POST", "/api/crystal/ask", {"prompt": "Differentiate x^3 + 5*x"})
        assert code_cas == 200
        assert "3*x^2 + 5" in data_cas["answer"]

        # 5. Test Instant 1-Shot Teaching Endpoint
        code_t, data_t, _ = make_request(
            server,
            "POST",
            "/api/crystal/teach",
            {"question": "What is Helios Protocol 9?", "answer": "Helios Protocol 9 initiates reactor scram."},
        )
        assert code_t == 200
        assert data_t["ok"] is True

        code_t_ask, data_t_ask, _ = make_request(server, "POST", "/api/crystal/ask", {"prompt": "What is Helios Protocol 9?"})
        assert "reactor scram" in data_t_ask["answer"]

        # 6. Test Document Auto-Crystallizer Endpoint
        doc_txt = "# Byzantine Consensus Rule\nByzantine fault tolerance requires 3f + 1 nodes to survive f faulty nodes."
        code_d, data_d, _ = make_request(
            server,
            "POST",
            "/api/crystal/crystallize_doc",
            {"title": "BFT_Doc", "text": doc_txt},
        )
        assert code_d == 200
        assert data_d["ok"] is True

        code_d_ask, data_d_ask, _ = make_request(
            server, "POST", "/api/crystal/ask", {"prompt": "Explain Byzantine Consensus Rule"}
        )
        assert "3f + 1" in data_d_ask["answer"]

        # 7. Test Micro-Crystal Teleport Endpoint
        code_tel, data_tel, _ = make_request(
            server,
            "POST",
            "/api/crystal/teleport",
            {"scrubbed_prompt": "What is Grover search?", "device_id": "test_pocket_device"},
        )
        assert code_tel == 200
        assert data_tel["status"] == "ok"
        assert data_tel["capsule"] is not None
        assert "signature_hex" in data_tel["capsule"]

    finally:
        server.shutdown()


def test_crystal_http_server_real_vault_autoload():
    vault_file = Path("artifacts/mirror_crystal_all_datasets.dcrystal")
    if not vault_file.exists():
        return
    service = BackendService(enable_crystal_auto_load=True)
    server = create_server(port=0, service=service)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        status, st_data, _ = make_request(server, "GET", "/api/crystal/status")
        assert status == 200
        assert st_data["crystal"]["rule_count"] >= 32000

        # Test Grover typo query on real 32.4K vault through HTTP
        status_q, q_data, _ = make_request(
            server, "POST", "/api/crystal/ask", {"prompt": "How does Grvoer quantm serach achieve a speedup?"}
        )
        assert status_q == 200
        assert "grover" in q_data["answer"].lower() or "sqrt" in q_data["answer"].lower()
        assert q_data["latency_ms"] < 100.0
    finally:
        server.shutdown()
        if service.crystal_vault is not None:
            service.crystal_vault.close()
