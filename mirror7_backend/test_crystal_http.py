from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import urllib.error
import urllib.request

os.environ.setdefault("MIRROR_SANDBOX_MODE", "process")

import pytest

from mirror7_backend.http_server import create_server
from mirror7_backend.service import BackendService
from mirror_crystal import MirrorCrystalEngine


def make_request(
    server,
    method: str,
    path: str,
    payload: dict | None = None,
    is_bytes: bool = False,
    timeout_seconds: float = 5.0,
):
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
    with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
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
        # Retrieval returns the stored answer; it does not independently prove the fact.
        assert data_q["epistemic_tier"] == "honest_abstain"
        assert data_q["verified"] is False
        assert data_q["abstained"] is True
        assert data_q["answer_qualification"] == "unverified_candidate"

        code_cas, data_cas, _ = make_request(server, "POST", "/api/crystal/ask", {"prompt": "Differentiate x^3 + 5*x"})
        assert code_cas == 200
        assert "3*x^2 + 5" in data_cas["answer"]

        # Compositional quantum math routing seamlessly through /api/crystal/ask
        code_qm, data_qm, _ = make_request(
            server,
            "POST",
            "/api/crystal/ask",
            {"prompt": "How many Grover iterations are needed for N = 10000 items?"},
        )
        assert code_qm == 200
        assert "78" in data_qm["answer"]
        assert data_qm["epistemic_tier"] == "honest_abstain"
        assert data_qm["verified"] is False
        assert data_qm["abstained"] is True

        # Genuine out-of-distribution prose must still abstain honestly
        code_ood, data_ood, _ = make_request(
            server, "POST", "/api/crystal/ask", {"prompt": "Explain why Hamlet hesitated in Act III."}
        )
        assert code_ood == 200
        assert data_ood["epistemic_tier"] == "honest_abstain"

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

        # 8. Test Dual-Brain Cerebrum-Crystal Bridge Endpoint (Qwen-Math parity)
        code_cer, data_cer, _ = make_request(
            server,
            "POST",
            "/api/cerebrum/solve",
            {"prompt": "differentiate x^4 + 3*x^2 wrt x"},
        )
        assert code_cer == 200
        assert data_cer["ok"] is True
        assert "4*x^3 + 6*x" in data_cer["answer"]

        # 9. Test Autonomous TDD Synthesizer Endpoint (Opus Coder parity)
        code_tdd, data_tdd, _ = make_request(
            server,
            "POST",
            "/api/tdd/synthesize",
            {
                "code": "def fib(n):\n    return fib(n - 1) + fib(n - 2)",
                "spec": "compute nth fibonacci number",
            },
            timeout_seconds=30.0,
        )
        assert code_tdd == 200
        assert data_tdd["verified"] is True
        assert "if n <= 0" in data_tdd["final_code"]

        # 10. Test Six-Gate Data Hygiene Endpoint ("Quality Over Quantity")
        sample_txt = "# Theorem on Prime Sieve\nO(N log log N) complexity.\n```python\ndef sieve(n): return [2]\n```\n"
        code_gate, data_gate, _ = make_request(
            server,
            "POST",
            "/api/data/six_gate_audit",
            {"text": sample_txt, "spdx": "MIT"},
        )
        assert code_gate == 200
        assert data_gate["passed"] is True
        assert data_gate["metrics"]["gate1_spdx_pii"] == "PASS"

    finally:
        server.shutdown()


def test_crystal_http_hygiene_and_safety_guards():
    eng = MirrorCrystalEngine()
    service = BackendService(crystal_vault=eng, enable_crystal_auto_load=False)
    server = create_server(port=0, service=service)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        # 1. Reject PII leak in /api/crystal/teach with HTTP 400
        with pytest.raises(urllib.error.HTTPError) as exc_pii:
            make_request(
                server,
                "POST",
                "/api/crystal/teach",
                {"question": "Who is admin?", "answer": "Reach admin@secret-internal.org immediately."},
            )
        assert exc_pii.value.code == 400

        # 2. Reject unsafe Python code in /api/crystal/teach with HTTP 400
        with pytest.raises(urllib.error.HTTPError) as exc_unsafe:
            make_request(
                server,
                "POST",
                "/api/crystal/teach",
                {"question": "Run shell?", "answer": "```python\nimport os\ndef boom(): return os.getcwd()\n```"},
            )
        assert exc_unsafe.value.code == 400

        # 3. Reject benchmark contamination in /api/crystal/crystallize_doc with HTTP 400
        with pytest.raises(urllib.error.HTTPError) as exc_doc:
            make_request(
                server,
                "POST",
                "/api/crystal/crystallize_doc",
                {"title": "Contam", "text": "def has_close_elements(numbers, threshold): return False"},
            )
        assert exc_doc.value.code == 400
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
        if service.crystal_vault is not None and hasattr(service.crystal_vault, 'close'):
            service.crystal_vault.close()


def test_streamer_and_memory_http_endpoints(tmp_path: Path):
    from mirror_crystal.cross_session_memory import CrossSessionMemoryGraph
    eng = MirrorCrystalEngine()
    db_path = tmp_path / "test_mem.sqlite"
    service = BackendService(crystal_vault=eng, enable_crystal_auto_load=False)
    service._memory_graph = CrossSessionMemoryGraph(db_path=db_path)
    server = create_server(port=0, service=service)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        # 1. Streamer Status
        code, st, _ = make_request(server, "GET", "/api/streamer/status")
        assert code == 200
        assert "is_running" in st

        # 2. Streamer Pulse
        code_p, p_data, _ = make_request(server, "POST", "/api/streamer/pulse", {})
        assert code_p == 200
        assert "admitted" in p_data

        # 3. Memory Profile POST & GET
        prof_req = {
            "session_id": "sess_test",
            "name": "ECommerceEngine",
            "framework": "FastAPI",
            "database": "PostgreSQL",
            "port": 8080,
        }
        code_prof, prof_res, _ = make_request(server, "POST", "/api/memory/profile", prof_req)
        assert code_prof == 200
        assert prof_res["ok"] is True
        assert prof_res["profile"]["name"] == "ECommerceEngine"

        code_gprof, gprof_res, _ = make_request(server, "GET", "/api/memory/profile?session_id=sess_test")
        assert code_gprof == 200
        assert gprof_res["profile"]["framework"] == "FastAPI"

        # 4. Memory Remember POST & Entities GET
        rem_req = {
            "session_id": "sess_test",
            "key": "deploy_target",
            "value": "us-east-1-cluster",
            "entity_type": "infrastructure",
        }
        code_rem, rem_res, _ = make_request(server, "POST", "/api/memory/remember", rem_req)
        assert code_rem == 200
        assert rem_res["ok"] is True

        code_ent, ent_res, _ = make_request(server, "GET", "/api/memory/entities?session_id=sess_test")
        assert code_ent == 200
        assert any(e["key"] == "deploy_target" for e in ent_res["entities"])

        # 5. Memory Forget POST
        f_req = {"session_id": "sess_test", "key": "deploy_target"}
        code_f, f_res, _ = make_request(server, "POST", "/api/memory/forget", f_req)
        assert code_f == 200
        assert f_res["deleted"] is True

        # 6. Ask with Auto-Extraction
        ask_req = {
            "session_id": "sess_test",
            "prompt": "Our analytics server runs on port 9443 with s3 bucket telemetry-logs",
        }
        code_ask, ask_res, _ = make_request(server, "POST", "/api/crystal/ask", ask_req)
        assert code_ask == 200

        # Check that port 9443 was auto-extracted and stored
        code_ent2, ent_res2, _ = make_request(server, "GET", "/api/memory/entities?session_id=sess_test")
        assert any(e["key"] in ("server_port", "database_port") and str(e["value"]) == "9443" for e in ent_res2["entities"]) and any(e["key"] == "s3_bucket" and e["value"] == "telemetry-logs" for e in ent_res2["entities"])
    finally:
        server.shutdown()
        if service._memory_graph is not None:
            service._memory_graph.close()
        if service.crystal_vault is not None and hasattr(service.crystal_vault, 'close'):
            service.crystal_vault.close()

def test_bottlenecks_9_to_13_http_endpoints(tmp_path: Path):
    from mirror_crystal.cross_session_memory import CrossSessionMemoryGraph
    eng = MirrorCrystalEngine()
    db_path = tmp_path / "test_b9_13.sqlite"
    service = BackendService(crystal_vault=eng, enable_crystal_auto_load=False)
    service._memory_graph = CrossSessionMemoryGraph(db_path=db_path)
    server = create_server(port=0, service=service)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        # 1. Coverage Map (Bottleneck 9)
        code_cov, cov_data, _ = make_request(server, "GET", "/api/crystal/coverage")
        assert code_cov == 200
        assert cov_data["ok"] is True
        assert "active_domains" in cov_data
        assert "telemetry" in cov_data
        assert "quantum_computing" in cov_data["active_domains"]
        assert "bioinformatics" in cov_data["active_domains"]

        # 2. Abstention and Teach-Me Template (Bottleneck 9)
        unknown_q = "What is the thermodynamic entropy of a Kerr-Newman singularity in d dimensions?"
        code_ask, ask_data, _ = make_request(server, "POST", "/api/crystal/ask", {"prompt": unknown_q, "consent_log": True})
        assert code_ask == 200
        assert ask_data["abstained"] is True
        assert ask_data["can_teach"] is True
        assert ask_data["teach_template"]["question"] == unknown_q

        # 3. Context-Aware Extraction & Secret Scrubbing (Bottleneck 11 & 12)
        secret_prompt = "App server on port 7070 using secret api key sk-live-999999999988888"
        code_sec, sec_data, _ = make_request(server, "POST", "/api/crystal/ask", {"prompt": secret_prompt, "session_id": "b11_sess"})
        assert code_sec == 200

        code_ents, ents_res, _ = make_request(server, "GET", "/api/memory/entities?session_id=b11_sess")
        assert code_ents == 200
        ents = ents_res["entities"]
        # Verify secret was scrubbed
        assert not any("sk-live" in str(e["value"]) for e in ents)
        # Verify context-aware port is server_port
        server_port_ent = next((e for e in ents if e["key"] == "server_port"), None)
        assert server_port_ent is not None
        assert str(server_port_ent["value"]) == "7070"

        # 4. Undo and Confirm (Bottleneck 11)
        code_undo, undo_data, _ = make_request(server, "POST", "/api/memory/undo", {"session_id": "b11_sess"})
        assert code_undo == 200
        assert undo_data["ok"] is True

        code_conf, conf_data, _ = make_request(server, "POST", "/api/memory/confirm", {"session_id": "b11_sess", "key": "server_port"})
        assert code_conf == 200

        # 5. GDPR Export & Purge (Bottleneck 12)
        code_exp, exp_data, _ = make_request(server, "GET", "/api/memory/gdpr/export?session_id=b11_sess")
        assert code_exp == 200
        assert exp_data["ok"] is True
        assert "entities" in exp_data["export"]

        code_purge, purge_data, _ = make_request(server, "POST", "/api/memory/gdpr/purge", {"session_id": "b11_sess"})
        assert code_purge == 200
        assert purge_data["ok"] is True
        assert purge_data["purged"]["entities_purged"] >= 0

        # 6. Compaction and Quota (Bottleneck 13)
        code_comp, comp_data, _ = make_request(server, "POST", "/api/memory/compact", {"session_id": "b11_sess"})
        assert code_comp == 200
        assert comp_data["ok"] is True
        assert "inactive_entities_purged" in comp_data["compacted"]

        # 7. Telemetry updated
        code_cov2, cov_data2, _ = make_request(server, "GET", "/api/crystal/coverage")
        assert code_cov2 == 200
        assert cov_data2["telemetry"]["queries_abstained"] >= 1
    finally:
        server.shutdown()
        if service._memory_graph is not None:
            service._memory_graph.close()
        if service.crystal_vault is not None and hasattr(service.crystal_vault, "close"):
            service.crystal_vault.close()


def test_market_launch_concept_lattice_http_endpoints(tmp_path: Path):
    from mirror_crystal.cross_session_memory import CrossSessionMemoryGraph
    eng = MirrorCrystalEngine()
    db_path = tmp_path / "test_lattice_http.sqlite"
    service = BackendService(crystal_vault=eng, enable_crystal_auto_load=False)
    service._memory_graph = CrossSessionMemoryGraph(db_path=db_path)
    server = create_server(port=0, service=service)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        # 1. GET /api/lattice/status
        code_st, st_data, _ = make_request(server, "GET", "/api/lattice/status")
        assert code_st == 200
        assert st_data["ok"] is True
        assert st_data["uses_neural_parameters"] is False
        assert st_data["uses_third_party_llm"] is False
        assert st_data["concept_nodes"] >= 1000
        assert st_data["domains_count"] >= 20

        # 2. GET /api/lattice/domain?name=physics&limit=5
        code_dom, dom_data, _ = make_request(server, "GET", "/api/lattice/domain?name=physics&limit=5")
        assert code_dom == 200
        assert dom_data["ok"] is True
        assert dom_data["domain"] == "physics"
        assert len(dom_data["concepts"]) == 5

        # 3. POST /api/lattice/query (Cross-Concept Head-to-Head Comparison)
        code_q, q_data, _ = make_request(
            server,
            "POST",
            "/api/lattice/query",
            {"prompt": "Compare Docker and Kubernetes"},
        )
        assert code_q == 200
        assert q_data["ok"] is True
        assert q_data["matched"] is True
        assert "Cross-Concept Comparison" in q_data["response"]

        # 4. POST /api/lattice/compile — No-Overwrite Guard blocks overwriting protected seed node 'blockchain' (409)
        try:
            code_blk, blk_data, _ = make_request(
                server,
                "POST",
                "/api/lattice/compile",
                {
                    "topic": "blockchain",
                    "text": "Weak attempt to overwrite protected blockchain concept with noise.",
                    "persist": False,
                },
            )
        except urllib.error.HTTPError as exc:
            code_blk = exc.code
            blk_data = json.loads(exc.read().decode("utf-8"))
        assert code_blk == 409
        assert blk_data["ok"] is False
        assert blk_data["protected"] is True

        # 5. POST /api/lattice/compile — Compiling a brand-new ephemeral concept succeeds (200)
        code_new, new_data, _ = make_request(
            server,
            "POST",
            "/api/lattice/compile",
            {
                "topic": "MirrorLatticeProbeConcept",
                "text": (
                    "MirrorLatticeProbeConcept is a test concept verifying live zero-bloat compilation. "
                    "It works through deterministic 5-facet extraction without neural weights."
                ),
                "category": "computer_science",
                "persist": False,
            },
        )
        assert code_new == 200
        assert new_data["ok"] is True
        assert new_data["concept_id"] == "mirrorlatticeprobeconcept"
    finally:
        server.shutdown()
        if service._memory_graph is not None:
            service._memory_graph.close()
        if service.crystal_vault is not None and hasattr(service.crystal_vault, "close"):
            service.crystal_vault.close()


def test_multi_shard_lattice_status_and_coverage_http_endpoints(tmp_path: Path):
    from mirror_crystal.cross_session_memory import CrossSessionMemoryGraph
    from mirror_crystal.open_ended_concept_lattice import ConceptNode, OpenEndedConceptLatticeEngine

    primary_path = tmp_path / "primary_http.dcrystal"
    shard_dir = tmp_path / "concept_shards"
    lat = OpenEndedConceptLatticeEngine(vault_path=primary_path, shard_dir=shard_dir)
    lat.register_nodes_bulk(
        [
            ConceptNode(
                concept_id="raft consensus",
                title="Raft Consensus Algorithm",
                category="computer_science",
                aliases=("raft consensus",),
                definition="Raft is an understandable distributed consensus algorithm that manages a replicated log across server clusters.",
                mechanism="Elects one leader per term via randomized timeouts and replicates log entries over AppendEntries quorums.",
                applications="Strongly consistent metadata services such as etcd, Consul, and CockroachDB.",
                tradeoffs="Requires majority quorum (2f+1 nodes) and incurs wide-area consensus round-trip latency.",
                provenance="mirror:domain_agent_harvester:v1",
            )
        ],
        persist=True,
    )
    lat.admit_to_shard(
        shard_dir / "shard_github_engineering.dcrystal",
        [
            ConceptNode(
                concept_id="cilium ebpf",
                title="Cilium eBPF Networking",
                category="cybersecurity_networking",
                aliases=("cilium ebpf", "ebpf service mesh"),
                definition="Cilium is a cloud-native container networking and security platform powered by in-kernel Linux eBPF bytecode programs.",
                mechanism="Attaches verified eBPF programs directly to Linux socket and XDP hooks to enforce L3/L4/L7 policies without iptables chains.",
                applications="High-throughput Kubernetes service meshes, transparent encryption, and Hubble runtime network observability.",
                tradeoffs="Requires modern Linux kernels (5.x+) and introduces kernel-level debugging complexity for custom eBPF filters.",
                provenance="cncf:glossary:cilium",
            )
        ],
        min_quality=0.78,
        vacuum=True,
    )
    lat.admit_to_shard(
        shard_dir / "shard_public_datasets.dcrystal",
        [
            ConceptNode(
                concept_id="alphafold structure prediction",
                title="Protein Structure Prediction",
                category="biology_genomics",
                aliases=("alphafold structure prediction", "protein folding prediction"),
                definition="Computational protein structure prediction infers three-dimensional atomic coordinates directly from amino-acid sequences.",
                mechanism="Integrates multiple sequence alignments, co-evolutionary distance constraints, and equivariant geometric refinement steps.",
                applications="Structural biology discovery, drug target pocket analysis, and enzyme engineering workflows.",
                tradeoffs="Static single-conformation predictions can miss intrinsically disordered regions and dynamic ligand-induced shifts.",
                provenance="openalex:T40412",
            )
        ],
        min_quality=0.78,
        vacuum=True,
    )

    eng = MirrorCrystalEngine()
    db_path = tmp_path / "test_multishard_http.sqlite"
    service = BackendService(crystal_vault=eng, enable_crystal_auto_load=False)
    service._memory_graph = CrossSessionMemoryGraph(db_path=db_path)
    service._concept_lattice = lat
    service._concept_lattice_sig = "__custom__"
    server = create_server(port=0, service=service)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        # 1. GET /api/lattice/status exposes multi-shard counts and per-shard breakdown
        code_st, st_data, _ = make_request(server, "GET", "/api/lattice/status")
        assert code_st == 200
        assert st_data["ok"] is True
        assert st_data["shards_count"] == 2
        assert st_data["primary_node_count"] == 11
        assert st_data["shards_node_count"] == 2
        assert st_data["concept_nodes"] == 13
        assert st_data["storage_backend"] == "sqlite_fts5_zlib_multi_shard"
        assert len(st_data["shards_breakdown"]) == 2
        shard_names = {s["filename"] for s in st_data["shards_breakdown"]}
        assert shard_names == {"shard_github_engineering.dcrystal", "shard_public_datasets.dcrystal"}
        for s in st_data["shards_breakdown"]:
            assert s["node_count"] == 1
            assert isinstance(s["disk_mb"], (int, float))

        # 2. GET /api/lattice/shards returns dedicated shard telemetry
        code_sh, sh_data, _ = make_request(server, "GET", "/api/lattice/shards")
        assert code_sh == 200
        assert sh_data["ok"] is True
        assert sh_data["shards_count"] == 2
        assert len(sh_data["shards_breakdown"]) == 2

        # 3. GET /api/crystal/coverage exposes multi-shard telemetry inside open_ended_concept_lattice
        code_cov, cov_data, _ = make_request(server, "GET", "/api/crystal/coverage")
        assert code_cov == 200
        lat_cov = cov_data["open_ended_concept_lattice"]
        assert lat_cov["shards_count"] == 2
        assert lat_cov["concept_nodes"] == 13
        assert len(lat_cov["shards_breakdown"]) == 2

        # 4. POST /api/lattice/query seamlessly queries shard concepts & cross-shard comparisons
        code_q1, q1, _ = make_request(server, "POST", "/api/lattice/query", {"prompt": "Explain Cilium eBPF"})
        assert code_q1 == 200
        assert q1["matched"] is True
        assert "XDP" in q1["response"]

        code_q2, q2, _ = make_request(
            server,
            "POST",
            "/api/lattice/query",
            {"prompt": "Compare Raft consensus and Cilium eBPF"},
        )
        assert code_q2 == 200
        assert q2["matched"] is True
        assert "Cross-Concept Comparison" in q2["response"]

        # 5. GET / serves Studio HTML with Multi-Shard Topology modal support
        code_ui, ui_bytes, _ = make_request(server, "GET", "/", is_bytes=True)
        assert code_ui == 200
        assert b"Multi-Shard Disk-Backed SQLite FTS5 Topology" in ui_bytes
    finally:
        server.shutdown()
        lat.close()
        if service._memory_graph is not None:
            service._memory_graph.close()
        if service.crystal_vault is not None and hasattr(service.crystal_vault, "close"):
            service.crystal_vault.close()
