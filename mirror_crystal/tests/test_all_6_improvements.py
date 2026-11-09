"""
Unit tests for all 6 Advanced Architectural Improvements:
  1. Subword Character N-Gram Fingerprinting & Typo Resilience
  2. Zero-Prompt Autonomous Document Crystallizer (Directory & Markdown)
  3. Multi-Hop Relational Knowledge Chaining
  4. Memory-to-Rule Dreaming & Consolidation
  5. Server-Sent Events (SSE) Streaming
  6. Interactive Ambiguity Disambiguation
"""
import json
import tempfile
import urllib.request
from pathlib import Path
from http.server import HTTPServer

from mirror_crystal.bit_signatures import morph_byte_signature, ContextBitTable
from mirror_crystal.disk_vault import DiskCrystalVault
from mirror_crystal.cross_session_memory import CrossSessionMemoryGraph
from mirror_crystal.phase5_bicameral_and_cas import DocumentAutoCrystallizer
from mirror7_backend.service import BackendService
from mirror7_backend.http_server import MirrorAPIHandler


def test_improvement_1_subword_ngram():
    sig_orig = morph_byte_signature("relativity")
    sig_typo = morph_byte_signature("relativty")  # 1-char deletion
    shared = (sig_orig & sig_typo).bit_count()
    assert shared >= 15, f"Subword overlap too low: {shared}/32"

    table = ContextBitTable()
    table.observe_text("Photosynthesis transforms solar energy into chemical energy.")
    resolved = table.resolve_typo_token("photosynthesiss")
    assert resolved == "photosynthesis"


def test_improvement_2_document_directory_crystallizer():
    with tempfile.TemporaryDirectory() as td:
        docs_dir = Path(td) / "docs"
        docs_dir.mkdir()
        (docs_dir / "physics.md").write_text(
            "# Quantum Entanglement\nQuantum entanglement is a phenomenon where quantum states are linked.\n\n"
            "Qubit: The fundamental unit of quantum computational memory.\n",
            encoding="utf-8"
        )
        vault_path = Path(td) / "vault.dcrystal"
        vault = DiskCrystalVault(vault_path)
        try:
            stats = DocumentAutoCrystallizer.ingest_directory(vault, docs_dir, domain="quantum")
            assert stats["files_processed"] == 1
            assert stats["sections_crystallized"] >= 1
            assert stats["definitions_crystallized"] >= 1
            assert vault.rule_count() >= 2
        finally:
            vault.close()


def test_improvement_3_multi_hop_relational():
    with tempfile.TemporaryDirectory() as td:
        vault = DiskCrystalVault(Path(td) / "vault.dcrystal")
        try:
            vault.ingest_pair("What is the capital of France?", "The capital of France is Paris.", domain="geography")
            vault.ingest_pair("What is the population of Paris?", "The population of Paris is 2.1 million.", domain="geography")

            # Possessive multi-hop
            resp1 = vault.respond("What is the population of France's capital?")
            assert resp1.mode == "multi_hop_relational_chain"
            assert "Paris" in resp1.answer
            assert "2.1 million" in resp1.answer
            assert resp1.confidence >= 0.90

            # Of-the multi-hop
            resp2 = vault.respond("What is the population of the capital of France?")
            assert resp2.mode == "multi_hop_relational_chain"
            assert "Paris" in resp2.answer
            assert "2.1 million" in resp2.answer
        finally:
            vault.close()


def test_improvement_4_memory_dreaming():
    with tempfile.TemporaryDirectory() as td:
        v_path = Path(td) / "vault.dcrystal"
        mem_path = Path(td) / "mem.sqlite"
        vault = DiskCrystalVault(v_path)
        mem = CrossSessionMemoryGraph(db_path=mem_path)
        try:
            mem.remember("proj_alpha", "docker_registry", "ghcr.io/mirror7", confidence=0.95)
            mem.remember("proj_alpha", "db_port", "5432", confidence=0.92)

            stats = mem.consolidate_to_vault(vault, min_confidence=0.85)
            assert stats["entities_scanned"] == 2
            assert stats["rules_crystallized"] == 2
            assert vault.rule_count() == 2

            # Query the consolidated rule
            resp = vault.respond("What is the docker registry of proj alpha?")
            assert "ghcr.io/mirror7" in resp.answer
        finally:
            mem.close()
            vault.close()


def test_improvement_5_sse_streaming():
    with tempfile.TemporaryDirectory() as td:
        v_path = Path(td) / "vault.dcrystal"
        vault = DiskCrystalVault(v_path)
        vault.ingest_pair("What is quantum computing?", "Quantum computing uses superposition.", domain="quantum")

        svc = BackendService(crystal_vault=vault, enable_crystal_auto_load=False)
        handler = type("TestAPIHandler", (MirrorAPIHandler,), {"service": svc})

        # Spin up local test HTTP server on ephemeral port
        server = HTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        import threading
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            url = f"http://127.0.0.1:{port}/api/crystal/stream?q=What+is+quantum+computing%3F"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                assert resp.status == 200
                content_type = resp.headers.get("Content-Type", "")
                assert "text/event-stream" in content_type

                lines = []
                while True:
                    raw_line = resp.readline().decode("utf-8")
                    if not raw_line:
                        break
                    lines.append(raw_line.strip())
                    if '"done": true' in raw_line.lower():
                        break
                chunks = []
                final_event = None
                for line in lines:
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data.get("done"):
                            final_event = data
                        elif "chunk" in data:
                            chunks.append(data["chunk"])

                assembled = "".join(chunks)
                assert "UNVERIFIED CANDIDATE" in assembled
                assert "Quantum computing uses superposition." in assembled
                assert final_event is not None
                assert final_event["done"] is True
                assert final_event["confidence"] == 0.0
                assert final_event["verified"] is False
                assert final_event["abstained"] is True
                assert final_event["answer_qualification"] == "unverified_candidate"
        finally:
            server.shutdown()
            vault.close()


def test_improvement_6_interactive_disambiguation():
    with tempfile.TemporaryDirectory() as td:
        vault = DiskCrystalVault(Path(td) / "vault.dcrystal")
        try:
            vault.ingest_pair("Explain python concept", "Python is a high-level programming language.", domain="programming")
            vault.ingest_pair("Explain python animal", "Python is a large nonvenomous constrictor snake.", domain="zoology")

            resp = vault.respond("Explain python")
            assert resp.mode == "clarification_needed"
            assert resp.epistemic_tier == "diagnostic_clarification"
        finally:
            vault.close()
