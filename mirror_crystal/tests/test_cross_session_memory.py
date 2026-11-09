"""
Tests for CrossSessionMemoryGraph:
Entity persistence, relational linking, project profiles, contradiction reconciliation, and auto-extraction.
"""
from pathlib import Path
import tempfile
from mirror_crystal.cross_session_memory import CrossSessionMemoryGraph


def test_cross_session_memory_crud():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test_memory.sqlite"
        mem = CrossSessionMemoryGraph(db_path)

        # 1. Store entity
        res = mem.remember("sess_alpha", "db_port", 5432, entity_type="network_config")
        assert res["status"] == "created"
        assert res["value"] == "5432"

        # 2. Recall entity
        e = mem.recall_entity("sess_alpha", "db_port")
        assert e is not None
        assert e["value"] == "5432"
        assert e["type"] == "network_config"

        # 3. Update entity (reconciliation)
        res_update = mem.remember("sess_alpha", "db_port", 5433, entity_type="network_config")
        assert res_update["status"] == "updated"
        e_updated = mem.recall_entity("sess_alpha", "db_port")
        assert e_updated["value"] == "5433"

        # 4. List entities
        all_ents = mem.list_entities("sess_alpha")
        assert len(all_ents) == 1
        assert all_ents[0]["key"] == "db_port"

        # 5. Forget entity
        ok = mem.forget_entity("sess_alpha", "db_port")
        assert ok is True
        assert mem.recall_entity("sess_alpha", "db_port") is None

        mem.close()


def test_memory_relations_and_profiles():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test_memory.sqlite"
        mem = CrossSessionMemoryGraph(db_path)

        # Link entities
        mem.remember("sess_beta", "auth_service", "FastAPI", "service")
        mem.remember("sess_beta", "user_db", "PostgreSQL", "database")
        ok_link = mem.link_entities("sess_beta", "auth_service", "connects_to", "user_db")
        assert ok_link is True

        rels = mem.query_relations("sess_beta", "auth_service")
        assert len(rels) == 1
        assert rels[0]["rel"] == "connects_to"
        assert rels[0]["dst"] == "user_db"

        # Project profile
        prof = mem.set_project_profile(
            "sess_beta",
            project_name="QuantumMirror",
            root_dir="/workspace/mirror",
            framework="FastAPI",
            database="PostgreSQL",
            port=8787,
            constraints={"cpu_only": True},
        )
        assert prof["name"] == "QuantumMirror"
        assert prof["port"] == 8787

        retrieved = mem.get_project_profile("sess_beta")
        assert retrieved is not None
        assert retrieved["name"] == "QuantumMirror"
        assert retrieved["constraints"]["cpu_only"] is True

        mem.close()


def test_auto_extract_and_resolve_context():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test_memory.sqlite"
        mem = CrossSessionMemoryGraph(db_path)
        try:
            text = "Our project database is postgresql running on port 5432 with s3 bucket analytics-prod-logs."
            extracted = mem.auto_extract_and_remember("sess_gamma", text)
            assert len(extracted) >= 3

            ctx = mem.resolve_memory_context("sess_gamma", "What is our database port and s3 bucket?")
            assert ctx is not None
            assert "5432" in ctx
            assert "analytics-prod-logs" in ctx
        finally:
            mem.close()


def test_context_extraction_privacy_gdpr_and_compaction():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test_privacy_gdpr.sqlite"
        mem = CrossSessionMemoryGraph(
            db_path, encryption_key="test-only-memory-key-with-at-least-32-chars"
        )
        try:
            # 1. Context-Aware Port Extraction (Bottleneck 11)
            t_srv = "Our analytics server runs on port 9443 with framework FastAPI"
            ex_srv = mem.auto_extract_and_remember("sess_priv", t_srv)
            srv_port = mem.recall_entity("sess_priv", "server_port")
            assert srv_port is not None
            assert srv_port["value"] == "9443"
            assert srv_port["type"] == "service_config"

            t_db = "Postgres database listening on port 5432"
            ex_db = mem.auto_extract_and_remember("sess_priv", t_db)
            db_port = mem.recall_entity("sess_priv", "database_port")
            assert db_port is not None
            assert db_port["value"] == "5432"
            assert db_port["type"] == "config_database"

            # 2. Undo and Confirm (Bottleneck 11)
            undo_res = mem.undo_last("sess_priv")
            assert undo_res is not None
            assert undo_res["undone_key"] == "database_port"
            assert mem.recall_entity("sess_priv", "database_port") is None

            mem.confirm_entity("sess_priv", "server_port")
            confirmed = mem.recall_entity("sess_priv", "server_port")
            assert confirmed["confidence"] == 1.0

            # 3. Secret Scrubbing & Encryption at Rest (Bottleneck 12)
            raw_cred = "Deploy key is sk-1234567890abcdef1234567890 with password=SuperSecretPassword123"
            res_sec = mem.remember("sess_priv", "prod_token", raw_cred, entity_type="credential")
            # Must not contain plain secrets
            assert "SuperSecretPassword123" not in res_sec["value"]
            assert "sk-1234567890" not in res_sec["value"]

            # Underlying database must store encrypted string
            raw_row = mem._conn.execute("SELECT value_text FROM memory_entities WHERE session_id = 'sess_priv' AND entity_key = 'prod_token'").fetchone()
            assert raw_row[0].startswith("aead:v1:")
            raw_json = mem._conn.execute(
                "SELECT value_json FROM memory_entities WHERE session_id = 'sess_priv' AND entity_key = 'prod_token'"
            ).fetchone()
            assert raw_json[0] is None

            # Recall decrypts transparently
            recalled_sec = mem.recall_entity("sess_priv", "prod_token")
            assert "[REDACTED_API_KEY]" in recalled_sec["value"]

            # 4. GDPR Export & Purge (Bottleneck 12)
            user_data = mem.export_user_data("sess_priv")
            assert user_data["session_id"] == "sess_priv"
            assert len(user_data["entities"]) >= 2
            assert any(e["key"] == "server_port" for e in user_data["entities"])

            purge_stats = mem.purge_user_data("sess_priv")
            assert purge_stats["entities_purged"] >= 2
            assert mem.list_entities("sess_priv") == []

            # 5. Quota & Compaction (Bottleneck 13)
            mem.max_entities_per_session = 3
            mem.remember("sess_quota", "k1", "v1", confidence=0.5)
            mem.remember("sess_quota", "k2", "v2", confidence=0.8)
            mem.remember("sess_quota", "k3", "v3", confidence=0.9)
            assert len(mem.list_entities("sess_quota")) == 3

            # 4th entity evicts lowest confidence (k1)
            mem.remember("sess_quota", "k4", "v4", confidence=1.0)
            entities_after = mem.list_entities("sess_quota")
            assert len(entities_after) == 3
            assert not any(e["key"] == "k1" for e in entities_after)

            # Compaction purges the evicted inactive row
            compact_res = mem.compact_memory("sess_quota")
            assert compact_res["inactive_entities_purged"] >= 1
        finally:
            mem.close()


def test_sensitive_memory_requires_a_configured_key(tmp_path):
    import pytest

    mem = CrossSessionMemoryGraph(tmp_path / "no-key.sqlite")
    try:
        with pytest.raises(ValueError, match="MIRROR_MEMORY_KEY"):
            mem.remember("session", "api_token", "token-value", entity_type="credential")
    finally:
        mem.close()


def test_aead_cipher_uses_random_nonces_and_detects_tampering():
    import pytest
    from mirror_crystal.cross_session_memory import StdlibCipher

    key = "correct-horse-battery-staple-and-more"
    first = StdlibCipher.encrypt("private value", key)
    second = StdlibCipher.encrypt("private value", key)
    assert first.startswith("aead:v1:")
    assert second.startswith("aead:v1:")
    assert first != second
    assert StdlibCipher.decrypt(first, key) == "private value"

    prefix, payload = first.rsplit(":", 1)
    tampered = payload[:-2] + ("AA" if payload[-2:] != "AA" else "BB")
    with pytest.raises(Exception):
        StdlibCipher.decrypt(prefix + ":" + tampered, key)
