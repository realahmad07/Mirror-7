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
