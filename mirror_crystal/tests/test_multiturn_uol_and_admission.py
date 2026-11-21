"""Tests for Multi-Turn Stateful UOL Reasoning and Staged Blueprint Admission Integrity."""
from __future__ import annotations

import math
from pathlib import Path
import sqlite3

from mirror_crystal import DiskCrystalVault
from mirror_crystal.blueprint_contracts import EvidenceLevel
from mirror_crystal.nl_uol_router import NaturalLanguageUOLRouter
from mirror_crystal.uol_skill_store import load_uol_from_vault
from mirror7_backend.cognitive_orchestrator import UnifiedCognitiveOrchestrator

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_multiturn_stateful_uol_reasoning_and_session_isolation():
    router = NaturalLanguageUOLRouter()

    # Session A: build up a 3-step physics->engineering derivation across 4 turns
    t1 = router.route("My mass is 10 kg and acceleration is 5", session_id="sess_a")
    assert t1.matched and "Stored session quantities" in t1.explanation

    t2 = router.route("Now find force", session_id="sess_a")
    assert t2.matched and math.isclose(float(t2.output), 50.0, rel_tol=1e-6)

    t3 = router.route("If area is 2, find stress", session_id="sess_a")
    assert t3.matched and math.isclose(float(t3.output), 25.0, rel_tol=1e-6)

    t4 = router.route("With yield strength 100, what is safety factor?", session_id="sess_a")
    assert t4.matched and math.isclose(float(t4.output), 4.0, rel_tol=1e-6)

    # Session B must NOT see Session A's stored quantities
    tb = router.route("Now find force", session_id="sess_b")
    assert tb.matched is False

    # Clear session A
    tc = router.route("clear quantities", session_id="sess_a")
    assert tc.matched and router.get_session_quantities("sess_a") == {}


def test_multiturn_reasoning_through_backend_orchestrator():
    class _Mem:
        def resolve_memory_context(self, session_id, prompt):
            return None

    class _Reas:
        def solve(self, prompt):
            return {"ok": False}

    class _Code:
        def run_tdd_cycle(self, *args, **kwargs):
            raise AssertionError("not code")

    orch = UnifiedCognitiveOrchestrator(
        vault_provider=lambda: None,
        memory_provider=_Mem,
        reasoning_provider=_Reas,
        code_provider=_Code,
    )
    a1 = orch.ask("Set current is 4 and resistance is 5", session_id="chat_1")
    assert a1["task_result_verified"] is True
    a2 = orch.ask("Now calculate electrical power", session_id="chat_1")
    assert a2["task_result_verified"] is True
    assert math.isclose(float(a2["uol_route"]["output"]), 80.0, rel_tol=1e-6)


def test_persisted_vault_preserves_120_uol_skills_and_1343_unverified_blueprints():
    vault_path = REPO_ROOT / "artifacts/mirror_uol_skills.dcrystal"
    assert vault_path.is_file()

    with sqlite3.connect(vault_path) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

    _uol, report = load_uol_from_vault(vault_path)
    assert report.imported_count == 120
    assert report.conflicted_rule_ids == ()

    vault = DiskCrystalVault(vault_path)
    try:
        bps = vault.list_blueprints(limit=5000)
        assert len(bps) == 1343
        assert all(bp.verification.level is EvidenceLevel.UNVERIFIED for bp in bps)
    finally:
        vault.close()
