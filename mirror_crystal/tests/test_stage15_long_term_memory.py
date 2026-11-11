"""Stage 15 Test Suite: Long-Term Memory & Learning Subsystem.

Tests:
1. Multi-tier memory node creation and validation.
2. Ebbinghaus forgetting curve retention decay and stability scaling.
3. Consolidation promotion from working episodic to consolidated semantic memory.
4. Pruning of decayed memories with permanent protection for immutable vault knowledge.
5. Feedback correction with tombstoning, superseding pointer, and audit trail.
6. Strict Trust Invariant Guard preventing modification or deletion of immutable trusted vault.
7. SQLite persistent storage, search retrieval, and tier prioritization.
"""
from __future__ import annotations

import datetime
from pathlib import Path
import pytest

from mirror_crystal.long_term_memory import (
    ForgettingCurveScheduler,
    LongTermMemoryManager,
    MemoryConsolidationEngine,
    MemoryError,
    MemoryNode,
    MemoryProtectionError,
    MemoryTier,
)


def test_memory_node_creation_and_validation(tmp_path: Path):
    """Verify memory node field contracts and sha256 generation."""
    node = MemoryNode(
        memory_id="mem_001",
        content="The user preferred dark-mode syntax highlighting.",
        tier=MemoryTier.WORKING_EPISODIC,
        importance_score=0.85,
    )
    assert node.memory_id == "mem_001"
    assert node.tier == MemoryTier.WORKING_EPISODIC
    assert len(node.sha256_hash) == 64

    # Empty content must fail
    with pytest.raises(MemoryError, match="content must not be empty"):
        MemoryNode(memory_id="m2", content="", tier=MemoryTier.WORKING_EPISODIC)

    # Invalid importance must fail
    with pytest.raises(MemoryError, match="importance_score"):
        MemoryNode(memory_id="m3", content="valid", tier=MemoryTier.WORKING_EPISODIC, importance_score=1.5)


def test_ebbinghaus_forgetting_curve_dynamics():
    """Verify retention decays over time but stability scales with access frequency and importance."""
    now = datetime.datetime.now(datetime.timezone.utc)
    one_day_ago = (now - datetime.timedelta(days=1)).isoformat()
    now_iso = now.isoformat()

    # Brand new memory (0 elapsed time) -> retention ~ 1.0
    r0 = ForgettingCurveScheduler.compute_retention(
        last_accessed_iso=now_iso,
        half_life_seconds=86400.0,
        access_count=1,
        importance_score=0.5,
        current_time_iso=now_iso,
    )
    assert r0 >= 0.99

    # 1-day old memory with low access and low importance -> decays significantly
    r_decayed = ForgettingCurveScheduler.compute_retention(
        last_accessed_iso=one_day_ago,
        half_life_seconds=86400.0,
        access_count=1,
        importance_score=0.2,
        current_time_iso=now_iso,
    )
    assert r_decayed < 0.50

    # 1-day old memory with high access count (10) and high importance (0.9) -> retains strongly
    r_stable = ForgettingCurveScheduler.compute_retention(
        last_accessed_iso=one_day_ago,
        half_life_seconds=86400.0,
        access_count=10,
        importance_score=0.9,
        current_time_iso=now_iso,
    )
    assert r_stable > r_decayed
    assert r_stable >= 0.65


def test_memory_consolidation_and_pruning():
    """Verify promotion from episodic to semantic and pruning of severely decayed nodes."""
    engine = MemoryConsolidationEngine(
        consolidation_threshold_accesses=3,
        consolidation_threshold_importance=0.70,
        pruning_retention_threshold=0.10,
    )

    now = datetime.datetime.now(datetime.timezone.utc)
    now_iso = now.isoformat()
    three_weeks_ago = (now - datetime.timedelta(days=21)).isoformat()

    # 1. Frequently accessed episodic memory gets promoted
    node_frequent = MemoryNode(
        memory_id="m_freq",
        content="Frequent user command pattern",
        tier=MemoryTier.WORKING_EPISODIC,
        access_count=5,
        importance_score=0.5,
        last_accessed_at=now_iso,
    )
    action1, _ = engine.evaluate_node(node_frequent, current_time_iso=now_iso)
    assert action1 == "consolidate"

    # 2. High importance episodic memory gets promoted even on first turn
    node_important = MemoryNode(
        memory_id="m_imp",
        content="User core objective",
        tier=MemoryTier.WORKING_EPISODIC,
        access_count=1,
        importance_score=0.90,
        last_accessed_at=now_iso,
    )
    action2, _ = engine.evaluate_node(node_important, current_time_iso=now_iso)
    assert action2 == "consolidate"

    # 3. Old, forgotten memory gets pruned
    node_old = MemoryNode(
        memory_id="m_old",
        content="Transient ephemeral calculation note",
        tier=MemoryTier.WORKING_EPISODIC,
        access_count=1,
        importance_score=0.2,
        last_accessed_at=three_weeks_ago,
        half_life_seconds=3600.0,
    )
    action3, r3 = engine.evaluate_node(node_old, current_time_iso=now_iso)
    assert action3 == "prune"
    assert r3 < 0.10

    # 4. Immutable trusted vault node is NEVER pruned even if ancient
    node_vault = MemoryNode(
        memory_id="vault_rule_01",
        content="Fundamental physical constant c = 299792458 m/s",
        tier=MemoryTier.IMMUTABLE_TRUSTED_VAULT,
        access_count=1,
        importance_score=1.0,
        last_accessed_at=three_weeks_ago,
    )
    action4, r4 = engine.evaluate_node(node_vault, current_time_iso=now_iso)
    assert action4 == "retain"
    assert r4 == 1.0


def test_sqlite_persistence_and_consolidation_cycle(tmp_path: Path):
    """Verify LongTermMemoryManager stores to SQLite and runs consolidation cycles."""
    db_file = tmp_path / "memory_test.sqlite"
    mgr = LongTermMemoryManager(db_path=db_file)

    now = datetime.datetime.now(datetime.timezone.utc)
    now_iso = now.isoformat()
    old_iso = (now - datetime.timedelta(days=30)).isoformat()

    # Store 3 nodes
    m1 = mgr.store_memory(
        content="User prefers Python 3.12 syntax",
        tier=MemoryTier.WORKING_EPISODIC,
        importance_score=0.95,
        memory_id="m1_pref",
    )
    m2 = mgr.store_memory(
        content="Transient scratchpad result 42",
        tier=MemoryTier.WORKING_EPISODIC,
        importance_score=0.1,
        memory_id="m2_scratch",
        half_life_seconds=600.0,
    )
    # Manually update m2's timestamp to the past to test pruning
    with mgr._lock:
        import sqlite3
        with sqlite3.connect(mgr.db_path) as conn:
            conn.execute("UPDATE memories SET last_accessed_at = ? WHERE memory_id = ?", (old_iso, "m2_scratch"))
            conn.commit()

    m3 = mgr.store_memory(
        content="SI base unit definition of second",
        tier=MemoryTier.IMMUTABLE_TRUSTED_VAULT,
        importance_score=1.0,
        memory_id="m3_si",
    )

    cycle_stats = mgr.run_consolidation_cycle(current_time_iso=now_iso)
    assert cycle_stats["promoted_to_semantic"] == 1  # m1_pref has importance 0.95
    assert cycle_stats["pruned_decayed"] == 1        # m2_scratch decayed
    assert cycle_stats["retained"] == 1              # m3_si retained

    # Check updated tier in database
    retrieved_m1 = mgr.get_memory("m1_pref")
    assert retrieved_m1.tier == MemoryTier.CONSOLIDATED_SEMANTIC
    assert retrieved_m1.is_active is True

    retrieved_m2 = mgr.get_memory("m2_scratch", touch=False)
    assert retrieved_m2.is_active is False


def test_feedback_correction_and_tombstone(tmp_path: Path):
    """Verify feedback updates tombstone old memories and records correction lineage."""
    db_file = tmp_path / "memory_feedback.sqlite"
    mgr = LongTermMemoryManager(db_path=db_file)

    orig = mgr.store_memory(
        content="Project deployment target is AWS us-east-1",
        tier=MemoryTier.CONSOLIDATED_SEMANTIC,
        importance_score=0.6,
        memory_id="deploy_loc",
    )

    # Apply correction
    corrected = mgr.apply_feedback_correction(
        memory_id="deploy_loc",
        correction_content="Project deployment target is GCP us-central1",
        feedback_notes="User migrated infrastructure from AWS to GCP",
        feedback_score=-1.0,
    )

    assert corrected.memory_id != "deploy_loc"
    assert "GCP us-central1" in corrected.content
    assert corrected.metadata["supersedes"] == "deploy_loc"

    # Check original memory is tombstoned
    old = mgr.get_memory("deploy_loc", touch=False)
    assert old.is_active is False
    assert old.superseded_by == corrected.memory_id
    assert len(old.feedback_history) == 1
    assert old.feedback_history[0]["notes"] == "User migrated infrastructure from AWS to GCP"


def test_strict_trust_invariant_guard_blocks_vault_mutation(tmp_path: Path):
    """Verify attempt to modify or delete immutable trusted vault knowledge fails closed."""
    db_file = tmp_path / "memory_trust.sqlite"
    mgr = LongTermMemoryManager(db_path=db_file)

    # Register an immutable trusted vault knowledge item
    mgr.store_memory(
        content="Speed of light in vacuum c is exactly 299792458 m/s",
        tier=MemoryTier.IMMUTABLE_TRUSTED_VAULT,
        importance_score=1.0,
        memory_id="vault_physics_c",
    )

    # Attempting to correct immutable vault knowledge MUST raise MemoryProtectionError
    with pytest.raises(MemoryProtectionError, match="cryptographically protected and cannot be altered"):
        mgr.apply_feedback_correction(
            memory_id="vault_physics_c",
            correction_content="Speed of light is 300000000 m/s",
            feedback_notes="Approximating c for simplicity",
        )

    # Verify vault node remains intact and untouched
    vault_node = mgr.get_memory("vault_physics_c", touch=False)
    assert vault_node.is_active is True
    assert vault_node.superseded_by is None
    assert vault_node.content == "Speed of light in vacuum c is exactly 299792458 m/s"


def test_memory_search_and_tier_prioritization(tmp_path: Path):
    """Verify memory search ranks by lexical relevance, retention, and tier trust multiplier."""
    db_file = tmp_path / "memory_search.sqlite"
    mgr = LongTermMemoryManager(db_path=db_file)

    mgr.store_memory(
        content="Working note on Python async loops",
        tier=MemoryTier.WORKING_EPISODIC,
        importance_score=0.4,
    )
    mgr.store_memory(
        content="Verified canonical Python coroutine execution model",
        tier=MemoryTier.IMMUTABLE_TRUSTED_VAULT,
        importance_score=0.9,
    )

    results = mgr.search_memories("Python coroutine loops", limit=5)
    assert len(results) >= 2
    # The immutable vault node receives priority trust weighting
    top_node, score = results[0]
    assert top_node.tier == MemoryTier.IMMUTABLE_TRUSTED_VAULT
    assert "coroutine execution" in top_node.content
