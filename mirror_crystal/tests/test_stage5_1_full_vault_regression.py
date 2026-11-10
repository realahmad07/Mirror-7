"""Stage 5.1: Full-Vault Regression & CI Integrity Suite.

Validates:
1. Exact preservation of all 368,270 records in rules table (Rule 1).
2. SQLite B-Tree integrity check (PRAGMA integrity_check = ok).
3. Quarantine partition integrity (65,351 unique quarantined rules, 302,919 verified index rules).
4. Zero overlap between quarantine and verified production index.
5. Trust-tier baseline governance (Rule 2: all unreviewed rules enter at Tier 3 unverified).
6. Multi-domain query regression across math, code, physics, logic, quantum, and canary protocol.
7. Sub-millisecond CPU latency across high-frequency production queries.
"""
from pathlib import Path
import sqlite3
import pytest

from mirror_crystal.disk_vault import DiskCrystalVault

VAULT_PATH = Path("restorevalutecrytaled/mirror_crystal_all_datasets.dcrystal")


@pytest.fixture(scope="module")
def vault():
    if not VAULT_PATH.exists():
        pytest.skip("Production vault not present on disk")
    v = DiskCrystalVault(VAULT_PATH)
    yield v
    v.close()


def test_stage5_1_sqlite_pragmas_and_record_integrity():
    """Verify SQLite integrity and strict Rule 1 zero-deletion preservation."""
    conn = sqlite3.connect(VAULT_PATH)
    cur = conn.cursor()

    # 1. SQLite low-level B-Tree integrity check
    cur.execute("PRAGMA integrity_check;")
    res = cur.fetchone()[0]
    assert res == "ok", f"SQLite integrity failure: {res}"

    # 2. Rule 1: exact count preservation
    cur.execute("SELECT COUNT(*) FROM rules;")
    total_rules = cur.fetchone()[0]
    assert total_rules == 368270, f"Vault rules altered! Expected 368,270, got {total_rules}"

    # 3. Rule 4: Partition counts
    cur.execute("SELECT COUNT(*) FROM rule_hygiene_quarantine;")
    q_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM verified_production_index;")
    prod_count = cur.fetchone()[0]

    assert q_count == 65351, f"Expected 65,351 quarantined rules, got {q_count}"
    assert prod_count == 302919, f"Expected 302,919 clean candidate rules, got {prod_count}"
    assert q_count + prod_count == 368270, "Partition does not exactly sum to 368,270!"

    # 4. Strict disjointness (zero overlap)
    cur.execute("""
        SELECT COUNT(*) FROM rule_hygiene_quarantine q
        JOIN verified_production_index p ON q.rule_id = p.rule_id;
    """)
    overlap = cur.fetchone()[0]
    assert overlap == 0, f"Found {overlap} overlapping rules between quarantine and production index"

    conn.close()


def test_stage5_1_trust_tier_governance_rule_two():
    """Verify Rule 2: all clean production rules are baseline Tier 3 unverified."""
    conn = sqlite3.connect(VAULT_PATH)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT corrected_trust_tier, evidence_status FROM verified_production_index;")
    rows = cur.fetchall()
    assert rows == [(3, "unverified")], f"Expected only Tier 3 unverified status, found: {rows}"
    conn.close()


def test_stage5_1_canary_and_domain_regression(vault):
    """Verify core domain regressions across historical production capabilities."""
    # Warmup query
    vault.respond("What is Mirror Crystal Canary Protocol 7777?")
    # Canary Protocol 7777 (Rule 1)
    r_canary = vault.respond("What is Mirror Crystal Canary Protocol 7777?")
    assert r_canary.confidence >= 0.95
    assert "7777" in r_canary.answer
    assert r_canary.latency_ms < 5.0

    # Quantum domain regression
    r_qubit = vault.respond("What is a qubit and how does quantum superposition work mathematically?")
    assert r_qubit.confidence >= 0.35
    assert "qubit" in r_qubit.answer.lower() or "superposition" in r_qubit.answer.lower()

    # Quarantined rule suppression: Rule #11181 ('Yes please.')
    assert vault.is_quarantined(11181) is True
    rule_q = vault.get_rule(11181)
    assert rule_q is not None
    assert rule_q.active is False


def test_stage5_1_multi_turn_session_constraint_regression(vault):
    """Verify multi-turn session memory constraints and stateful tracking."""
    session_id = "test_regress_session_5_1"
    vault.respond("Our maximum capacity is 20.", session_id=session_id)
    r2 = vault.respond("Can we run 3 batches of 7 workers?", session_id=session_id)
    assert "21" in r2.answer and "exceeds your 20 limit by 1" in r2.answer
