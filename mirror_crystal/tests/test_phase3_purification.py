"""Test suite for Phase 3 Knowledge Purification & Non-Destructive Quarantine."""
from pathlib import Path
import sqlite3
import pytest

from mirror_crystal.disk_vault import DiskCrystalVault

VAULT_PATH = Path("restorevalutecrytaled/mirror_crystal_all_datasets.dcrystal")
BACKUP_PATH = Path("restorevalutecrytaled/mirror_crystal_all_datasets.dcrystal.phase3.bak")


@pytest.fixture(scope="module")
def vault_conn():
    if not VAULT_PATH.exists():
        pytest.skip("Production vault not present on disk")
    conn = sqlite3.connect(VAULT_PATH)
    yield conn
    conn.close()


def test_phase3_preservation_rule_one_zero_records_deleted(vault_conn):
    """Rule 1: Never delete original vault records."""
    cur = vault_conn.cursor()
    cur.execute("SELECT COUNT(*) FROM rules;")
    total_rules = cur.fetchone()[0]
    assert total_rules == 368270, f"Original rules altered! Expected 368,270, found {total_rules}"


def test_phase3_backup_file_integrity():
    """Verify that the pre-purification backup file exists and matches original size."""
    if not BACKUP_PATH.exists():
        pytest.skip("Phase 3 backup file not found")
    assert BACKUP_PATH.stat().st_size == 422084608


def test_phase3_quarantine_and_production_index_partition(vault_conn):
    """Rule 4: Count overlapping defects by unique rule ID before quarantine."""
    cur = vault_conn.cursor()
    cur.execute("SELECT COUNT(*) FROM rule_hygiene_quarantine;")
    q_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM verified_production_index;")
    prod_count = cur.fetchone()[0]

    assert q_count == 65351, f"Expected 65,351 quarantined rules, found {q_count}"
    assert prod_count == 302919, f"Expected 302,919 verified clean candidate rules, found {prod_count}"
    assert q_count + prod_count == 368270, "Partition does not exactly sum to 368,270 rules!"

    # Ensure disjoint sets
    cur.execute("""
        SELECT COUNT(*) FROM rule_hygiene_quarantine q
        JOIN verified_production_index p ON q.rule_id = p.rule_id;
    """)
    overlap = cur.fetchone()[0]
    assert overlap == 0, f"Found {overlap} rule IDs in both quarantine and verified index!"


def test_phase3_trust_tier_correction_rule_two(vault_conn):
    """Rule 2: Never automatically trust existing Tier 1 labels; set to unverified (Tier 3)."""
    cur = vault_conn.cursor()
    cur.execute("SELECT DISTINCT corrected_trust_tier, evidence_status FROM verified_production_index;")
    rows = cur.fetchall()
    assert rows == [(3, "unverified")], f"Expected only corrected Tier 3 unverified rules, got {rows}"


def test_phase3_vault_quarantine_deactivation():
    """Verify DiskCrystalVault dynamically deactivates quarantined rules and corrects trust tier."""
    if not VAULT_PATH.exists():
        pytest.skip("Vault not found")

    vault = DiskCrystalVault(VAULT_PATH)
    assert vault.is_quarantined(11181) is True # Collision rule: 'Yes please.'

    # Quarantined rule must be inactive
    rule_q = vault.get_rule(11181)
    assert rule_q is not None
    assert rule_q.active is False

    # Clean verified rule must be active and corrected to Tier 3
    # Rule 1 is canary Protocol 7777 (not in quarantine)
    assert vault.is_quarantined(1) is False
    rule_clean = vault.get_rule(1)
    assert rule_clean is not None
    assert rule_clean.active is True
    assert rule_clean.trust_tier == 3
