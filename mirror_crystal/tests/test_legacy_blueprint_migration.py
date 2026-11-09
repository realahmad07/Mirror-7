from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

from mirror_crystal import DiskCrystalVault


REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATION_SCRIPT = REPO_ROOT / "scripts" / "migrate_legacy_blueprints.py"


def _make_legacy_vault(path: Path) -> bytes:
    vault = DiskCrystalVault(path)
    try:
        assert vault.ingest_pair(
            "What does the migration fixture mean?",
            "This is a legacy answer that must remain unverified after migration.",
            domain="migration-test",
            trust_tier=1,
        )
        vault.optimize_disk()
    finally:
        vault.close()
    return path.read_bytes()


def _run_cli(*args: str):
    env = os.environ.copy()
    env["MIRROR_SANDBOX_MODE"] = "process"
    return subprocess.run(
        [sys.executable, str(MIGRATION_SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_migration_dry_run_is_read_only(tmp_path):
    vault_path = tmp_path / "legacy.dcrystal"
    before = _make_legacy_vault(vault_path)
    result = _run_cli("--vault", str(vault_path))

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["mode"] == "dry_run"
    assert report["vault_opened_read_only"] is True
    assert report["legacy_rules"] == 1
    assert vault_path.read_bytes() == before
    assert not Path(str(vault_path) + ".pre-blueprint-migration.bak").exists()


def test_apply_migration_creates_backup_and_preserves_legacy_rules(tmp_path):
    vault_path = tmp_path / "legacy.dcrystal"
    _make_legacy_vault(vault_path)
    backup_path = Path(str(vault_path) + ".pre-blueprint-migration.bak")

    result = _run_cli("--vault", str(vault_path), "--apply", "--page-size", "1")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)

    assert report["mode"] == "applied"
    assert report["legacy_rules_preserved"] is True
    assert report["legacy_rule_count_before"] == 1
    assert report["legacy_rule_count_after"] == 1
    assert report["sqlite_integrity_check"] == "ok"
    assert report["canonical_blueprints_added"] == 1
    assert backup_path.is_file()

    with sqlite3.connect(backup_path) as backup:
        assert backup.execute("SELECT COUNT(*) FROM rules").fetchone()[0] == 1
        tables = {row[0] for row in backup.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "blueprints" not in tables

    with sqlite3.connect(vault_path) as migrated:
        assert migrated.execute("SELECT COUNT(*) FROM rules").fetchone()[0] == 1
        assert migrated.execute("SELECT COUNT(*) FROM blueprints").fetchone()[0] == 1
        evidence = migrated.execute(
            "SELECT canonical_json FROM blueprints WHERE blueprint_id='crystal:0'"
        ).fetchone()[0]
        assert json.loads(evidence)["verification"]["level"] == "unverified"


def test_apply_refuses_to_overwrite_existing_backup(tmp_path):
    vault_path = tmp_path / "legacy.dcrystal"
    _make_legacy_vault(vault_path)
    backup_path = Path(str(vault_path) + ".pre-blueprint-migration.bak")
    backup_path.write_bytes(b"existing user backup")

    result = _run_cli("--vault", str(vault_path), "--apply")
    assert result.returncode != 0
    assert "backup already exists" in result.stderr
    assert backup_path.read_bytes() == b"existing user backup"
