from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import zlib

from scripts.audit_domain_rule_inventory import audit_vault


def _payload(prompt: str, response: str, domain: str, tier: int = 2) -> bytes:
    return zlib.compress(
        json.dumps(
            {"ap": prompt, "ar": response, "dom": domain, "tt": tier},
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _make_vault(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE rules (
                rule_id INTEGER PRIMARY KEY,
                prompt_hash BLOB UNIQUE NOT NULL,
                sig_bytes BLOB NOT NULL,
                num_slots INTEGER NOT NULL,
                pass_count INTEGER NOT NULL,
                fail_count INTEGER NOT NULL,
                payload BLOB NOT NULL
            );
            CREATE TABLE blueprints (
                blueprint_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                intent TEXT NOT NULL,
                kind TEXT NOT NULL,
                canonical_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            """
        )
        rows = [
            (1, b"hash1", b"\0", 0, 1, 0, _payload("How do I decompose a task?", "Split it into subgoals.", "general_reasoning", 2)),
            (2, b"hash2", b"\0", 0, 1, 0, _payload("HOW do I decompose a task!", "Break the goal into smaller parts.", "deductive_logic", 3)),
            (3, b"hash3", b"\0", 0, 1, 0, _payload("What is photosynthesis?", "Plants convert light energy.", "biology", 1)),
        ]
        conn.executemany(
            "INSERT INTO rules VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.execute(
            "INSERT INTO blueprints VALUES (?, ?, ?, ?, ?, ?)",
            (
                "reasoning:1",
                "general_reasoning",
                "decompose",
                "procedure",
                json.dumps({"verification": {"level": "unverified"}}),
                1.0,
            ),
        )


def test_audit_reports_related_domains_and_normalized_duplicates(tmp_path: Path):
    vault = tmp_path / "existing.dcrystal"
    export = tmp_path / "selected.jsonl"
    _make_vault(vault)

    report = audit_vault(
        vault,
        domains=("general_reasoning", "deductive_logic"),
        export_jsonl=export,
    )

    assert report["read_only"] is True
    assert report["rules_total"] == 3
    assert report["selected_rules_total"] == 2
    assert report["selected_rules_by_domain"] == {
        "deductive_logic": 1,
        "general_reasoning": 1,
    }
    assert report["selected_exact_normalized_prompt_duplicate_groups"] == 1
    assert report["canonical_blueprints_total"] == 1
    assert report["canonical_blueprints_by_evidence_level"] == {"unverified": 1}

    exported = [json.loads(line) for line in export.read_text(encoding="utf-8").splitlines()]
    assert len(exported) == 2
    assert {row["domain"] for row in exported} == {"general_reasoning", "deductive_logic"}
    assert all("prompt_sha256" in row and "response_sha256" in row for row in exported)


def test_audit_rejects_missing_vault_without_creating_it(tmp_path: Path):
    missing = tmp_path / "missing.dcrystal"
    try:
        audit_vault(missing)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing vault should be rejected")
    assert not missing.exists()
