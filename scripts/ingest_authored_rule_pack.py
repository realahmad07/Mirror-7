#!/usr/bin/env python3
"""Validate and optionally ingest the authored Mirror domain-rule seed pack.

Dry-run is the default. Applying creates a SQLite online backup before ingestion.
All records stay at trust_tier=3 and are explicitly marked unverified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = ROOT / "research" / "rule_packs" / "mirror_domain_seed_v1.jsonl"
DEFAULT_VAULT = ROOT / "artifacts" / "mirror_crystal_all_datasets.dcrystal"
EXPECTED_DOMAINS = {
    "olympiad_math",
    "general_reasoning",
    "algorithms",
    "general",
    "relational_logic",
    "physics_science",
    "provide a helpful response",
    "mathematics",
    "quantum_computing",
    "deductive_logic",
    "Specialized Cognitive Pillars",
}
EXPECTED_PER_DOMAIN = 10


def _normalise_prompt(value: str) -> str:
    import re
    return re.sub(r"[^\w]+", " ", value.casefold(), flags=re.UNICODE).strip()


def load_and_validate(path: Path) -> list[dict]:
    rows: list[dict] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            domain = str(row.get("domain", "")).strip()
            if len(question) < 12 or len(answer) < 40 or len(answer) > 6000:
                raise ValueError(f"line {line_number}: question/answer length outside policy")
            if domain not in EXPECTED_DOMAINS:
                raise ValueError(f"line {line_number}: unexpected domain {domain!r}")
            if int(row.get("trust_tier", 0)) != 3:
                raise ValueError(f"line {line_number}: authored seed must remain trust_tier=3")
            if row.get("source") != "mirror_authored_seed_v1" or row.get("evidence_status") != "unverified":
                raise ValueError(f"line {line_number}: provenance/evidence label is missing")
            key = _normalise_prompt(question)
            if not key or key in seen:
                raise ValueError(f"line {line_number}: duplicate or empty prompt")
            seen.add(key)
            counts[domain] += 1
            rows.append(row)

    missing = EXPECTED_DOMAINS - set(counts)
    extra = set(counts) - EXPECTED_DOMAINS
    wrong = {domain: count for domain, count in counts.items() if count != EXPECTED_PER_DOMAIN}
    if missing or extra or wrong:
        raise ValueError(
            f"domain balance failure: missing={sorted(missing)}, extra={sorted(extra)}, "
            f"counts={dict(sorted(counts.items()))}"
        )
    if len(rows) != EXPECTED_PER_DOMAIN * len(EXPECTED_DOMAINS):
        raise ValueError(f"expected 110 rows, found {len(rows)}")
    return rows


def make_backup(vault_path: Path, backup_path: Path) -> None:
    if backup_path.exists():
        raise FileExistsError(f"backup already exists: {backup_path}")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    uri = vault_path.resolve().as_uri() + "?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as source, sqlite3.connect(backup_path) as target:
            source.backup(target)
            integrity = target.execute("PRAGMA integrity_check").fetchone()
            if not integrity or integrity[0] != "ok":
                raise RuntimeError(f"backup integrity check failed: {integrity}")
    except Exception:
        backup_path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--apply", action="store_true", help="Write to the local vault (default: dry-run only)")
    args = parser.parse_args()

    rows = load_and_validate(args.pack)
    counts = dict(sorted(Counter(row["domain"] for row in rows).items()))
    report: dict[str, object] = {
        "mode": "apply_requested" if args.apply else "dry_run",
        "pack": str(args.pack),
        "rows_validated": len(rows),
        "per_domain": counts,
        "trust_tier": 3,
        "evidence_status": "unverified",
        "vault": str(args.vault),
        "vault_modified": False,
    }
    if not args.apply:
        print(json.dumps(report, indent=2))
        return 0

    if not args.vault.is_file():
        parser.error(f"local production vault does not exist: {args.vault}")
    backup = args.backup or Path(str(args.vault) + ".pre-authored-seed-v1.bak")
    make_backup(args.vault, backup)

    from mirror_crystal import DiskCrystalVault
    vault = DiskCrystalVault(args.vault)
    try:
        before = vault.rule_count()
        ingest = vault.ingest_jsonl(args.pack, batch_flush_rows=500)
        vault.optimize_disk()
        after = vault.rule_count()
        integrity = vault.conn.execute("PRAGMA integrity_check").fetchone()
        integrity_value = integrity[0] if integrity else "missing"
        if integrity_value != "ok":
            raise RuntimeError(f"vault integrity check failed; backup preserved at {backup}")
        if after - before != int(ingest.get("new_rules", -1)):
            raise RuntimeError(f"rule-count mismatch; backup preserved at {backup}")
    finally:
        vault.close()

    report.update({
        "mode": "applied",
        "backup_path": str(backup),
        "vault_modified": True,
        "rules_before": before,
        "rules_after": after,
        "new_rules": ingest.get("new_rules", 0),
        "duplicates_skipped": ingest.get("duplicates_skipped", 0),
        "sandbox_rejected": ingest.get("sandbox_rejected", 0),
        "sqlite_integrity_check": integrity_value,
    })
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
