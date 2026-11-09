"""Conservative, resumable migration of legacy .dcrystal rules into canonical blueprints.

Default mode is dry-run. Legacy rules are always imported as unverified. A review
manifest is required before independent-review promotion can occur.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from mirror_crystal import DiskCrystalVault
from mirror_crystal.claim_verification import ClaimEvidence, ClaimEvidenceError


def _read_manifest(path: Path) -> list[tuple[ClaimEvidence, tuple[str, ...]]]:
    rows: list[tuple[ClaimEvidence, tuple[str, ...]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            data = json.loads(raw)
            expected = tuple(str(x) for x in data.pop("expected_claim_ids", []))
            rows.append((ClaimEvidence.from_dict(data), expected))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", required=True, type=Path, help="Path to an existing .dcrystal SQLite vault")
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--apply", action="store_true", help="Apply the additive migration (default: dry-run)")
    parser.add_argument("--review-manifest", type=Path, help="JSONL of independently reviewed ClaimEvidence records")
    parser.add_argument(
        "--promote-reviewed",
        action="store_true",
        help="Promote only blueprints whose entire frozen claim scope has two independent approvals per claim",
    )
    args = parser.parse_args(argv)

    if not args.vault.is_file():
        parser.error(f"vault does not exist: {args.vault}")
    if not 1 <= args.page_size <= 1000:
        parser.error("--page-size must be between 1 and 1000")
    if args.promote_reviewed and (not args.apply or not args.review_manifest):
        parser.error("--promote-reviewed requires both --apply and --review-manifest")

    manifest_rows = _read_manifest(args.review_manifest) if args.review_manifest else []

    if not args.apply:
        # Use SQLite read-only URI so even opening a legacy vault cannot run schema migrations.
        uri = f"file:{args.vault.resolve().as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as read_only:
            tables = {row[0] for row in read_only.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            total_rules = int(read_only.execute("SELECT COUNT(*) FROM rules").fetchone()[0]) if "rules" in tables else 0
            total_blueprints = (
                int(read_only.execute("SELECT COUNT(*) FROM blueprints").fetchone()[0])
                if "blueprints" in tables else 0
            )
        missing = max(0, total_rules - total_blueprints)
        print(json.dumps({
            "mode": "dry_run",
            "vault": str(args.vault),
            "legacy_rules": total_rules,
            "canonical_blueprints": total_blueprints,
            "legacy_rules_not_yet_backfilled_lower_bound": missing,
            "review_manifest_records": len(manifest_rows),
            "promotion_will_occur": False,
            "vault_opened_read_only": True,
            "warning": "Dry-run makes no schema or data changes. Legacy trust_tier never becomes verification evidence.",
        }, indent=2))
        return 0

    vault = DiskCrystalVault(args.vault)
    try:
        cursor = -1
        migrated = 0
        scanned = 0
        while True:
            page = vault.backfill_legacy_blueprints(after_rule_id=cursor, limit=args.page_size)
            migrated += int(page["migrated"])
            scanned += int(page["scanned"])
            next_cursor = int(page["next_after_rule_id"])
            if int(page["scanned"]) == 0 or next_cursor <= cursor:
                break
            cursor = next_cursor

        evidence_stored = 0
        for evidence, expected_claim_ids in manifest_rows:
            vault.store_claim_evidence(evidence)
            evidence_stored += 1
            if expected_claim_ids:
                vault.register_blueprint_claim_scope(evidence.blueprint_id, expected_claim_ids)

        promoted: list[str] = []
        pending: dict[str, str] = {}
        if args.promote_reviewed:
            blueprint_ids = sorted({evidence.blueprint_id for evidence, _ in manifest_rows})
            for blueprint_id in blueprint_ids:
                try:
                    vault.curate_blueprint_from_evidence(blueprint_id)
                    promoted.append(blueprint_id)
                except ClaimEvidenceError as exc:
                    pending[blueprint_id] = str(exc)

        vault.optimize_disk()
        print(json.dumps({
            "mode": "applied",
            "legacy_rules_scanned": scanned,
            "canonical_blueprints_added": migrated,
            "claim_review_records_stored": evidence_stored,
            "independently_reviewed_blueprints_promoted": promoted,
            "pending_blueprints": pending,
            "total_blueprints": len(vault.list_blueprints(limit=10000)),
            "legacy_rules_preserved": True,
        }, indent=2))
        return 0
    finally:
        vault.close()


if __name__ == "__main__":
    raise SystemExit(main())
