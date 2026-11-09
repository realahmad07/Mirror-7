#!/usr/bin/env python3
"""Read-only inventory of an existing Mirror Crystal vault before domain expansion.

This tool never opens the vault through DiskCrystalVault and never writes to it.
It reads the compressed legacy rule payloads and canonical blueprint metadata
directly from SQLite in read-only mode. An optional JSONL export is local-only
and is intended for duplicate/coverage review; do not commit private exports.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any
import zlib

DEFAULT_DOMAINS = (
    "general_reasoning",
    "general",
    "relational_logic",
    "deductive_logic",
)


def normalize_prompt(value: str) -> str:
    """Normalize case and punctuation for exact-text duplicate detection."""
    return re.sub(r"[^\w]+", " ", value.casefold(), flags=re.UNICODE).strip()


def audit_vault(
    vault_path: Path | str,
    domains: tuple[str, ...] | list[str] = DEFAULT_DOMAINS,
    export_jsonl: Path | str | None = None,
) -> dict[str, Any]:
    """Inventory rules and blueprints without modifying the supplied vault."""
    path = Path(vault_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"vault does not exist: {path}")

    selected_domains = {str(domain).strip().casefold() for domain in domains if str(domain).strip()}
    if not selected_domains:
        raise ValueError("at least one domain must be selected")

    rules_by_domain: Counter[str] = Counter()
    selected_by_domain: Counter[str] = Counter()
    selected_by_tier: Counter[str] = Counter()
    duplicate_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_rules = 0
    selected_rules = 0
    export_path = Path(export_jsonl).expanduser() if export_jsonl else None
    if export_path:
        export_path.parent.mkdir(parents=True, exist_ok=True)

    # URI mode=ro is important: opening an audit must not initialize or mutate a vault.
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        tables = {
            str(row["name"])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "rules" not in tables:
            raise ValueError(f"no rules table found in vault: {path}")

        export_handle = export_path.open("w", encoding="utf-8", newline="\n") if export_path else None
        try:
            cursor = conn.execute(
                "SELECT rule_id, payload FROM rules ORDER BY rule_id"
            )
            for row in cursor:
                total_rules += 1
                rule_id = int(row["rule_id"])
                try:
                    payload = json.loads(zlib.decompress(row["payload"]).decode("utf-8"))
                except Exception as exc:
                    raise ValueError(f"cannot decode rule payload {rule_id}: {exc}") from exc

                domain = str(payload.get("dom", "general") or "general").strip().casefold()
                prompt = str(payload.get("ap", "") or "")
                response = str(payload.get("ar", "") or "")
                trust_tier = str(payload.get("tt", 2))
                rules_by_domain[domain] += 1

                if domain not in selected_domains:
                    continue

                selected_rules += 1
                selected_by_domain[domain] += 1
                selected_by_tier[trust_tier] += 1
                normalized = normalize_prompt(prompt)
                if normalized:
                    duplicate_groups[normalized].append(
                        {"rule_id": rule_id, "domain": domain}
                    )

                if export_handle:
                    item = {
                        "rule_id": rule_id,
                        "domain": domain,
                        "trust_tier": int(trust_tier),
                        "prompt": prompt,
                        "response": response,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                        "abstract_tool": str(payload.get("at", "") or ""),
                        "derived_expr": str(payload.get("de", "") or ""),
                    }
                    export_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        finally:
            if export_handle:
                export_handle.close()

        blueprints_total = 0
        blueprints_by_domain: Counter[str] = Counter()
        blueprints_by_evidence: Counter[str] = Counter()
        if "blueprints" in tables:
            for row in conn.execute("SELECT domain, canonical_json FROM blueprints"):
                blueprints_total += 1
                domain = str(row["domain"] or "general").strip().casefold()
                blueprints_by_domain[domain] += 1
                try:
                    record = json.loads(row["canonical_json"])
                    level = str(record.get("verification", {}).get("level", "unknown"))
                except Exception:
                    level = "invalid"
                blueprints_by_evidence[level] += 1

    duplicates = [
        {"normalized_prompt": key, "records": sorted(records, key=lambda x: x["rule_id"])}
        for key, records in duplicate_groups.items()
        if len(records) > 1
    ]
    duplicates.sort(key=lambda item: (-len(item["records"]), item["normalized_prompt"]))

    return {
        "vault_path": str(path),
        "read_only": True,
        "rules_total": total_rules,
        "rules_by_domain": dict(sorted(rules_by_domain.items())),
        "selected_domains": sorted(selected_domains),
        "selected_rules_total": selected_rules,
        "selected_rules_by_domain": dict(sorted(selected_by_domain.items())),
        "selected_rules_by_legacy_trust_tier": dict(sorted(selected_by_tier.items())),
        "selected_exact_normalized_prompt_duplicate_groups": len(duplicates),
        "selected_exact_normalized_prompt_duplicates": duplicates,
        "canonical_blueprints_total": blueprints_total,
        "canonical_blueprints_by_domain": dict(sorted(blueprints_by_domain.items())),
        "canonical_blueprints_by_evidence_level": dict(sorted(blueprints_by_evidence.items())),
        "export_jsonl": str(export_path.resolve()) if export_path else None,
        "limitations": [
            "Exact normalized-prompt matches are detected; semantic near-duplicates require review.",
            "Legacy trust tiers are metadata, not proof of factual correctness.",
            "Domain labels are reported as stored and are not silently remapped.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, required=True, help="Existing local .dcrystal SQLite vault")
    parser.add_argument(
        "--domain",
        action="append",
        dest="domains",
        help="Domain label to include; repeat for related labels. Defaults to general_reasoning, general, relational_logic, deductive_logic.",
    )
    parser.add_argument("--output", type=Path, help="Write the JSON inventory report to this local path")
    parser.add_argument(
        "--export-jsonl",
        type=Path,
        help="Optionally export selected prompts/responses for manual semantic review; keep this file private and out of Git",
    )
    args = parser.parse_args(argv)

    try:
        report = audit_vault(args.vault, tuple(args.domains or DEFAULT_DOMAINS), args.export_jsonl)
    except (OSError, sqlite3.Error, ValueError) as exc:
        parser.error(str(exc))

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
