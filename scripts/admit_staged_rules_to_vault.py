#!/usr/bin/env python3
"""Conservative, deduplicated admission of staged Mirror 7 blueprints & reasoning rules.

Follows rules_and_blueprints/README.md and ANTIGRAVITY_CODEX_IMPORT_PROMPT.md:
- Never upgrades `unverified` or `source_supported` records to trusted/verified facts.
- Deduplicates by exact normalized prompt across all packs.
- Selects 1 canonical representative per `family_id` (320 families) from the
  10,240 context-specialized synthetic pack instead of inflating duplicates.
- Preserves all 120 verified executable UOL skills in artifacts/mirror_uol_skills.dcrystal.
- Emits a full SHA-256 audit report at artifacts/staged_rules_admission_report.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mirror_crystal import DiskCrystalVault
from mirror_crystal.blueprint_contracts import (
    Blueprint,
    BlueprintKind,
    BlueprintProvenance,
    EvidenceLevel,
    VerificationEvidence,
)
from mirror_crystal.uol_skill_store import load_uol_from_vault

_VALID_KINDS = {k.value for k in BlueprintKind}


def _safe_kind(raw: object) -> str:
    val = str(raw or "").strip().casefold()
    if val in _VALID_KINDS:
        return val
    if "formula" in val or "math" in val:
        return "formula"
    if "fact" in val or "def" in val:
        return "fact"
    if "policy" in val or "rule" in val:
        return "policy"
    if "heur" in val:
        return "heuristic"
    return "procedure"


def _norm_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).casefold().strip())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_staged_blueprints(repo: Path) -> tuple[list[Blueprint], dict]:
    seen_prompts: set[str] = set()
    seen_ids: set[str] = set()
    admitted: list[Blueprint] = []
    stats = {
        "roadmap_997_input": 0,
        "roadmap_997_selected": 0,
        "source_supported_26_input": 0,
        "source_supported_26_selected": 0,
        "synthetic_10240_input": 0,
        "synthetic_families_unique": 0,
        "synthetic_families_selected": 0,
        "duplicates_skipped": 0,
        "invalid_skipped": 0,
    }

    def try_add(bp: Blueprint) -> bool:
        norm = _norm_prompt(bp.prompt_pattern)
        if not norm or bp.blueprint_id in seen_ids or norm in seen_prompts:
            stats["duplicates_skipped"] += 1
            return False
        seen_ids.add(bp.blueprint_id)
        seen_prompts.add(norm)
        admitted.append(bp)
        return True

    # 1. Roadmap 997 unambiguous candidates
    p_roadmap = repo / "rules_and_blueprints/blueprints/staging/mirror_997_unambiguous_prompt_candidates.jsonl"
    with p_roadmap.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            stats["roadmap_997_input"] += 1
            row = json.loads(line)
            prompt = str(row.get("prompt", "")).strip()
            response = str(row.get("response", "")).strip()
            if not prompt or not response:
                stats["invalid_skipped"] += 1
                continue
            sha = hashlib.sha256((prompt + "\0" + response).encode("utf-8")).hexdigest()
            bp = Blueprint(
                blueprint_id=f"staged:roadmap:{row['blueprint_id']}",
                kind=_safe_kind(row.get("type", "procedure")),
                domain=str(row.get("domain", "general")).casefold()[:64] or "general",
                intent="explain",
                prompt_pattern=prompt,
                response_template=response,
                provenance=BlueprintProvenance(
                    source="mirror_997_unambiguous_prompt_candidates",
                    source_ref=f"{row['blueprint_id']}|stage={row.get('stage', '')}",
                    license_spdx="NOASSERTION",
                    content_sha256=sha,
                ),
                verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
                legacy_trust_tier=3,
            )
            if try_add(bp):
                stats["roadmap_997_selected"] += 1

    # 2. 26 source-supported general reasoning methods (kept UNVERIFIED per canonical enum)
    p_src26 = repo / "rules_and_blueprints/general_reasoning/rules/general_reasoning_source_supported_26.jsonl"
    with p_src26.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            stats["source_supported_26_input"] += 1
            row = json.loads(line)
            prompt = str(row.get("trigger", "")).strip()
            steps = row.get("steps") or ()
            exceptions = row.get("exceptions") or ()
            response = str(row.get("statement", "")).strip()
            if steps:
                response += "\n\nProcedure:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
            if exceptions:
                response += "\n\nLimitations:\n" + "\n".join(f"- {x}" for x in exceptions)
            if not prompt or not response:
                stats["invalid_skipped"] += 1
                continue
            sha = hashlib.sha256((prompt + "\0" + response).encode("utf-8")).hexdigest()
            prov = row.get("provenance") or {}
            bp = Blueprint(
                blueprint_id=f"staged:gr_src:{row['id']}",
                kind=_safe_kind(row.get("kind", "procedure")),
                domain=str(row.get("domain", "general_reasoning")).casefold()[:64],
                intent=str(row.get("intent", "analyze"))[:32],
                prompt_pattern=prompt,
                response_template=response,
                preconditions=tuple(str(x) for x in (row.get("preconditions") or ())[:8]),
                postconditions=tuple(str(x) for x in (row.get("postconditions") or ())[:8]),
                provenance=BlueprintProvenance(
                    source="general_reasoning_source_supported_26",
                    source_ref=f"{row['id']}|{prov.get('source_url', '')}",
                    license_spdx=str(prov.get("license_spdx", "NOASSERTION")),
                    content_sha256=sha,
                    captured_at=str(prov.get("retrieved_at", "")),
                ),
                verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
                legacy_trust_tier=3,
            )
            if try_add(bp):
                stats["source_supported_26_selected"] += 1

    # 3. 10,240 synthetic candidates -> 1 canonical representative per family_id (320 families)
    p_10k = repo / "rules_and_blueprints/general_reasoning/rules/general_reasoning_candidates_10240.jsonl"
    seen_families: set[str] = set()
    with p_10k.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            stats["synthetic_10240_input"] += 1
            row = json.loads(line)
            fam = str(row.get("family_id", "")).strip()
            if fam in seen_families:
                continue
            seen_families.add(fam)
            stats["synthetic_families_unique"] += 1
            prompt = str(row.get("trigger", "")).strip()
            steps = row.get("steps") or ()
            exceptions = row.get("exceptions") or ()
            response = str(row.get("statement", "")).strip()
            if steps:
                response += "\n\nProcedure:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
            if exceptions:
                response += "\n\nLimitations:\n" + "\n".join(f"- {x}" for x in exceptions)
            if not prompt or not response:
                stats["invalid_skipped"] += 1
                continue
            sha = hashlib.sha256((prompt + "\0" + response).encode("utf-8")).hexdigest()
            prov = row.get("provenance") or {}
            bp = Blueprint(
                blueprint_id=f"staged:gr_fam:{row['id']}",
                kind=_safe_kind(row.get("kind", "procedure")),
                domain=str(row.get("domain", "general_reasoning")).casefold()[:64],
                intent=str(row.get("intent", "analyze"))[:32],
                prompt_pattern=prompt,
                response_template=response,
                preconditions=tuple(str(x) for x in (row.get("preconditions") or ())[:8]),
                postconditions=tuple(str(x) for x in (row.get("postconditions") or ())[:8]),
                provenance=BlueprintProvenance(
                    source="general_reasoning_10k_family_representatives",
                    source_ref=f"{row['id']}|family={fam}|context={row.get('context_id', '')}",
                    license_spdx=str(prov.get("license_spdx", "NOASSERTION")),
                    content_sha256=sha,
                ),
                verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
                legacy_trust_tier=3,
            )
            if try_add(bp):
                stats["synthetic_families_selected"] += 1

    return admitted, stats


def main() -> int:
    vault_path = REPO_ROOT / "artifacts/mirror_uol_skills.dcrystal"
    report_path = REPO_ROOT / "artifacts/staged_rules_admission_report.json"

    if not vault_path.is_file():
        raise FileNotFoundError(f"Expected seeded UOL vault at {vault_path}")

    pre_sha = _sha256_file(vault_path)
    blueprints, stats = _load_staged_blueprints(REPO_ROOT)

    vault = DiskCrystalVault(vault_path)
    try:
        stored = 0
        for i in range(0, len(blueprints), 100):
            batch = blueprints[i : i + 100]
            for bp in batch:
                vault.store_blueprint(bp)
                stored += 1
        all_bps = vault.list_blueprints(limit=5000)
        unverified_count = sum(1 for b in all_bps if b.verification.level is EvidenceLevel.UNVERIFIED)
    finally:
        vault.close()

    # Integrity & UOL safety check
    with sqlite3.connect(vault_path) as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    _uol, uol_report = load_uol_from_vault(vault_path)

    report = {
        "vault_path": "artifacts/mirror_uol_skills.dcrystal",
        "pre_admission_sha256": pre_sha,
        "post_admission_sha256": _sha256_file(vault_path),
        "vault_size_bytes": vault_path.stat().st_size,
        "sqlite_integrity": integrity,
        "verified_uol_skills_preserved": uol_report.imported_count,
        "uol_conflicts": list(uol_report.conflicted_rule_ids),
        "staged_blueprints_admitted": stored,
        "total_blueprints_in_vault": len(all_bps),
        "all_admitted_blueprints_unverified": unverified_count == len(all_bps),
        "admission_stats": stats,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
