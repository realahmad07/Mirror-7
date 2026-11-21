"""Step 2: Persistent UOL Skill Store backed by Mirror's canonical .dcrystal vault.

Persists verified forward, inverse-derived, and newly synthesized UOL rules as
canonical KnowledgeRecords in DiskCrystalVault, and reloads them cleanly through
KnowledgeRuleImporter across sessions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .disk_vault import DiskCrystalVault
from .knowledge_uol import KnowledgeRuleImporter, KnowledgeRuleImportReport
from .operational_language import OperationalRule, Program, RuleStatus, UOL
from .uol_inverse_deriver import (
    derive_all_inverse_rules,
    operational_rule_to_knowledge_record,
)
from .uol_rule_library import default_uol_knowledge_records, import_default_uol_rules


def persist_uol_rules_to_vault(
    vault: DiskCrystalVault,
    rules: Iterable[OperationalRule],
) -> dict[str, int]:
    """Persist verified OperationalRules into a .dcrystal vault as canonical KnowledgeRecords."""
    records = [
        operational_rule_to_knowledge_record(rule)
        for rule in rules
        if rule.status not in {RuleStatus.UNKNOWN, RuleStatus.DISPUTED}
    ]
    if not records:
        return {"stored": 0, "skipped": 0, "failed": 0}
    return vault.store_knowledge_records(records)


def persist_verified_program(
    vault: DiskCrystalVault,
    program: Program,
    *,
    rule_id: str,
    domains: tuple[str, ...] | None = None,
) -> OperationalRule:
    """Promote an independently verified UOL Program into a durable .dcrystal skill."""
    rule = OperationalRule(
        rule_id=rule_id,
        name=program.name,
        inputs=program.inputs,
        body=program.body,
        domains=domains or (program.domain,),
        status=RuleStatus.VERIFIED,
        provenance=f"uol-skill-store:promoted:{program.program_id}",
    )
    vault.store_knowledge_record(operational_rule_to_knowledge_record(rule))
    return rule


def seed_vault_with_uol_skills(
    vault_path: str | Path,
    *,
    include_inverses: bool = True,
) -> dict[str, Any]:
    """Create or update a .dcrystal vault with all 45 forward + 75 derived inverse UOL skills."""
    path = Path(vault_path)
    vault = DiskCrystalVault(path)
    try:
        forward_records = default_uol_knowledge_records()
        fwd_stats = vault.store_knowledge_records(forward_records)

        inv_count = 0
        inv_stats: dict[str, int] = {"stored": 0, "skipped": 0, "failed": 0}
        if include_inverses:
            imported = import_default_uol_rules()
            inverses = derive_all_inverse_rules(imported.rules)
            inv_count = len(inverses)
            inv_stats = persist_uol_rules_to_vault(vault, inverses)

        reloaded_records = vault.list_knowledge_records(limit=2000)
        import_report = KnowledgeRuleImporter().import_records(reloaded_records)
        return {
            "vault_path": str(path),
            "forward_rules": len(forward_records),
            "derived_inverse_rules": inv_count,
            "total_persisted_uol_rules": import_report.imported_count,
            "rejected": import_report.rejected_count,
            "conflicts": list(import_report.conflicted_rule_ids),
            "forward_store_stats": fwd_stats,
            "inverse_store_stats": inv_stats,
        }
    finally:
        vault.close()


def load_uol_from_vault(
    vault_or_path: DiskCrystalVault | str | Path,
    *,
    domain: str | None = None,
    limit: int = 2000,
) -> tuple[UOL, KnowledgeRuleImportReport]:
    """Load persisted UOL skills from a .dcrystal vault through KnowledgeRuleImporter."""
    owns_vault = not isinstance(vault_or_path, DiskCrystalVault)
    vault = DiskCrystalVault(vault_or_path) if owns_vault else vault_or_path
    try:
        records = vault.list_knowledge_records(domain=domain, limit=limit)
        report = KnowledgeRuleImporter().import_records(records)
        return UOL(report.rules), report
    finally:
        if owns_vault:
            vault.close()
