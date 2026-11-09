"""Conservative bridge from canonical Mirror knowledge records into executable UOL rules.

Only explicitly structured content.uol_rule payloads are imported. Natural-language
claims are never compiled or guessed into executable operations.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from typing import Iterable

from .knowledge_foundation import KnowledgeKind, KnowledgeRecord, KnowledgeRegistry
from .operational_language import OperationalRule, RuleCatalog, RuleStatus, UOLError, knowledge_rule, expr_dict


@dataclass(frozen=True)
class KnowledgeRuleImportIssue:
    knowledge_id: str
    code: str
    message: str


@dataclass(frozen=True)
class KnowledgeRuleImportReport:
    rules: tuple[OperationalRule, ...]
    imported_knowledge_ids: tuple[str, ...]
    skipped: tuple[KnowledgeRuleImportIssue, ...]
    rejected: tuple[KnowledgeRuleImportIssue, ...]
    conflicted_rule_ids: tuple[str, ...]

    @property
    def imported_count(self) -> int:
        return len(self.imported_knowledge_ids)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    def catalog(self, base_rules: Iterable[OperationalRule] = ()) -> RuleCatalog:
        return RuleCatalog((*tuple(base_rules), *self.rules))


class KnowledgeRuleImporter:
    """Convert explicit, typed knowledge payloads into a validated UOL catalog."""

    VERSION = "knowledge-uol-import-v1"
    _ELIGIBLE_KINDS = {
        KnowledgeKind.RULE, KnowledgeKind.FORMULA, KnowledgeKind.PROCEDURE,
        KnowledgeKind.CONSTRAINT, KnowledgeKind.HEURISTIC, KnowledgeKind.ASSUMPTION,
    }
    _PAYLOAD_FIELDS = {"schema_version", "rule_id", "name", "inputs", "body", "domains"}

    def __init__(self, base_rules: Iterable[OperationalRule] = ()) -> None:
        self.base_rules = tuple(base_rules)
        RuleCatalog(self.base_rules)

    @staticmethod
    def _json(value) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _conflicted_claims(records) -> set[str]:
        values = defaultdict(set)
        for record in records:
            values[record.claim_key].add(record.object_fingerprint)
        return {key for key, fingerprints in values.items() if len(fingerprints) > 1}

    @staticmethod
    def _semantic_signature(rule: OperationalRule) -> str:
        return KnowledgeRuleImporter._json({
            "inputs": [{"name": item.name, "value_type": item.value_type} for item in rule.inputs],
            "body": expr_dict(rule.body),
            "domains": list(rule.domains),
        })

    @staticmethod
    def _weakest_status(statuses) -> RuleStatus:
        statuses = set(statuses)
        for status in (
            RuleStatus.DISPUTED, RuleStatus.UNKNOWN, RuleStatus.ASSUMED,
            RuleStatus.DERIVED, RuleStatus.OBSERVED, RuleStatus.VERIFIED,
        ):
            if status in statuses:
                return status
        return RuleStatus.UNKNOWN

    def import_records(self, records: Iterable[KnowledgeRecord] | KnowledgeRegistry) -> KnowledgeRuleImportReport:
        items = records.list() if isinstance(records, KnowledgeRegistry) else tuple(records)
        if any(not isinstance(record, KnowledgeRecord) for record in items):
            raise TypeError("all imported items must be KnowledgeRecord instances")

        conflicted_claims = self._conflicted_claims(items)
        skipped: list[KnowledgeRuleImportIssue] = []
        rejected: list[KnowledgeRuleImportIssue] = []
        candidates: list[tuple[KnowledgeRecord, OperationalRule]] = []

        for record in items:
            payload = record.content.get("uol_rule")
            if payload is None:
                skipped.append(KnowledgeRuleImportIssue(
                    record.knowledge_id, "no_structured_uol_rule",
                    "No content.uol_rule payload; prose is not compiled into executable code.",
                ))
                continue
            if record.kind not in self._ELIGIBLE_KINDS:
                rejected.append(KnowledgeRuleImportIssue(
                    record.knowledge_id, "unsupported_knowledge_kind",
                    f"Knowledge kind {record.kind.value!r} is not eligible for operational import.",
                ))
                continue
            if not hasattr(payload, "keys"):
                rejected.append(KnowledgeRuleImportIssue(
                    record.knowledge_id, "invalid_payload", "content.uol_rule must be an object.",
                ))
                continue
            unknown = set(payload) - self._PAYLOAD_FIELDS
            if unknown:
                rejected.append(KnowledgeRuleImportIssue(
                    record.knowledge_id, "unknown_payload_fields",
                    f"Unsupported uol_rule fields: {', '.join(sorted(unknown))}.",
                ))
                continue
            if payload.get("schema_version", 1) != 1:
                rejected.append(KnowledgeRuleImportIssue(
                    record.knowledge_id, "unsupported_schema",
                    "Only uol_rule schema_version 1 is supported.",
                ))
                continue
            inputs = payload.get("inputs")
            domains = payload.get("domains", (record.domain,))
            if not isinstance(inputs, (tuple, list)) or "body" not in payload:
                rejected.append(KnowledgeRuleImportIssue(
                    record.knowledge_id, "incomplete_payload",
                    "uol_rule requires typed inputs and a serialized UOL body.",
                ))
                continue
            if any(not hasattr(item, "get") or not isinstance(item.get("name"), str)
                   or not isinstance(item.get("value_type"), str) for item in inputs):
                rejected.append(KnowledgeRuleImportIssue(
                    record.knowledge_id, "invalid_input_schema",
                    "Each UOL input requires a string name and value_type.",
                ))
                continue
            if not isinstance(domains, (tuple, list)) or not domains or any(
                not isinstance(item, str) or not item.strip() for item in domains
            ):
                rejected.append(KnowledgeRuleImportIssue(
                    record.knowledge_id, "invalid_domains",
                    "UOL domains must be a non-empty list of non-empty strings.",
                ))
                continue
            if "rule_id" in payload and (
                not isinstance(payload["rule_id"], str) or not payload["rule_id"].strip()
            ):
                rejected.append(KnowledgeRuleImportIssue(
                    record.knowledge_id, "invalid_rule_id", "rule_id must be a non-empty string.",
                ))
                continue
            if "name" in payload and (
                not isinstance(payload["name"], str) or not payload["name"].strip()
            ):
                rejected.append(KnowledgeRuleImportIssue(
                    record.knowledge_id, "invalid_rule_name", "name must be a non-empty string.",
                ))
                continue
            try:
                rule = knowledge_rule(record, force_disputed=record.claim_key in conflicted_claims)
                candidates.append((record, rule))
            except (KeyError, TypeError, ValueError, UOLError) as exc:
                rejected.append(KnowledgeRuleImportIssue(
                    record.knowledge_id, "invalid_uol_rule", str(exc),
                ))

        grouped: dict[str, list[tuple[KnowledgeRecord, OperationalRule]]] = defaultdict(list)
        for pair in candidates:
            grouped[pair[1].rule_id].append(pair)

        accepted: list[tuple[KnowledgeRecord, OperationalRule]] = []
        accepted_source_ids: list[str] = []
        conflicted_rule_ids: list[str] = []
        for rule_id, group in sorted(grouped.items()):
            signatures = {self._semantic_signature(rule) for _, rule in group}
            if len(signatures) > 1:
                conflicted_rule_ids.append(rule_id)
                for record, _ in group:
                    rejected.append(KnowledgeRuleImportIssue(
                        record.knowledge_id, "rule_id_collision",
                        f"Rule id {rule_id!r} maps to different executable definitions; all colliding records were quarantined.",
                    ))
                continue
            if len(group) == 1:
                accepted.append(group[0])
                accepted_source_ids.append(group[0][0].knowledge_id)
                continue
            first_record, first_rule = group[0]
            merged_provenance = self._json([rule.provenance for _, rule in group])
            merged = OperationalRule(
                first_rule.rule_id, first_rule.name, first_rule.inputs, first_rule.body,
                first_rule.domains,
                self._weakest_status(rule.status for _, rule in group),
                merged_provenance,
            )
            accepted.append((first_record, merged))
            accepted_source_ids.extend(record.knowledge_id for record, _ in group)

        try:
            catalog = RuleCatalog((*self.base_rules, *(rule for _, rule in accepted)))
        except UOLError as exc:
            for record, _ in accepted:
                rejected.append(KnowledgeRuleImportIssue(
                    record.knowledge_id, "invalid_rule_graph", str(exc),
                ))
            accepted = []
            accepted_source_ids = []
            catalog = RuleCatalog(self.base_rules)

        accepted_ids = tuple(sorted(set(accepted_source_ids)))
        base_ids = {rule.rule_id for rule in self.base_rules}
        imported_rules = tuple(rule for rule in catalog.values() if rule.rule_id not in base_ids)
        return KnowledgeRuleImportReport(
            rules=imported_rules,
            imported_knowledge_ids=accepted_ids,
            skipped=tuple(skipped),
            rejected=tuple(rejected),
            conflicted_rule_ids=tuple(sorted(conflicted_rule_ids)),
        )
