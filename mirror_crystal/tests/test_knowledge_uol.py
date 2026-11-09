import pytest

from mirror_crystal.blueprint_contracts import EvidenceLevel, VerificationEvidence
from mirror_crystal.knowledge_foundation import KnowledgeKind, KnowledgeRecord
from mirror_crystal.knowledge_uol import KnowledgeRuleImporter
from mirror_crystal.operational_language import RuleStatus, UOL, UOLExecutionError, VariableSpec, Call, Var, Program
from mirror_crystal.universal_representation import EpistemicStatus, RepresentationProvenance


def rule_payload(rule_id="double"):
    return {
        "schema_version": 1,
        "rule_id": rule_id,
        "name": rule_id,
        "inputs": [{"name": "n", "value_type": "integer"}],
        "body": {
            "op": "call",
            "name": "multiply",
            "args": [{"op": "var", "name": "n"}, {"op": "lit", "value": 2}],
        },
        "domains": ["mathematics"],
    }


def record(
    knowledge_id="k:double", *, content=None, object_value="double",
    subject="double a value", epistemic=EpistemicStatus.ASSERTED,
    verification=None, provenance=None, kind=KnowledgeKind.RULE,
):
    return KnowledgeRecord(
        knowledge_id=knowledge_id, kind=kind, domain="mathematics",
        subject=subject, predicate="rule", object_value=object_value,
        statement="Multiply the input by two.", epistemic_status=epistemic,
        content=content if content is not None else {"uol_rule": rule_payload()},
        provenance=provenance or RepresentationProvenance(
            source="curated-source", source_ref=f"source://{knowledge_id}",
        ),
        verification=verification or VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
    )


def test_structured_rule_import_keeps_unverified_status_and_source_lineage():
    source = record()
    report = KnowledgeRuleImporter().import_records([source])
    assert report.imported_count == 1
    imported = report.rules[0]
    assert imported.status is RuleStatus.UNKNOWN
    assert source.knowledge_id in imported.provenance
    assert source.provenance.source_ref in imported.provenance
    program = Program(
        "p:test", "test", (VariableSpec("x", "integer"),),
        Call("double", (Var("x"),)), "mathematics", rule_refs=("double",),
    )
    with pytest.raises(UOLExecutionError):
        UOL(report.rules).execute(program, {"x": 4})


def test_formal_evidence_is_required_before_rule_is_marked_verified():
    evidence = VerificationEvidence(
        level=EvidenceLevel.FORMALLY_VERIFIED,
        checks_passed=("symbolic-proof-check",),
        verifier="independent-math-verifier",
    )
    provenance = RepresentationProvenance(
        source="proof-corpus", source_ref="proof://double",
        content_sha256="a" * 64,
    )
    source = record(epistemic=EpistemicStatus.DERIVED, verification=evidence, provenance=provenance)
    result = KnowledgeRuleImporter().import_records([source])
    assert result.rules[0].status is RuleStatus.VERIFIED


def test_claim_conflicts_force_all_related_rules_to_disputed():
    first = record("k:first", object_value="double", content={"uol_rule": rule_payload("double_a")})
    second = record("k:second", object_value="triple", content={"uol_rule": rule_payload("triple_a")})
    report = KnowledgeRuleImporter().import_records([first, second])
    assert len(report.rules) == 2
    assert {item.status for item in report.rules} == {RuleStatus.DISPUTED}


def test_natural_language_only_rule_is_skipped_not_compiled():
    source = record(content={"description": "multiply the input by two"})
    report = KnowledgeRuleImporter().import_records([source])
    assert not report.rules
    assert report.skipped[0].code == "no_structured_uol_rule"


def test_malformed_payload_is_rejected_and_rule_id_collision_quarantines_both():
    malformed = record("k:bad", content={"uol_rule": {"inputs": "not-a-list", "body": {}}})
    bad_report = KnowledgeRuleImporter().import_records([malformed])
    assert bad_report.rejected_count == 1
    assert not bad_report.rules

    first = record("k:one", content={"uol_rule": rule_payload("shared")})
    second_payload = rule_payload("shared")
    second_payload["body"] = {
        "op": "call", "name": "multiply",
        "args": [{"op": "var", "name": "n"}, {"op": "lit", "value": 3}],
    }
    second = record("k:two", content={"uol_rule": second_payload})
    collision = KnowledgeRuleImporter().import_records([first, second])
    assert not collision.rules
    assert collision.conflicted_rule_ids == ("shared",)
    assert collision.rejected_count == 2
