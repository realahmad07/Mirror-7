from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mirror_crystal.blueprint_contracts import (
    Blueprint, BlueprintKind, BlueprintProvenance, EvidenceLevel, VerificationEvidence,
)
from mirror_crystal.disk_vault import DiskCrystalVault
from mirror_crystal.knowledge_foundation import (
    KnowledgeConflictError, KnowledgeKind, KnowledgeRecord, KnowledgeRegistry,
    KnowledgeValidationError,
)
from mirror_crystal.universal_representation import (
    EpistemicStatus, RepresentationProvenance,
)


def provenance(name="source", ref="https://example.org/knowledge"):
    return RepresentationProvenance(
        source=name,
        source_ref=ref,
        license_spdx="CC-BY-4.0",
        content_sha256=hashlib.sha256(name.encode()).hexdigest(),
        captured_at="2026-11-09T12:00:00+00:00",
    )


def verified():
    return VerificationEvidence(
        level=EvidenceLevel.INDEPENDENTLY_REPRODUCED,
        checks_passed=("reviewer-a:claim-check", "reviewer-b:claim-check"),
        verifier="independent reviewers",
        verified_at="2026-11-09T12:00:00+00:00",
    )


def fact(identifier, value, *, evidence=None, status=EpistemicStatus.ASSERTED):
    return KnowledgeRecord(
        knowledge_id=identifier,
        kind=KnowledgeKind.FACT,
        domain="physics",
        subject="water",
        predicate="boiling_point_at_standard_pressure",
        object_value=value,
        statement=f"Water boils at {value} K at standard pressure.",
        epistemic_status=status,
        provenance=provenance(),
        verification=evidence or verified(),
    )


def procedure(identifier, intent, input_name, output_name, handler_postcondition):
    return KnowledgeRecord(
        knowledge_id=identifier,
        kind=KnowledgeKind.PROCEDURE,
        domain="mathematics",
        subject=intent,
        predicate="native_operation",
        object_value={"operation": intent},
        statement=f"Native procedure for {intent}.",
        provenance=RepresentationProvenance(
            source="native-procedure-catalog",
            source_ref=f"mirror://procedure/{identifier}",
        ),
        verification=VerificationEvidence(
            level=EvidenceLevel.EXECUTION_TESTED,
            checks_passed=("procedure-contract-test",),
            verifier="native-procedure-test-suite",
        ),
        capability_contract={
            "domains": ["mathematics"],
            "intents": [intent],
            "inputs": [{"name": input_name, "value_type": "integer"}],
            "outputs": [{"name": output_name, "value_type": "integer"}],
            "postconditions": [handler_postcondition],
        },
    )


def test_assumptions_assertions_and_verified_facts_remain_distinct():
    assumption = KnowledgeRecord(
        knowledge_id="assumption:water",
        kind=KnowledgeKind.ASSUMPTION,
        domain="physics",
        subject="water",
        predicate="boiling_point",
        object_value=370,
        statement="Assume water boils at 370 K for this exercise.",
        epistemic_status=EpistemicStatus.ASSUMED,
        provenance=provenance("user-assumption", "user://assumption/1"),
    )
    unsupported = fact(
        "fact:unsupported",
        373.15,
        evidence=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
    )
    supported = fact("fact:supported", 373.15)
    assert not assumption.assertable
    assert assumption.epistemic_status is EpistemicStatus.ASSUMED
    assert not unsupported.assertable
    assert supported.assertable


def test_blueprint_adapter_preserves_unverified_evidence_and_source():
    prompt, response = "What is a test fact?", "A test fact is preserved."
    blueprint = Blueprint(
        blueprint_id="legacy:test-fact",
        kind=BlueprintKind.FACT,
        domain="general",
        intent="test_fact",
        prompt_pattern=prompt,
        response_template=response,
        provenance=BlueprintProvenance(
            source="legacy-import",
            source_ref="legacy://rule/12",
            license_spdx="MIT",
            content_sha256=Blueprint._digest(prompt, response),
            captured_at="2026-11-01T00:00:00+00:00",
        ),
        verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
    )
    record = KnowledgeRecord.from_blueprint(blueprint)
    assert record.knowledge_id == "blueprint:legacy:test-fact"
    assert record.verification.level is EvidenceLevel.UNVERIFIED
    assert record.provenance.source_ref == "legacy://rule/12"
    assert not record.assertable


def test_registry_detects_conflicting_claims_without_choosing_a_winner():
    first = fact("water:373", 373.15)
    second = fact("water:370", 370.0)
    registry = KnowledgeRegistry((first, second))
    assert len(registry.conflicts()) == 1
    results = registry.query("water boiling point", domain="physics")
    assert len(results) == 2
    assert all(item.conflicted and not item.assertable for item in results)


def test_registry_rejects_reused_identifier_with_different_content():
    registry = KnowledgeRegistry((fact("same-id", 373.15),))
    with pytest.raises(KnowledgeConflictError):
        registry.add(fact("same-id", 370.0))


def test_lexical_retrieval_is_explicit_and_can_exclude_assumptions():
    assumption = KnowledgeRecord(
        knowledge_id="assumption:water",
        kind=KnowledgeKind.ASSUMPTION,
        domain="physics",
        subject="water",
        predicate="boiling_point",
        object_value=370,
        statement="Assume water boils at 370 K.",
        epistemic_status=EpistemicStatus.ASSUMED,
        provenance=provenance("user-assumption", "user://assumption/2"),
    )
    supported = fact("fact:water", 373.15)
    registry = KnowledgeRegistry((assumption, supported))
    all_results = registry.query("water boiling point")
    filtered = registry.query("water standard pressure", include_assumptions=False)
    assert len(all_results) == 2
    assert len(filtered) == 1
    assert filtered[0].record.knowledge_id == "fact:water"
    assert filtered[0].score == 1.0


def test_procedure_becomes_capability_only_with_typed_contract_and_passing_evidence():
    record = procedure("double", "build", "input_value", "result", "result_created")
    capability = record.to_capability_spec()
    assert capability.capability_id == "knowledge:double"
    assert capability.executable
    assert capability.inputs[0].name == "input_value"
    assert capability.outputs[0].name == "result"

    no_contract = KnowledgeRecord(
        knowledge_id="procedure:unclear",
        kind=KnowledgeKind.PROCEDURE,
        domain="mathematics",
        subject="unclear",
        predicate="procedure",
        object_value="steps",
        statement="An untyped procedure.",
        provenance=provenance(),
        verification=VerificationEvidence(
            level=EvidenceLevel.EXECUTION_TESTED,
            checks_passed=("test",),
            verifier="tester",
        ),
    )
    with pytest.raises(KnowledgeValidationError, match="typed capability contract"):
        no_contract.to_capability_spec()


def test_knowledge_persists_in_representation_graph_and_contradictions_are_linked(tmp_path: Path):
    path = tmp_path / "knowledge.dcrystal"
    first = fact("water:373", 373.15)
    second = fact("water:370", 370.0)
    vault = DiskCrystalVault(path)
    try:
        assert vault.store_knowledge_record(first) == first.knowledge_id
        assert vault.store_knowledge_record(first) == first.knowledge_id
        assert vault.store_knowledge_record(second) == second.knowledge_id
        assert vault.get_knowledge_record(first.knowledge_id) == first
        assert len(vault.knowledge_conflicts(domain="physics")) == 1
        relations = vault.list_representation_relations(predicate="contradicts")
        assert len(relations) == 1
        assert relations[0].status is EpistemicStatus.DISPUTED
        search = vault.search_knowledge("water boiling point", domain="physics")
        assert len(search) == 2
        assert all(item.conflicted and not item.assertable for item in search)
    finally:
        vault.close()

    reopened = DiskCrystalVault(path)
    try:
        assert reopened.get_knowledge_record(first.knowledge_id) == first
        assert len(reopened.knowledge_conflicts(domain="physics")) == 1
    finally:
        reopened.close()


def test_knowledge_to_universal_representation_keeps_assumption_and_evidence():
    assumption = KnowledgeRecord(
        knowledge_id="assumption:mass",
        kind=KnowledgeKind.ASSUMPTION,
        domain="physics",
        subject="object",
        predicate="mass",
        object_value={"value": 2, "unit": "kg"},
        statement="Assume the object's mass is 2 kg.",
        epistemic_status=EpistemicStatus.ASSUMED,
        provenance=provenance("user", "user://input"),
    )
    node = assumption.to_representation_node()
    assert node.kind.value == "fact"
    assert node.status is EpistemicStatus.ASSUMED
    assert node.verification.level is EvidenceLevel.UNVERIFIED
    assert node.content["node_role"] == "knowledge_record"

def test_blueprint_with_review_grade_evidence_still_needs_claim_scope_review():
    prompt, response = "What is a reviewed fact?", "The reviewed value is 42."
    blueprint = Blueprint(
        blueprint_id="review-gate",
        kind=BlueprintKind.FACT,
        domain="general",
        intent="reviewed_fact",
        prompt_pattern=prompt,
        response_template=response,
        provenance=BlueprintProvenance(
            source="primary-source",
            source_ref="https://example.org/fact",
            license_spdx="CC-BY-4.0",
            content_sha256=Blueprint._digest(prompt, response),
            captured_at="2026-11-09T12:00:00+00:00",
        ),
        verification=VerificationEvidence(
            level=EvidenceLevel.INDEPENDENTLY_REPRODUCED,
            checks_passed=("source-check",),
            verifier="review-process",
        ),
    )
    record = KnowledgeRecord.from_blueprint(blueprint)
    assert not record.assertable
    assert record.content["claim_evidence_reliable"] is False
