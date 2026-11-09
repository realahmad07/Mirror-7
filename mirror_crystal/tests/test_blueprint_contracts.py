from __future__ import annotations
from types import SimpleNamespace
import pytest
from mirror_crystal.blueprint_contracts import (
    Blueprint, BlueprintDependencyGraph, BlueprintKind, BlueprintProvenance,
    BlueprintValidationError, EvidenceLevel, VerificationEvidence,
)
from mirror_crystal.lsh_rule_store import CrystalRule

def make_blueprint(blueprint_id: str, dependencies: tuple[str, ...] = ()) -> Blueprint:
    return Blueprint(
        blueprint_id=blueprint_id, kind=BlueprintKind.PROCEDURE, domain="systems",
        intent="explain_and_apply", prompt_pattern="How do I do {v0}?",
        response_template="Apply the procedure to {v0}.", slots=("v0",),
        dependencies=dependencies, provenance=BlueprintProvenance(source="unit_test", license_spdx="MIT"),
    )

def test_blueprint_round_trip_is_canonical():
    blueprint = make_blueprint("systems:procedure:1")
    restored = Blueprint.from_dict(blueprint.to_dict())
    assert restored == blueprint
    assert restored.canonical_json() == blueprint.canonical_json()

def test_verified_evidence_requires_a_named_passing_check():
    with pytest.raises(BlueprintValidationError, match="passing check"):
        VerificationEvidence(level=EvidenceLevel.EXECUTION_TESTED)

def test_legacy_crystal_rule_is_not_promoted_to_verified():
    rule = CrystalRule(
        rule_id=7, signature=3, salient_tokens=("mod", "pow"),
        abstract_prompt="Compute {v0}^{v1} mod {v2}", abstract_response="Result: {result}",
        num_slots=3, derived_expr="pow({v0}, {v1}, {v2})", trust_tier=1,
    )
    blueprint = Blueprint.from_crystal_rule(rule)
    assert blueprint.kind is BlueprintKind.FORMULA
    assert blueprint.verification.level is EvidenceLevel.UNVERIFIED
    assert blueprint.provenance.source == "legacy_crystal_rule"
    assert blueprint.legacy_trust_tier == 1

def test_roadmap_record_test_code_is_not_assumed_to_have_passed():
    record = SimpleNamespace(
        stage="STAGE_12_CODING", domain="coding", prompt="Write f(x)",
        response="def f(x): return x", tool="python", test_code="assert f(2) == 2",
        spdx_license="MIT", trust_tier=1, metadata={},
    )
    blueprint = Blueprint.from_roadmap_record(record)
    assert blueprint.kind is BlueprintKind.PROCEDURE
    assert blueprint.verification.level is EvidenceLevel.UNVERIFIED

def test_dependency_graph_returns_dependency_first_order():
    base, middle, leaf = make_blueprint("base"), make_blueprint("middle", ("base",)), make_blueprint("leaf", ("middle",))
    graph = BlueprintDependencyGraph((leaf, middle, base))
    assert [item.blueprint_id for item in graph.resolve("leaf")] == ["base", "middle", "leaf"]

def test_dependency_graph_rejects_missing_dependencies_and_cycles():
    missing = make_blueprint("needs-missing", ("absent",))
    with pytest.raises(BlueprintValidationError, match="missing dependency"):
        BlueprintDependencyGraph((missing,)).resolve("needs-missing")
    a, b = make_blueprint("a", ("b",)), make_blueprint("b", ("a",))
    with pytest.raises(BlueprintValidationError, match="dependency cycle"):
        BlueprintDependencyGraph((a, b)).resolve("a")

def test_contract_rejects_unsupported_schema_and_bad_trust_tier():
    with pytest.raises(BlueprintValidationError, match="schema_version"):
        Blueprint(blueprint_id="x", kind="fact", domain="general", intent="answer",
                  prompt_pattern="q", response_template="a", schema_version=99)
    with pytest.raises(BlueprintValidationError, match="trust_tier"):
        Blueprint(blueprint_id="x", kind="fact", domain="general", intent="answer",
                  prompt_pattern="q", response_template="a", legacy_trust_tier=0)
