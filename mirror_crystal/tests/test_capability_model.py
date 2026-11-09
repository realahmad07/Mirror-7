from __future__ import annotations

from pathlib import Path

import pytest

from mirror_crystal.blueprint_contracts import (
    Blueprint, BlueprintKind, BlueprintProvenance, EvidenceLevel, VerificationEvidence,
)
from mirror_crystal.capability_model import (
    CapabilityConflictError, CapabilityRegistry, CapabilitySpec,
    CapabilityValidationError, PortContract,
)
from mirror_crystal.disk_vault import DiskCrystalVault
from mirror_crystal.universal_representation import RepresentationProvenance


def verified():
    return VerificationEvidence(
        level=EvidenceLevel.EXECUTION_TESTED,
        checks_passed=("unit-test:add-basic", "unit-test:add-negative"),
        verifier="pytest",
        verified_at="2026-11-09T00:00:00Z",
    )


def capability(capability_id="math.add", **overrides):
    values = dict(
        capability_id=capability_id,
        name="Add two numbers",
        description="Return the sum of two numeric inputs.",
        domains=("mathematics",),
        intents=("calculate",),
        inputs=(PortContract("left", "number"), PortContract("right", "number")),
        outputs=(PortContract("result", "number"),),
        provenance=RepresentationProvenance(source="test-suite"),
        verification=verified(),
    )
    values.update(overrides)
    return CapabilitySpec(**values)


def test_port_runtime_types_are_strict_and_bool_is_not_integer():
    assert PortContract("n", "integer").accepts_value(4)
    assert not PortContract("n", "integer").accepts_value(True)
    assert PortContract("n", "number").accepts_value(4.5)
    assert not PortContract("n", "number").accepts_value("4")
    assert PortContract("record", "object").accepts_value({"x": 1})


def test_port_output_compatibility_preserves_units_and_dimensions():
    consumer = PortContract("distance", "number", unit="m", dimensions={"length": 1})
    producer = PortContract("distance", "integer", unit="m", dimensions={"length": 1})
    wrong_unit = PortContract("distance", "number", unit="cm", dimensions={"length": 1})
    wrong_dimension = PortContract("distance", "number", unit="m", dimensions={"time": 1})
    assert consumer.accepts_output(producer)
    assert not consumer.accepts_output(wrong_unit)
    assert not consumer.accepts_output(wrong_dimension)


def test_capability_contract_round_trip_and_evidence_gate():
    item = capability()
    restored = CapabilitySpec.from_dict(item.to_dict())
    assert restored == item
    assert restored.fingerprint == item.fingerprint
    assert restored.executable
    legacy = capability(
        "legacy.add",
        verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
    )
    assert not legacy.executable
    assert "not execution-eligible" in legacy.evidence_blocker


def test_registry_is_deterministic_and_rejects_identifier_conflicts():
    registry = CapabilityRegistry((capability("b"), capability("a")))
    assert [item.capability_id for item in registry.list()] == ["a", "b"]
    assert registry.get("a") is not None
    registry.add(capability("a"))
    with pytest.raises(CapabilityConflictError):
        registry.add(capability("a", name="Different meaning"))


def test_blueprint_adapter_does_not_promote_legacy_evidence():
    blueprint = Blueprint(
        blueprint_id="legacy-add",
        kind=BlueprintKind.FORMULA,
        domain="mathematics",
        intent="calculate",
        prompt_pattern="add {v1} and {v2}",
        response_template="{v1} + {v2}",
        slots=("v1", "v2"),
        provenance=BlueprintProvenance(source="legacy"),
        verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
    )
    item = CapabilitySpec.from_blueprint(blueprint)
    assert item.capability_id == "blueprint:legacy-add"
    assert tuple(port.name for port in item.inputs) == ("v1", "v2")
    assert item.verification.level is EvidenceLevel.UNVERIFIED
    assert not item.executable


def test_capabilities_persist_and_reload_through_representation_graph(tmp_path: Path):
    db = tmp_path / "capabilities.db"
    original = capability()
    vault = DiskCrystalVault(db)
    try:
        assert vault.store_capability(original) == original.capability_id
        assert vault.get_capability(original.capability_id) == original
        assert vault.list_capabilities(domain="mathematics") == (original,)
    finally:
        vault.close()
    reopened = DiskCrystalVault(db)
    try:
        assert reopened.get_capability(original.capability_id) == original
    finally:
        reopened.close()


def test_invalid_contracts_are_rejected():
    with pytest.raises(CapabilityValidationError):
        PortContract("", "number")
    with pytest.raises(CapabilityValidationError):
        PortContract("x", "number", dimensions={"length": True})
    with pytest.raises(CapabilityValidationError):
        capability(domains=())
    with pytest.raises(CapabilityValidationError):
        capability(inputs=(PortContract("x"), PortContract("x")))
