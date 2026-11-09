from __future__ import annotations

import math
import pytest

from mirror_crystal import (
    Blueprint,
    BlueprintKind,
    BlueprintProvenance,
    DiskCrystalVault,
    EpistemicStatus,
    EvidenceLevel,
    RepresentationConflictError,
    RepresentationGraph,
    RepresentationKind,
    RepresentationNode,
    RepresentationProvenance,
    RepresentationRelation,
    RepresentationUncertainty,
    RepresentationValidationError,
    UncertaintyKind,
    VerificationEvidence,
)


def make_node(node_id, kind=RepresentationKind.ENTITY, domain="general", **kwargs):
    return RepresentationNode(
        node_id=node_id,
        kind=kind,
        domain=domain,
        label=kwargs.pop("label", node_id),
        provenance=kwargs.pop(
            "provenance", RepresentationProvenance(source="test-fixture", license_spdx="MIT")
        ),
        **kwargs,
    )


def make_relation(relation_id, source, predicate, target, domain="general"):
    return RepresentationRelation(
        relation_id=relation_id,
        source_id=source,
        predicate=predicate,
        target_id=target,
        domain=domain,
        status=EpistemicStatus.ASSERTED,
        provenance=RepresentationProvenance(source="test-fixture"),
    )


def test_node_round_trip_preserves_domain_types_units_and_evidence():
    item = make_node(
        "measurement:water-boiling-point",
        RepresentationKind.MEASUREMENT,
        "chemistry",
        content={"value": "373.15", "scale": "absolute"},
        value_type="quantity",
        unit="K",
        dimensions={"temperature": 1},
        status=EpistemicStatus.ASSUMED,
        uncertainty=RepresentationUncertainty(
            kind=UncertaintyKind.MEASUREMENT,
            confidence=0.95,
            lower_bound=373.0,
            upper_bound=373.3,
            unit="K",
        ),
        verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
    )
    restored = RepresentationNode.from_dict(item.to_dict())
    assert restored == item
    assert restored.canonical_json() == item.canonical_json()
    assert restored.status is EpistemicStatus.ASSUMED
    assert restored.verification.level is EvidenceLevel.UNVERIFIED
    assert restored.dimensions["temperature"] == 1


def test_epistemic_status_is_not_verification_level():
    assumed = make_node(
        "fact:assumed",
        RepresentationKind.FACT,
        status=EpistemicStatus.ASSUMED,
        verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
    )
    observed = make_node(
        "fact:observed",
        RepresentationKind.FACT,
        status=EpistemicStatus.OBSERVED,
        verification=VerificationEvidence(level=EvidenceLevel.CURATED),
    )
    assert assumed.status is EpistemicStatus.ASSUMED
    assert assumed.verification.level is EvidenceLevel.UNVERIFIED
    assert observed.status is EpistemicStatus.OBSERVED
    assert observed.verification.level is EvidenceLevel.CURATED


def test_uncertainty_rejects_invalid_confidence_and_intervals():
    with pytest.raises(RepresentationValidationError, match="between 0 and 1"):
        RepresentationUncertainty(confidence=1.1)
    with pytest.raises(RepresentationValidationError, match="lower_bound exceeds"):
        RepresentationUncertainty(lower_bound=5.0, upper_bound=2.0)
    with pytest.raises(RepresentationValidationError, match="both bounds"):
        RepresentationUncertainty(lower_bound=1.0)


def test_json_content_rejects_non_finite_and_non_json_values():
    with pytest.raises(RepresentationValidationError, match="non-finite"):
        make_node("bad:nan", content={"value": math.nan})
    with pytest.raises(RepresentationValidationError, match="JSON-compatible"):
        make_node("bad:object", content=object())
    with pytest.raises(RepresentationValidationError, match="keys must be strings"):
        make_node("bad:key", content={1: "not-json-compatible"})


def test_provenance_rejects_malformed_digest_and_duplicate_lineage():
    with pytest.raises(RepresentationValidationError, match="SHA-256"):
        RepresentationProvenance(source="fixture", content_sha256="bad")
    with pytest.raises(RepresentationValidationError, match="unique"):
        RepresentationProvenance(source="fixture", parent_ids=("a", "a"))


def test_graph_supports_cross_domain_typed_relations_and_round_trip():
    water = make_node("entity:water", domain="chemistry", content={"formula": "H2O"})
    oxygen = make_node("entity:oxygen", domain="chemistry", content={"symbol": "O"})
    mass = make_node("quantity:oxygen-mass", RepresentationKind.MEASUREMENT, "physics")
    graph = RepresentationGraph((water, oxygen, mass))
    graph.add_relation(make_relation(
        "r:water-oxygen", water.node_id, "contains_element", oxygen.node_id, "chemistry"
    ))
    graph.add_relation(make_relation(
        "r:water-mass", water.node_id, "measured_by", mass.node_id, "cross_domain"
    ))
    restored = RepresentationGraph.from_dict(graph.to_dict())
    assert restored.canonical_json() == graph.canonical_json()
    assert [edge.predicate for edge in restored.connected(
        "entity:water", direction="outgoing"
    )] == ["contains_element", "measured_by"]


def test_graph_rejects_missing_endpoints_and_conflicting_identifiers():
    graph = RepresentationGraph((make_node("a"),))
    with pytest.raises(RepresentationValidationError, match="endpoints must exist"):
        graph.add_relation(make_relation("r:missing", "a", "relates_to", "missing"))
    graph.add_node(make_node("b"))
    graph.add_node(make_node("a"))
    with pytest.raises(RepresentationConflictError, match="conflicting representation node"):
        graph.add_node(make_node("a", label="different"))
    graph.add_relation(make_relation("r:ab", "a", "relates_to", "b"))
    with pytest.raises(RepresentationConflictError, match="conflicting representation relation"):
        graph.add_relation(make_relation("r:ab", "b", "relates_to", "a"))


def test_vault_storage_is_idempotent_queryable_and_persistent(tmp_path):
    path = tmp_path / "representation.dcrystal"
    water = make_node("entity:water", domain="chemistry", content={"formula": "H2O"})
    oxygen = make_node("entity:oxygen", domain="chemistry", content={"symbol": "O"})
    edge = make_relation("r:contains", water.node_id, "contains_element", oxygen.node_id, "chemistry")

    vault = DiskCrystalVault(path)
    try:
        assert vault.store_representation_node(water) == water.node_id
        assert vault.store_representation_node(water) == water.node_id
        assert vault.store_representation_node(oxygen) == oxygen.node_id
        assert vault.store_representation_relation(edge) == edge.relation_id
        assert vault.store_representation_relation(edge) == edge.relation_id
        assert vault.get_representation_node(water.node_id) == water
        assert vault.get_representation_relation(edge.relation_id) == edge
        assert vault.list_representation_nodes(
            domain="chemistry", kind=RepresentationKind.ENTITY
        ) == (oxygen, water)
        assert vault.list_representation_relations(predicate="contains_element") == (edge,)
        expected = RepresentationGraph((water, oxygen), (edge,))
        assert vault.load_representation_graph(domain="chemistry").canonical_json() == expected.canonical_json()
        with pytest.raises(RepresentationConflictError):
            vault.store_representation_node(make_node(
                "entity:water", domain="chemistry", content={"formula": "H2O2"}
            ))
        with pytest.raises(RepresentationValidationError, match="unknown relation endpoint"):
            vault.store_representation_relation(make_relation(
                "r:unknown", water.node_id, "uses", "entity:absent"
            ))
    finally:
        vault.close()

    reopened = DiskCrystalVault(path)
    try:
        assert reopened.get_representation_node(water.node_id) == water
        assert reopened.get_representation_relation(edge.relation_id) == edge
        assert len(reopened.load_representation_graph(domain="chemistry").nodes) == 2
    finally:
        reopened.close()


def test_graph_storage_rolls_back_a_partial_write(tmp_path):
    vault = DiskCrystalVault(tmp_path / "atomic.dcrystal")
    try:
        existing = make_node("existing")
        vault.store_representation_node(existing)
        new_node = make_node("new-node")
        conflicting = make_node("existing", label="different")
        graph = RepresentationGraph((new_node, conflicting))
        with pytest.raises(RepresentationConflictError):
            vault.store_representation_graph(graph)
        assert vault.get_representation_node("new-node") is None
    finally:
        vault.close()


def test_blueprint_adapter_preserves_provenance_and_unverified_evidence(tmp_path):
    prompt = "Describe water."
    response = "Water is H2O."
    blueprint = Blueprint(
        blueprint_id="chem:water",
        kind=BlueprintKind.PROCEDURE,
        domain="chemistry",
        intent="describe_water",
        prompt_pattern=prompt,
        response_template=response,
        provenance=BlueprintProvenance(
            source="legacy-import",
            source_ref="legacy://rule/17",
            license_spdx="MIT",
            content_sha256=Blueprint._digest(prompt, response),
            captured_at="2026-11-09T00:00:00+00:00",
        ),
        verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
    )
    vault = DiskCrystalVault(tmp_path / "blueprint-adapter.dcrystal")
    try:
        result = vault.ingest_blueprints_as_representations([blueprint])
        assert result == {"nodes_added": 1, "relations_added": 0}
        adapted = vault.get_representation_node("blueprint:chem:water")
        assert adapted is not None
        assert adapted.kind is RepresentationKind.PROCEDURE
        assert adapted.provenance.source_ref == "legacy://rule/17"
        assert adapted.provenance.content_sha256 == Blueprint._digest(prompt, response)
        assert adapted.verification.level is EvidenceLevel.UNVERIFIED
    finally:
        vault.close()


def test_blueprint_dependencies_become_edges_when_both_blueprints_exist(tmp_path):
    base = Blueprint(
        blueprint_id="base",
        kind=BlueprintKind.FACT,
        domain="general",
        intent="base",
        prompt_pattern="Base?",
        response_template="Base.",
    )
    derived = Blueprint(
        blueprint_id="derived",
        kind=BlueprintKind.HEURISTIC,
        domain="general",
        intent="derived",
        prompt_pattern="Derived?",
        response_template="Derived.",
        dependencies=("base", "missing"),
    )
    vault = DiskCrystalVault(tmp_path / "dependency-adapter.dcrystal")
    try:
        result = vault.ingest_blueprints_as_representations([base, derived])
        assert result == {"nodes_added": 2, "relations_added": 1}
        edges = vault.list_representation_relations(predicate="depends_on")
        assert len(edges) == 1
        assert edges[0].source_id == "blueprint:derived"
        assert edges[0].target_id == "blueprint:base"
    finally:
        vault.close()
