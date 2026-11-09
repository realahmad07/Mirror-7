"""Explicit capability persistence using the Phase 2 representation graph."""
from __future__ import annotations

from collections.abc import Mapping

from .capability_model import CapabilitySpec
from .universal_representation import (
    EpistemicStatus,
    RepresentationKind,
    RepresentationNode,
    RepresentationProvenance,
)


class VaultCapabilityRuntimeMixin:
    """Persist declared capabilities without executing or silently upgrading them."""

    def store_capability(self, capability: CapabilitySpec) -> str:
        if not isinstance(capability, CapabilitySpec):
            raise TypeError("capability must be a CapabilitySpec")
        node_id = f"capability:{capability.capability_id}"
        node = RepresentationNode(
            node_id=node_id,
            kind=RepresentationKind.OPERATION,
            domain=capability.domains[0],
            label=capability.name[:240],
            content={"node_role": "capability", "capability": capability.to_dict()},
            value_type="record",
            status=EpistemicStatus.ASSERTED,
            attributes={
                "capability_id": capability.capability_id,
                "executable": capability.executable,
                "evidence_level": capability.verification.level.value,
            },
            provenance=RepresentationProvenance(
                source="mirror_capability_registry",
                source_ref=capability.fingerprint,
                parent_ids=capability.dependencies,
            ),
            verification=capability.verification,
        )
        self.store_representation_node(node)
        return capability.capability_id

    def get_capability(self, capability_id: str) -> CapabilitySpec | None:
        node = self.get_representation_node(f"capability:{capability_id}")
        if node is None or node.kind is not RepresentationKind.OPERATION:
            return None
        if node.content.get("node_role") != "capability":
            return None
        payload = node.content.get("capability")
        if not isinstance(payload, Mapping):
            return None
        return CapabilitySpec.from_dict(payload)

    def list_capabilities(self, *, domain: str | None = None, limit: int = 500) -> tuple[CapabilitySpec, ...]:
        # A capability may support more than one domain while its representation
        # node has one canonical primary domain; filter after decoding the full
        # bounded catalog so secondary domains remain discoverable.
        nodes = self.list_representation_nodes(
            kind=RepresentationKind.OPERATION, limit=limit
        )
        result = []
        for node in nodes:
            if node.content.get("node_role") != "capability":
                continue
            payload = node.content.get("capability")
            if isinstance(payload, Mapping):
                result.append(CapabilitySpec.from_dict(payload))
        return tuple(sorted(result, key=lambda item: item.capability_id))
