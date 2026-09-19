"""Mirror 7 Phases 181-190: bounded unstructured multimodal grounding.

This block provides representation-level cross-modal grounding without pretrained
neural models. It is deliberately deterministic, inspectable, and fail-closed.
"""
from .mirror7_phase181_190 import (
    ModalityView, GroundedConcept, MultimodalGroundingEngine,
    canonicalize_view, align_views, discover_grounded_concepts,
    bind_concept, contradiction_filter, build_grounded_scene,
)
__all__ = [
    "ModalityView","GroundedConcept","MultimodalGroundingEngine",
    "canonicalize_view","align_views","discover_grounded_concepts",
    "bind_concept","contradiction_filter","build_grounded_scene",
]
