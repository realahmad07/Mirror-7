from __future__ import annotations

from typing import Any, Mapping

from phase343_semantic_state import SemanticState


def build_reasoning_context(state: SemanticState | None) -> Mapping[str, Any] | None:
    """Convert semantic task state into an explicit, read-only reasoning context."""
    if state is None:
        return None
    return {
        "entities": tuple(state.entities),
        "operations": tuple(state.operations),
        "constraints": tuple(state.constraints),
        "context": tuple(state.context),
        "desired_output": state.desired_output,
        "evidence": tuple(state.evidence),
    }
