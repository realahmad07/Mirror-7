"""
Mirror 7 — Phase 31.5: Raw -> State Integration Pipeline

Provides a clean, unified, end-to-end pipeline connecting:
  Raw Observation -> Discovered Representation -> Stable State -> Temporal State Identity

Callers supply raw undifferentiated byte data directly without manually instantiating
intermediate representations, schemas, or semantic labels.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

try:
    from .mirror7_representation import DiscoveredRepresentation, discover_representation
    from .mirror7_state import StructuralState, extract_state
    from .mirror7_temporal import TemporalStateTracker, StateTransition
except (ImportError, ValueError):
    from mirror7_representation import DiscoveredRepresentation, discover_representation
    from mirror7_state import StructuralState, extract_state
    from mirror7_temporal import TemporalStateTracker, StateTransition


def raw_to_state(raw_observation: Any) -> StructuralState:
    """
    Direct functional pipeline:
      raw bytes -> discover_representation() -> extract_state() -> StructuralState

    No intermediate object instantiation or caller-supplied schemas needed.
    """
    rep = discover_representation(raw_observation)
    return extract_state(rep)


def raw_to_state_identity(raw_observation: Any) -> Tuple[str, str]:
    """
    Extract canonical (state_id, state_hash) directly from raw observation.
    """
    st = raw_to_state(raw_observation)
    return st.state_id, st.state_hash


class Mirror7Pipeline:
    """
    Integrated stateful runtime processing streams of raw observations directly
    into representations, states, and temporal transitions.
    """

    def __init__(self) -> None:
        self._tracker = TemporalStateTracker()

    @property
    def tracker(self) -> TemporalStateTracker:
        return self._tracker

    @property
    def current_state(self) -> Optional[StructuralState]:
        return self._tracker.current_state

    @property
    def step_count(self) -> int:
        return self._tracker.step_count

    def process_observation(self, raw_observation: Any) -> Tuple[DiscoveredRepresentation, StructuralState, StateTransition]:
        """
        Process a single raw observation end-to-end:
          raw -> representation -> state -> temporal transition
        """
        rep = discover_representation(raw_observation)
        state = extract_state(rep)
        transition = self._tracker.step(state)
        return rep, state, transition

    def process_sequence(self, raw_sequence: Sequence[Any]) -> List[Tuple[DiscoveredRepresentation, StructuralState, StateTransition]]:
        """
        Process a sequence of raw observations.
        """
        return [self.process_observation(obs) for obs in raw_sequence]

    def reset(self) -> None:
        """Reset internal temporal tracker."""
        self._tracker = TemporalStateTracker()
