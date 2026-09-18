"""
Mirror 7 — Phase 31.5: Raw -> State Integration Pipeline
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
    rep = discover_representation(raw_observation)
    return extract_state(rep)


def raw_to_state_identity(raw_observation: Any) -> Tuple[str, str]:
    st = raw_to_state(raw_observation)
    return st.state_id, st.state_hash


class Mirror7Pipeline:
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

    def process_observation(
        self, raw_observation: Any
    ) -> Tuple[DiscoveredRepresentation, StructuralState, StateTransition]:
        rep = discover_representation(raw_observation)
        state = extract_state(rep)
        transition = self._tracker.step(state)
        return rep, state, transition

    def process_sequence(
        self, raw_sequence: Sequence[Any]
    ) -> List[Tuple[DiscoveredRepresentation, StructuralState, StateTransition]]:
        return [self.process_observation(obs) for obs in raw_sequence]

    def reset(self) -> None:
        self._tracker = TemporalStateTracker()
