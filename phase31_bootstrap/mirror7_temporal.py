"""
Mirror 7 — Phase 31.3: Temporal State Identity

Tracks and identifies structural states across discrete time steps.
Distinguishes:
- Unchanged state (S_t == S_{t-1})
- Novel state transition (S_t is newly observed)
- Repeated state (consecutive identical observations)
- Return to previous state (S_t was observed at t' < t-1, departed, and now returned)
- Temporal transition sequence

Timestamps are NOT used for semantic identity; identity is purely derived from
the structural state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .mirror7_state import StructuralState, extract_state
    from .mirror7_representation import DiscoveredRepresentation, discover_representation
except (ImportError, ValueError):
    from mirror7_state import StructuralState, extract_state
    from mirror7_representation import DiscoveredRepresentation, discover_representation


class TemporalRelation(str, Enum):
    INITIAL = "INITIAL"
    UNCHANGED = "UNCHANGED"
    TRANSITION_NOVEL = "TRANSITION_NOVEL"
    TRANSITION_RETURN = "TRANSITION_RETURN"


@dataclass(frozen=True)
class StateTransition:
    """
    Immutable record of a discrete step transition between structural states.
    """
    step: int
    from_state_id: Optional[str]
    to_state_id: str
    relation: TemporalRelation
    transition_hash: str

    def __repr__(self) -> str:
        return (
            f"StateTransition(step={self.step}, {self.from_state_id} -> {self.to_state_id}, "
            f"rel={self.relation.value})"
        )


def _compute_transition_hash(step: int, from_id: Optional[str], to_id: str, relation: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"STEP:{step}|FROM:{from_id or 'NONE'}|TO:{to_id}|REL:{relation}|".encode("ascii"))
    return hasher.hexdigest()


class TemporalStateTracker:
    """
    Deterministic tracker for identifying structural state identity and transitions across time.
    Does NOT rely on wall-clock timestamps.
    """

    def __init__(self) -> None:
        self._history: List[StructuralState] = []
        self._transitions: List[StateTransition] = []
        self._seen_state_ids: Dict[str, List[int]] = {}

    @property
    def history(self) -> Tuple[StructuralState, ...]:
        return tuple(self._history)

    @property
    def transitions(self) -> Tuple[StateTransition, ...]:
        return tuple(self._transitions)

    @property
    def current_state(self) -> Optional[StructuralState]:
        return self._history[-1] if self._history else None

    @property
    def step_count(self) -> int:
        return len(self._history)

    def step(self, state: Any) -> StateTransition:
        """
        Record a state observation at the next discrete time step.
        """
        if not isinstance(state, StructuralState):
            raise TypeError(f"step requires a StructuralState instance, got {type(state).__name__}")

        step_idx = len(self._history)
        to_id = state.state_id

        if step_idx == 0:
            relation = TemporalRelation.INITIAL
            from_id = None
        else:
            prev_state = self._history[-1]
            from_id = prev_state.state_id
            if state.state_hash == prev_state.state_hash:
                relation = TemporalRelation.UNCHANGED
            elif to_id in self._seen_state_ids:
                relation = TemporalRelation.TRANSITION_RETURN
            else:
                relation = TemporalRelation.TRANSITION_NOVEL

        thash = _compute_transition_hash(step_idx, from_id, to_id, relation.value)
        trans = StateTransition(
            step=step_idx,
            from_state_id=from_id,
            to_state_id=to_id,
            relation=relation,
            transition_hash=thash,
        )

        self._history.append(state)
        self._transitions.append(trans)
        if to_id not in self._seen_state_ids:
            self._seen_state_ids[to_id] = []
        self._seen_state_ids[to_id].append(step_idx)

        return trans

    def is_repeated_state(self, state_id: str) -> bool:
        """Returns True if the given state has been visited more than once."""
        return len(self._seen_state_ids.get(state_id, [])) > 1

    def has_returned_to(self, state_id: str) -> bool:
        """Returns True if the system was in this state, moved away, and later returned."""
        indices = self._seen_state_ids.get(state_id, [])
        if len(indices) < 2:
            return False
        # Check if there is any gap > 1 between occurrences
        for i in range(len(indices) - 1):
            if indices[i + 1] - indices[i] > 1:
                return True
        return False

    def temporal_signature(self) -> str:
        """
        Deterministic checksum characterizing the entire temporal trajectory.
        """
        hasher = hashlib.sha256()
        for t in self._transitions:
            hasher.update(f"{t.step}:{t.from_state_id}->{t.to_state_id}:{t.relation.value}|".encode("ascii"))
        return hasher.hexdigest()

    @classmethod
    def replay(cls, states: Sequence[StructuralState]) -> "TemporalStateTracker":
        """
        Deterministically reconstruct tracker trajectory from a sequence of states.
        """
        tracker = cls()
        for s in states:
            tracker.step(s)
        return tracker
