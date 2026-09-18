"""Mirror 7 Phase 32: explicit state-transition memory."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Dict, Optional, Tuple

class PredictionStatus(str, Enum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    AMBIGUOUS = "AMBIGUOUS"

@dataclass(frozen=True)
class TransitionObservation:
    from_state_id: str
    to_state_id: str
    support: int

@dataclass(frozen=True)
class PredictionResult:
    status: PredictionStatus
    from_state_id: str
    predicted_state_id: Optional[str]
    candidates: Tuple[TransitionObservation, ...]

class TransitionMemory:
    """Deterministic first-order transition model keyed by Phase 31 state identity."""
    def __init__(self) -> None:
        self._counts: Dict[str, Dict[str, int]] = {}

    @property
    def transition_count(self) -> int:
        return sum(len(v) for v in self._counts.values())

    @property
    def observation_count(self) -> int:
        return sum(sum(v.values()) for v in self._counts.values())

    def observe(self, from_state: Any, to_state: Any) -> TransitionObservation:
        return self.observe_ids(_state_id(from_state), _state_id(to_state))

    def observe_ids(self, from_state_id: str, to_state_id: str) -> TransitionObservation:
        if not isinstance(from_state_id, str) or not from_state_id:
            raise TypeError("from_state_id must be a non-empty string")
        if not isinstance(to_state_id, str) or not to_state_id:
            raise TypeError("to_state_id must be a non-empty string")
        targets = self._counts.setdefault(from_state_id, {})
        targets[to_state_id] = targets.get(to_state_id, 0) + 1
        return TransitionObservation(from_state_id, to_state_id, targets[to_state_id])

    def successors(self, state: Any) -> Tuple[TransitionObservation, ...]:
        return self.successors_id(_state_id(state))

    def successors_id(self, state_id: str) -> Tuple[TransitionObservation, ...]:
        if not isinstance(state_id, str) or not state_id:
            raise TypeError("state_id must be a non-empty string")
        return tuple(TransitionObservation(state_id, target, support)
                     for target, support in sorted(self._counts.get(state_id, {}).items()))

    def predict(self, state: Any) -> PredictionResult:
        state_id = _state_id(state)
        candidates = self.successors_id(state_id)
        if not candidates:
            return PredictionResult(PredictionStatus.UNKNOWN, state_id, None, ())
        if len(candidates) == 1:
            return PredictionResult(PredictionStatus.KNOWN, state_id, candidates[0].to_state_id, candidates)
        return PredictionResult(PredictionStatus.AMBIGUOUS, state_id, None, candidates)

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        return {k: dict(v) for k, v in sorted(self._counts.items())}

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, int]]) -> "TransitionMemory":
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")
        m = cls()
        for src, targets in data.items():
            if not isinstance(src, str) or not isinstance(targets, dict):
                raise TypeError("invalid serialized transition model")
            for dst, count in targets.items():
                if not isinstance(dst, str) or not isinstance(count, int) or count < 1:
                    raise ValueError("invalid transition support")
                m._counts.setdefault(src, {})[dst] = count
        return m

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> "TransitionMemory":
        return cls.from_dict(json.loads(payload))

def _state_id(state: Any) -> str:
    if isinstance(state, str):
        if not state:
            raise TypeError("state id cannot be empty")
        return state
    state_id = getattr(state, "state_id", None)
    if not isinstance(state_id, str) or not state_id:
        raise TypeError("state must be a Phase 31 StructuralState or non-empty state id")
    return state_id
