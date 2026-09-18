"""
Mirror 7 — Phase 32.1 & 32.2: Transition Representation and Transition Memory

Provides an explicit, inspectable, deterministic representation of observed state transitions
and an in-memory transition model that tracks observation frequencies and support.
Strictly non-opaque; no machine learning weights or embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, List, Optional, Set, Tuple


def _extract_state_id(state: Any) -> str:
    if state is None:
        raise TypeError("state cannot be None")
    if hasattr(state, "state_id") and isinstance(getattr(state, "state_id"), str):
        sid = getattr(state, "state_id")
        if not sid:
            raise ValueError("state_id cannot be empty")
        return sid
    if isinstance(state, str):
        if not state.strip():
            raise ValueError("state string ID cannot be empty or whitespace")
        return state.strip()
    raise TypeError(f"Expected StructuralState or str state_id, got {type(state).__name__}")


def _compute_transition_hash(from_id: str, to_id: str, count: int, support: float, last_step: int) -> str:
    hasher = hashlib.sha256()
    hasher.update(
        f"FROM:{from_id}|TO:{to_id}|COUNT:{count}|SUPPORT:{support:.8f}|STEP:{last_step}|".encode("ascii")
    )
    return hasher.hexdigest()


@dataclass(frozen=True)
class ObservedTransition:
    from_state_id: str
    to_state_id: str
    observation_count: int
    support: float
    last_observed_step: int
    transition_hash: str

    def __post_init__(self) -> None:
        if not self.from_state_id or not self.to_state_id:
            raise ValueError("state IDs cannot be empty")
        if self.observation_count < 1:
            raise ValueError("observation_count must be >= 1")
        if not (0.0 <= self.support <= 1.000001):
            raise ValueError("support must be in [0.0, 1.0]")
        if self.last_observed_step < 0:
            raise ValueError("last_observed_step must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_state_id": self.from_state_id,
            "to_state_id": self.to_state_id,
            "observation_count": self.observation_count,
            "support": round(self.support, 8),
            "last_observed_step": self.last_observed_step,
            "transition_hash": self.transition_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObservedTransition":
        return cls(
            from_state_id=data["from_state_id"],
            to_state_id=data["to_state_id"],
            observation_count=data["observation_count"],
            support=float(data["support"]),
            last_observed_step=data["last_observed_step"],
            transition_hash=data["transition_hash"],
        )


class TransitionMemory:
    """Deterministic, serializable, inspectable state-transition memory."""

    def __init__(self) -> None:
        self._transitions: Dict[Tuple[str, str], List[int]] = {}
        self._outgoing_totals: Dict[str, int] = {}
        self._seen_states: Set[str] = set()
        self._total_observations = 0
        self._current_step = 0

    @property
    def transition_count(self) -> int:
        return len(self._transitions)

    @property
    def total_observations(self) -> int:
        return self._total_observations

    @property
    def state_count(self) -> int:
        return len(self._seen_states)

    @property
    def known_states(self) -> Tuple[str, ...]:
        return tuple(sorted(self._seen_states))

    def has_state(self, state: Any) -> bool:
        try:
            return _extract_state_id(state) in self._seen_states
        except (TypeError, ValueError):
            return False

    def is_known_transition(self, from_state: Any, to_state: Any) -> bool:
        try:
            return (_extract_state_id(from_state), _extract_state_id(to_state)) in self._transitions
        except (TypeError, ValueError):
            return False

    def record_transition(self, from_state: Any, to_state: Any, step: Optional[int] = None) -> ObservedTransition:
        from_id, to_id = _extract_state_id(from_state), _extract_state_id(to_state)
        if step is not None:
            if not isinstance(step, int) or isinstance(step, bool) or step < 0:
                raise ValueError("step must be a non-negative integer")
            current_step = step
            self._current_step = max(self._current_step, step + 1)
        else:
            current_step = self._current_step
            self._current_step += 1

        key = (from_id, to_id)
        if key in self._transitions:
            self._transitions[key][0] += 1
            self._transitions[key][1] = current_step
        else:
            self._transitions[key] = [1, current_step]

        self._outgoing_totals[from_id] = self._outgoing_totals.get(from_id, 0) + 1
        self._seen_states.update((from_id, to_id))
        self._total_observations += 1
        return self.get_transition(from_id, to_id)  # type: ignore

    def get_transition(self, from_state: Any, to_state: Any) -> Optional[ObservedTransition]:
        from_id, to_id = _extract_state_id(from_state), _extract_state_id(to_state)
        key = (from_id, to_id)
        if key not in self._transitions:
            return None
        count, last_step = self._transitions[key]
        total = self._outgoing_totals.get(from_id, count)
        support = count / total if total else 1.0
        return ObservedTransition(from_id, to_id, count, support, last_step,
                                  _compute_transition_hash(from_id, to_id, count, support, last_step))

    def get_successors(self, from_state: Any) -> Tuple[ObservedTransition, ...]:
        from_id = _extract_state_id(from_state)
        total = self._outgoing_totals.get(from_id, 0)
        if not total:
            return ()
        results = []
        for (fid, tid), (count, last_step) in self._transitions.items():
            if fid == from_id:
                support = count / total
                results.append(ObservedTransition(
                    fid, tid, count, support, last_step,
                    _compute_transition_hash(fid, tid, count, support, last_step)))
        results.sort(key=lambda t: (-t.observation_count, t.to_state_id))
        return tuple(results)

    def all_transitions(self) -> Tuple[ObservedTransition, ...]:
        return tuple(self.get_transition(fid, tid) for fid, tid in sorted(self._transitions))  # type: ignore

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": "1.0",
            "total_observations": self._total_observations,
            "current_step": self._current_step,
            "seen_states": sorted(self._seen_states),
            "transitions": [t.to_dict() for t in self.all_transitions()],
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransitionMemory":
        mem = cls()
        mem._total_observations = data.get("total_observations", 0)
        mem._current_step = data.get("current_step", 0)
        mem._seen_states = set(data.get("seen_states", []))
        for t in data.get("transitions", []):
            fid, tid = t["from_state_id"], t["to_state_id"]
            count, last_step = t["observation_count"], t["last_observed_step"]
            mem._transitions[(fid, tid)] = [count, last_step]
            mem._outgoing_totals[fid] = mem._outgoing_totals.get(fid, 0) + count
        return mem

    @classmethod
    def from_json(cls, json_str: str) -> "TransitionMemory":
        return cls.from_dict(json.loads(json_str))

    def clear(self) -> None:
        self._transitions.clear()
        self._outgoing_totals.clear()
        self._seen_states.clear()
        self._total_observations = 0
        self._current_step = 0
