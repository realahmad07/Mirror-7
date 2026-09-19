from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Sequence

@dataclass(frozen=True)
class LatentState:
    state_id: int
    centroid: tuple[float | None, ...]
    support: int
    confidence: float

class HiddenStateInferer:
    """Unlabeled online prototype discovery with partial observations."""
    def __init__(self, max_states: int = 16, merge_threshold: float = 1.0):
        if max_states < 1 or merge_threshold < 0:
            raise ValueError("invalid bounds")
        self.max_states = max_states
        self.merge_threshold = float(merge_threshold)
        self.states: list[LatentState] = []

    @staticmethod
    def _obs(obs: Sequence[float | None]) -> tuple[float | None, ...]:
        if not obs:
            raise ValueError("empty observation")
        x = tuple(obs)
        if any(v is not None and (not isinstance(v, (int, float)) or not isfinite(float(v))) for v in x):
            raise ValueError("invalid observation")
        return tuple(None if v is None else float(v) for v in x)

    def _distance(self, a, b) -> tuple[float, int]:
        if len(a) != len(b):
            return float("inf"), 0
        vals = [abs(x-y) for x, y in zip(a, b) if x is not None and y is not None]
        return (sum(vals) / len(vals), len(vals)) if vals else (float("inf"), 0)

    def ingest(self, obs: Sequence[float | None]) -> LatentState:
        x = self._obs(obs)
        best = None
        for s in self.states:
            d, n = self._distance(x, s.centroid)
            if n and (best is None or d < best[0]):
                best = (d, s)
        if best is None or best[0] > self.merge_threshold:
            if len(self.states) >= self.max_states:
                raise RuntimeError("latent-state budget exhausted")
            state = LatentState(len(self.states), x, 1, 1.0)
            self.states.append(state)
            return state
        d, s = best
        centroid = tuple(
            None if x_i is None and c_i is None
            else x_i if c_i is None
            else c_i if x_i is None
            else c_i + (x_i-c_i)/(s.support+1)
            for x_i, c_i in zip(x, s.centroid)
        )
        updated = LatentState(s.state_id, centroid, s.support + 1, 1.0/(1.0+d))
        self.states[s.state_id] = updated
        return updated

    def infer(self, obs: Sequence[float | None]) -> LatentState | None:
        x = self._obs(obs)
        ranked = []
        for s in self.states:
            d, n = self._distance(x, s.centroid)
            if n:
                ranked.append((d, s))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1].state_id))
        d, s = ranked[0]
        return s if d <= self.merge_threshold else None

    def fail_closed(self) -> dict[str, int]:
        return {"states": len(self.states)}
