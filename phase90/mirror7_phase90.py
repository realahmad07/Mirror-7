from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Hashable, Sequence

@dataclass
class _Stat:
    n: int = 0
    mean: float = 0.0
    def add(self, x: float) -> None:
        self.n += 1
        self.mean += (x-self.mean)/self.n

@dataclass(frozen=True)
class Effect:
    action: Hashable
    lag: int
    dimension: int
    mean: float
    samples: int

class OverlappingDelayedEffects:
    """Additive sparse delayed-effect model for concurrently pending actions."""
    def __init__(self, max_lag: int = 8, min_samples: int = 3):
        if max_lag < 1 or min_samples < 2:
            raise ValueError("invalid bounds")
        self.max_lag = max_lag
        self.min_samples = min_samples
        self.effects: dict[tuple[Hashable,int,int], _Stat] = defaultdict(_Stat)

    def learn(self, action: Hashable, lag: int, dimension: int, delta: float) -> None:
        if not isinstance(lag, int) or lag < 1 or lag > self.max_lag or dimension < 0:
            raise ValueError("invalid effect index")
        x = float(delta)
        if not isfinite(x):
            raise ValueError("invalid effect")
        self.effects[(action, lag, dimension)].add(x)

    def estimate(self, action: Hashable, lag: int, dimension: int) -> Effect | None:
        s = self.effects.get((action, lag, dimension))
        if s is None or s.n < self.min_samples:
            return None
        return Effect(action, lag, dimension, s.mean, s.n)

    def predict(self, base: Sequence[float], pending: Sequence[tuple[Hashable,int]]) -> tuple[float, ...] | None:
        x = tuple(float(v) for v in base)
        if not x or not all(isfinite(v) for v in x):
            raise ValueError("invalid base state")
        out = list(x)
        found = False
        for action, age in pending:
            if age < 1 or age > self.max_lag:
                raise ValueError("pending age outside model bounds")
            for dim in range(len(out)):
                e = self.estimate(action, age, dim)
                if e is not None:
                    out[dim] += e.mean
                    found = True
        return tuple(out) if found else None

    def signature(self) -> dict[tuple, float]:
        return {k:s.mean for k,s in self.effects.items() if s.n >= self.min_samples}
