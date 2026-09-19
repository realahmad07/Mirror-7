from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Sequence

@dataclass(frozen=True)
class TemporalEvent:
    event_id: int
    start: int
    end: int
    signature: tuple[float, ...]
    support: int

class TemporalAbstraction:
    """Bounded variable-length event discovery from numeric streams."""
    def __init__(self, window: int = 3, change_threshold: float = 1.0, max_events: int = 128):
        if window < 1 or change_threshold < 0 or max_events < 1:
            raise ValueError("invalid bounds")
        self.window = window
        self.change_threshold = float(change_threshold)
        self.max_events = max_events
        self.vocabulary: dict[tuple[float, ...], int] = {}
        self.support: dict[tuple[float, ...], int] = {}
        self.events: list[TemporalEvent] = []

    @staticmethod
    def _norm(stream: Sequence[float]) -> tuple[float, ...]:
        x = tuple(float(v) for v in stream)
        if not x or not all(isfinite(v) for v in x):
            raise ValueError("invalid stream")
        return x

    def _signature(self, chunk: Sequence[float]) -> tuple[float, ...]:
        mean = sum(chunk) / len(chunk)
        span = max(chunk) - min(chunk)
        slope = (chunk[-1] - chunk[0]) / max(1, len(chunk) - 1)
        return (round(mean, 6), round(span, 6), round(slope, 6), float(len(chunk)))

    def discover(self, stream: Sequence[float]) -> tuple[TemporalEvent, ...]:
        x = self._norm(stream)
        cuts = [0]
        for i in range(self.window, len(x), self.window):
            left = x[i-self.window:i]
            right = x[i:min(len(x), i+self.window)]
            if abs(sum(right)/len(right) - sum(left)/len(left)) >= self.change_threshold:
                cuts.append(i)
        if cuts[-1] != len(x):
            cuts.append(len(x))
        out: list[TemporalEvent] = []
        for start, end in zip(cuts, cuts[1:]):
            if end <= start:
                continue
            sig = self._signature(x[start:end])
            if sig not in self.vocabulary:
                if len(self.vocabulary) >= self.max_events:
                    continue
                self.vocabulary[sig] = len(self.vocabulary)
            self.support[sig] = self.support.get(sig, 0) + 1
            out.append(TemporalEvent(self.vocabulary[sig], start, end, sig, self.support[sig]))
        self.events = out[-self.max_events:]
        return tuple(out)

    def stable_events(self, min_support: int = 2) -> tuple[TemporalEvent, ...]:
        if min_support < 1:
            raise ValueError("min_support must be positive")
        return tuple(e for e in self.events if self.support.get(e.signature, 0) >= min_support)

    def encode(self, stream: Sequence[float]) -> tuple[int, ...]:
        return tuple(e.event_id for e in self.discover(stream))

    def fail_closed(self) -> dict[str, int]:
        return {"vocabulary": len(self.vocabulary), "events": len(self.events)}
