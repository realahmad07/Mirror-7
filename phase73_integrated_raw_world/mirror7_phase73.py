from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence

from phase70_unified_world_model.mirror7_phase70 import UnifiedWorldModel, WorldPrediction
from phase71_raw_representation.mirror7_phase71 import RawRepresentationLearner


@dataclass(frozen=True)
class RawWorldPrediction:
    raw_state: tuple
    structured_state: tuple
    uncertainty: float
    evidence: int


class IntegratedRawWorldModel:
    """Bounded raw-stream -> representation -> world-model closed loop.

    The representation is schema-free but deliberately limited to numeric streams.
    A fixed number of segment slots keeps the world-model dimensionality stable while
    allowing the representation learner to discover boundaries and local signatures.
    """
    def __init__(self, window=3, change_threshold=2.0, min_support=2,
                 max_segments=4, min_samples=3, max_depth=5, beam_width=8):
        if max_segments < 1:
            raise ValueError("max_segments must be positive")
        self.representation = RawRepresentationLearner(window, change_threshold, min_support)
        self.world = UnifiedWorldModel(min_samples, max_depth, beam_width)
        self.max_segments = max_segments
        self._observations = 0

    def represent(self, raw: Iterable[float]) -> tuple:
        segments = self.representation.discover(tuple(raw))
        state = []
        for seg in segments[:self.max_segments]:
            mean, span, length = seg.signature
            state.extend((float(mean), float(span), float(length)))
        while len(state) < self.max_segments * 3:
            state.append(None)
        return tuple(state)

    def observe(self, action, before_raw: Iterable[float], after_raw: Iterable[float]):
        before = self.represent(before_raw)
        after = self.represent(after_raw)
        if len(before) != len(after):
            raise ValueError("representation dimension mismatch")
        self.world.observe(action, before, after)
        self._observations += 1
        return {"action": action, "before": before, "after": after,
                "observations": self._observations}

    def predict(self, action, raw: Iterable[float]):
        state = self.represent(raw)
        result = self.world.predict(action, state)
        if result is None:
            return None
        return RawWorldPrediction(tuple(result.state), state, result.uncertainty, result.evidence)

    def counterfactual(self, raw: Iterable[float], actions: Sequence[object]):
        state = self.represent(raw)
        result = self.world.counterfactual(state, actions)
        if result is None:
            return None
        return RawWorldPrediction(tuple(result.state), state, result.uncertainty, result.evidence)

    def plan(self, raw: Iterable[float], actions: Sequence[object], target_raw: Iterable[float]):
        state = self.represent(raw)
        target = self.represent(target_raw)
        result = self.world.plan(state, actions, target)
        if result is None:
            return None
        return RawWorldPrediction(tuple(result.state), state, result.uncertainty, result.evidence)

    def fail_closed(self):
        return {"observations": self._observations,
                "representation": self.representation.fail_closed(),
                "world": self.world.fail_closed()}
