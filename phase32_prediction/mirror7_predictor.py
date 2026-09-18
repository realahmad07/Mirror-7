"""Mirror 7 Phase 32: prediction, discrepancy, and online learning."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from .mirror7_transition import PredictionResult, PredictionStatus, TransitionMemory, _state_id

@dataclass(frozen=True)
class PredictionDiscrepancy:
    from_state_id: str
    predicted_state_id: Optional[str]
    actual_state_id: str
    status: PredictionStatus
    @property
    def correct(self) -> bool:
        return self.status == PredictionStatus.KNOWN and self.predicted_state_id == self.actual_state_id
    @property
    def error(self) -> bool:
        return not self.correct

class Predictor:
    def __init__(self, memory: Optional[TransitionMemory] = None) -> None:
        self.memory = memory if memory is not None else TransitionMemory()

    def learn_transition(self, previous_state: Any, next_state: Any):
        return self.memory.observe(previous_state, next_state)

    def predict_next(self, state: Any) -> PredictionResult:
        return self.memory.predict(state)

    def compare(self, state: Any, actual_next_state: Any, prediction: Optional[PredictionResult] = None) -> PredictionDiscrepancy:
        pred = prediction if prediction is not None else self.predict_next(state)
        return PredictionDiscrepancy(_state_id(state), pred.predicted_state_id, _state_id(actual_next_state), pred.status)

    def observe_and_correct(self, state: Any, actual_next_state: Any) -> PredictionDiscrepancy:
        prediction = self.predict_next(state)
        discrepancy = self.compare(state, actual_next_state, prediction)
        self.learn_transition(state, actual_next_state)
        return discrepancy

    def train_sequence(self, states: Tuple[Any, ...]) -> None:
        for previous, current in zip(states, states[1:]):
            self.learn_transition(previous, current)

    def rollout_ids(self, initial_state: Any, steps: int) -> Tuple[str, ...]:
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        current = _state_id(initial_state)
        result = [current]
        for _ in range(steps):
            prediction = self.memory.predict(current)
            if prediction.status != PredictionStatus.KNOWN:
                break
            current = prediction.predicted_state_id
            result.append(current)
        return tuple(result)
