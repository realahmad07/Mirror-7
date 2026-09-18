"""
Mirror 7 — Phase 32.3 to 32.10: Prediction Engine
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple

try:
    from phase31_bootstrap.mirror7_representation import DiscoveredRepresentation
    from phase31_bootstrap.mirror7_state import StructuralState
    from phase31_bootstrap.mirror7_temporal import StateTransition
    from phase31_bootstrap.mirror7_pipeline import Mirror7Pipeline
except (ImportError, ValueError):
    try:
        from mirror7_representation import DiscoveredRepresentation
        from mirror7_state import StructuralState
        from mirror7_temporal import StateTransition
        from mirror7_pipeline import Mirror7Pipeline
    except (ImportError, ValueError):
        pass

from .mirror7_transition import ObservedTransition, TransitionMemory, _extract_state_id


class PredictionStatus(str, Enum):
    KNOWN = "KNOWN"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"


class DiscrepancyType(str, Enum):
    MATCH = "MATCH"
    MISMATCH_INCORRECT = "MISMATCH_INCORRECT"
    MISMATCH_AMBIGUOUS = "MISMATCH_AMBIGUOUS"
    MISMATCH_UNKNOWN = "MISMATCH_UNKNOWN"


@dataclass(frozen=True)
class CandidateSuccessor:
    state_id: str
    observation_count: int
    support: float


@dataclass(frozen=True)
class PredictionResult:
    from_state_id: str
    predicted_state_id: Optional[str]
    status: PredictionStatus
    confidence: float
    candidates: Tuple[CandidateSuccessor, ...]
    policy_applied: str


@dataclass(frozen=True)
class PredictionDiscrepancy:
    step: int
    from_state_id: str
    predicted_state_id: Optional[str]
    actual_state_id: str
    is_correct: bool
    discrepancy_type: DiscrepancyType
    error_description: str
    candidates: Tuple[CandidateSuccessor, ...]


class TransitionPredictor:
    def __init__(self, memory: Optional[TransitionMemory] = None) -> None:
        self._memory = memory if memory is not None else TransitionMemory()

    @property
    def memory(self) -> TransitionMemory:
        return self._memory

    def predict_next(self, state: Any, resolve_ambiguity: bool = False) -> PredictionResult:
        from_id = _extract_state_id(state)
        succs = self._memory.get_successors(from_id)
        if not succs:
            return PredictionResult(from_id, None, PredictionStatus.UNKNOWN, 0.0, (), "NONE")
        candidates = tuple(CandidateSuccessor(s.to_state_id, s.observation_count, s.support) for s in succs)
        if len(candidates) == 1:
            return PredictionResult(from_id, candidates[0].state_id, PredictionStatus.KNOWN, 1.0, candidates, "UNIQUE")
        if not resolve_ambiguity:
            return PredictionResult(from_id, None, PredictionStatus.AMBIGUOUS, 0.0, candidates, "NONE")
        chosen = candidates[0]
        return PredictionResult(from_id, chosen.state_id, PredictionStatus.AMBIGUOUS, chosen.support, candidates,
                                "MAX_SUPPORT_TIE_LEXICOGRAPHICAL")

    def evaluate_prediction(self, prediction: PredictionResult, actual_state: Any, step: int = 0) -> PredictionDiscrepancy:
        actual_id = _extract_state_id(actual_state)
        if prediction.status == PredictionStatus.UNKNOWN:
            return PredictionDiscrepancy(step, prediction.from_state_id, None, actual_id, False,
                                         DiscrepancyType.MISMATCH_UNKNOWN,
                                         f"State {prediction.from_state_id} has no known transitions; actual was {actual_id}",
                                         prediction.candidates)
        if prediction.predicted_state_id is None:
            return PredictionDiscrepancy(step, prediction.from_state_id, None, actual_id, False,
                                         DiscrepancyType.MISMATCH_AMBIGUOUS,
                                         f"Transition from {prediction.from_state_id} is ambiguous; actual was {actual_id}",
                                         prediction.candidates)
        if prediction.predicted_state_id == actual_id:
            return PredictionDiscrepancy(step, prediction.from_state_id, prediction.predicted_state_id, actual_id, True,
                                         DiscrepancyType.MATCH, f"Prediction matched actual state {actual_id}",
                                         prediction.candidates)
        dtype = DiscrepancyType.MISMATCH_AMBIGUOUS if prediction.status == PredictionStatus.AMBIGUOUS else DiscrepancyType.MISMATCH_INCORRECT
        return PredictionDiscrepancy(step, prediction.from_state_id, prediction.predicted_state_id, actual_id, False,
                                     dtype, f"Predicted {prediction.predicted_state_id} but actual was {actual_id}",
                                     prediction.candidates)

    def online_update(self, from_state: Any, actual_state: Any, step: Optional[int] = None) -> ObservedTransition:
        return self._memory.record_transition(from_state, actual_state, step=step)

    def predict_and_update(self, from_state: Any, actual_state: Any, step: Optional[int] = None,
                           resolve_ambiguity: bool = False):
        pred = self.predict_next(from_state, resolve_ambiguity)
        disc = self.evaluate_prediction(pred, actual_state, step or 0)
        updated = self.online_update(from_state, actual_state, step)
        return pred, disc, updated

    def predict_sequence(self, start_state: Any, horizon: int, resolve_ambiguity: bool = False) -> List[PredictionResult]:
        if not isinstance(horizon, int) or horizon < 0:
            raise ValueError("horizon must be a non-negative integer")
        if horizon == 0:
            return []
        results = []
        current = start_state
        for _ in range(horizon):
            pred = self.predict_next(current, resolve_ambiguity)
            results.append(pred)
            if pred.predicted_state_id is None:
                break
            current = pred.predicted_state_id
        return results


class PredictivePipeline:
    def __init__(self, predictor: Optional[TransitionPredictor] = None) -> None:
        self._pipeline = Mirror7Pipeline()
        self._predictor = predictor if predictor is not None else TransitionPredictor()
        self._last_state: Optional[StructuralState] = None
        self._last_prediction: Optional[PredictionResult] = None
        self._step_count = 0

    @property
    def pipeline(self) -> Mirror7Pipeline:
        return self._pipeline

    @property
    def predictor(self) -> TransitionPredictor:
        return self._predictor

    @property
    def step_count(self) -> int:
        return self._step_count

    def process_and_predict(self, raw_observation: Any, resolve_ambiguity: bool = False):
        rep, state, trans = self._pipeline.process_observation(raw_observation)
        discrepancy = None
        if self._last_state is not None:
            if self._last_prediction is not None:
                discrepancy = self._predictor.evaluate_prediction(self._last_prediction, state, self._step_count)
            self._predictor.online_update(self._last_state, state, self._step_count)
        prediction = self._predictor.predict_next(state, resolve_ambiguity)
        self._last_state, self._last_prediction = state, prediction
        self._step_count += 1
        return rep, state, trans, discrepancy, prediction

    def process_sequence(self, raw_sequence: Sequence[Any], resolve_ambiguity: bool = False):
        return [self.process_and_predict(raw, resolve_ambiguity) for raw in raw_sequence]

    def reset(self) -> None:
        self._pipeline.reset()
        self._last_state = None
        self._last_prediction = None
        self._step_count = 0
