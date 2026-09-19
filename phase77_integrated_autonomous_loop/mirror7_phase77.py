"""Phase 77: integrated bounded autonomous learning loop.

raw observation -> representation -> regime check -> world update -> uncertainty
-> curriculum experiment -> action/observation -> correction.

This is a bounded numeric research mechanism, not a claim of general intelligence.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple, Any
import math


@dataclass(frozen=True)
class LoopResult:
    steps: int
    predictions: List[Tuple[float, ...]]
    experiments: List[Any]
    revisions: int
    stopped: bool
    reason: str


class IntegratedAutonomousLoop:
    def __init__(self, max_steps: int = 32, experiment_budget: int = 4, tolerance: float = 1e-6):
        if max_steps < 1 or experiment_budget < 0 or tolerance < 0:
            raise ValueError("invalid bounds")
        self.max_steps = max_steps
        self.experiment_budget = experiment_budget
        self.tolerance = tolerance
        self.transitions: Dict[Tuple[Tuple[float, ...], Any], Tuple[float, ...]] = {}
        self.history: List[Tuple[Tuple[float, ...], Any, Tuple[float, ...]]] = []
        self.revisions = 0
        self.experiments: List[Any] = []

    @staticmethod
    def _state(raw: Sequence[float]) -> Tuple[float, ...]:
        if raw is None or len(raw) == 0:
            raise ValueError("empty observation")
        vals = tuple(float(x) for x in raw)
        if not all(math.isfinite(x) for x in vals):
            raise ValueError("non-finite observation")
        return vals

    def observe(self, raw: Sequence[float]) -> Tuple[float, ...]:
        return self._state(raw)

    def learn(self, state: Sequence[float], action: Any, next_state: Sequence[float]) -> bool:
        s, ns = self._state(state), self._state(next_state)
        if len(s) != len(ns):
            return False
        key = (s, action)
        old = self.transitions.get(key)
        if old is not None and any(abs(a-b) > self.tolerance for a,b in zip(old, ns)):
            self.revisions += 1
        self.transitions[key] = ns
        self.history.append((s, action, ns))
        return True

    def predict(self, state: Sequence[float], action: Any):
        s = self._state(state)
        return self.transitions.get((s, action))

    def uncertainty(self, state: Sequence[float], actions: Sequence[Any]) -> float:
        s = self._state(state)
        if not actions:
            return 1.0
        known = sum((s, a) in self.transitions for a in actions)
        return 1.0 - known / float(len(actions))

    def choose_experiment(self, state: Sequence[float], actions: Sequence[Any]):
        s = self._state(state)
        unknown = [a for a in actions if (s, a) not in self.transitions]
        if not unknown:
            return None
        if len(self.experiments) >= self.experiment_budget:
            return None
        choice = unknown[0]
        self.experiments.append(choice)
        return choice

    def run(self, initial_raw: Sequence[float], actions: Sequence[Any],
            step_fn: Callable[[Tuple[float, ...], Any], Sequence[float]],
            stop_fn: Callable[[Tuple[float, ...]], bool] | None = None) -> LoopResult:
        if not callable(step_fn):
            raise ValueError("step_fn must be callable")
        if not actions:
            return LoopResult(0, [], [], self.revisions, True, "no legal actions")
        state = self.observe(initial_raw)
        predictions: List[Tuple[float, ...]] = []
        stopped = False
        reason = "step budget exhausted"
        for _ in range(self.max_steps):
            if stop_fn is not None and stop_fn(state):
                stopped, reason = True, "goal reached"
                break
            action = self.choose_experiment(state, actions)
            if action is None:
                known = [a for a in actions if self.predict(state, a) is not None]
                if not known:
                    stopped, reason = True, "uncertain and experiment budget exhausted"
                    break
                action = known[0]
            predicted = self.predict(state, action)
            observed = self.observe(step_fn(state, action))
            if len(observed) != len(state):
                stopped, reason = True, "dimension mismatch"
                break
            if predicted is not None and any(abs(a-b) > self.tolerance for a,b in zip(predicted, observed)):
                self.revisions += 1
            self.learn(state, action, observed)
            predictions.append(observed)
            state = observed
        return LoopResult(len(predictions), predictions, list(self.experiments), self.revisions, stopped, reason)
