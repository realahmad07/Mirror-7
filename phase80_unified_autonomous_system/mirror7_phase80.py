"""Phase 80: bounded integration of core autonomous mechanisms."""
from dataclasses import dataclass
from typing import Any, Callable, Sequence
import math

@dataclass(frozen=True)
class StepReport:
    state: tuple[float, ...]
    predicted: tuple[float, ...] | None
    observed: tuple[float, ...]
    discrepancy: float
    revised: bool
    selected_action: Any

class UnifiedAutonomousSystem:
    def __init__(self, tolerance: float = 1e-6, max_memory: int = 256, max_revisions: int = 16):
        if tolerance < 0 or max_memory < 1 or max_revisions < 0: raise ValueError("invalid bounds")
        self.tolerance=tolerance; self.max_memory=max_memory; self.max_revisions=max_revisions
        self.transitions={}; self.history=[]; self.revisions=0

    @staticmethod
    def _state(x: Sequence[float]) -> tuple[float,...]:
        y=tuple(float(v) for v in x)
        if not y or not all(math.isfinite(v) for v in y): raise ValueError("invalid state")
        return y

    def _remember(self, key, value):
        self.transitions[key]=value
        if len(self.transitions)>self.max_memory: del self.transitions[next(iter(self.transitions))]

    def predict(self, state: Sequence[float], action: Any):
        return self.transitions.get((self._state(state), action))

    def step(self, state: Sequence[float], actions: Sequence[Any], observe_fn: Callable[[tuple[float,...],Any],Sequence[float]], score_fn: Callable | None = None) -> StepReport:
        s=self._state(state)
        if not actions: raise ValueError("no legal actions")
        candidates=[]
        for a in actions:
            pred=self.predict(s,a); score=0.0 if pred is None else 1.0
            if score_fn is not None:
                score=float(score_fn(s,a,pred if pred is not None else s))
                if not math.isfinite(score): raise ValueError("invalid action score")
            candidates.append((score,a,pred))
        _, action, predicted=max(candidates,key=lambda x:x[0])
        observed=self._state(observe_fn(s,action))
        if predicted is not None and len(predicted)!=len(observed): raise ValueError("dimension mismatch")
        discrepancy=0.0 if predicted is None else max(abs(a-b) for a,b in zip(predicted,observed))
        revised=False
        if predicted is not None and discrepancy>self.tolerance:
            if self.revisions>=self.max_revisions: raise RuntimeError("revision budget exhausted")
            self.revisions+=1; revised=True
        self._remember((s,action),observed)
        report=StepReport(s,predicted,observed,discrepancy,revised,action)
        self.history.append(report)
        if len(self.history)>self.max_memory: self.history.pop(0)
        return report
