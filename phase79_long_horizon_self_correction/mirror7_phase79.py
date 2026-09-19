"""Phase 79: bounded long-horizon discrepancy-driven self-correction."""
from dataclasses import dataclass
from typing import Any, Callable, Sequence, Tuple
import math

@dataclass(frozen=True)
class CycleReport:
    cycles: int
    errors: int
    revisions: int
    converged: bool
    stopped: bool
    reason: str

class SelfCorrectingLoop:
    def __init__(self, max_cycles: int = 32, tolerance: float = 1e-6, max_revisions: int = 8):
        if max_cycles < 1 or tolerance < 0 or max_revisions < 0:
            raise ValueError("invalid bounds")
        self.max_cycles, self.tolerance, self.max_revisions = max_cycles, tolerance, max_revisions
        self.model = {}
        self.errors = 0
        self.revisions = 0

    @staticmethod
    def _v(x):
        y = tuple(float(z) for z in x)
        if not y or not all(math.isfinite(z) for z in y):
            raise ValueError("invalid state")
        return y

    def run(self, initial: Sequence[float], action: Any,
            observe_fn: Callable[[Tuple[float, ...], Any], Sequence[float]],
            stop_fn: Callable[[Tuple[float, ...]], bool] | None = None) -> CycleReport:
        state = self._v(initial)
        converged = stopped = False
        reason = "cycle budget exhausted"
        cycles = 0

        if stop_fn and stop_fn(state):
            return CycleReport(0, self.errors, self.revisions, True, True, "goal reached")

        for cycles in range(1, self.max_cycles + 1):
            try:
                observed = self._v(observe_fn(state, action))
            except StopIteration:
                stopped, reason = True, "observation exhausted"
                break

            key = (state, action)
            predicted = self.model.get(key)
            if predicted is not None and len(predicted) != len(observed):
                stopped, reason = True, "dimension mismatch"
                break
            if predicted is not None and any(abs(a-b) > self.tolerance for a, b in zip(predicted, observed)):
                self.errors += 1
                if self.revisions >= self.max_revisions:
                    stopped, reason = True, "revision budget exhausted"
                    break
                self.revisions += 1

            self.model[key] = observed
            state = observed

            if stop_fn and stop_fn(state):
                converged, stopped, reason = True, True, "goal reached"
                break

        return CycleReport(cycles, self.errors, self.revisions, converged, stopped, reason)
