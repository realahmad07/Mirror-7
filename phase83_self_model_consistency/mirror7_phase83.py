"""Phase 83: bounded self-model consistency checks."""
from dataclasses import dataclass
from typing import Sequence
import math

@dataclass(frozen=True)
class ConsistencyReport:
    consistent: bool
    expected: tuple[float,...]
    actual: tuple[float,...]
    discrepancy: float

class SelfModelConsistency:
    def __init__(self,tolerance=1e-6):
        if tolerance<0: raise ValueError("invalid tolerance")
        self.tolerance=float(tolerance)
    def evaluate(self, expected: Sequence[float], actual: Sequence[float]) -> ConsistencyReport:
        e=tuple(float(x) for x in expected); a=tuple(float(x) for x in actual)
        if not e or len(e)!=len(a) or not all(math.isfinite(x) for x in e+a): raise ValueError("invalid state")
        d=max(abs(x-y) for x,y in zip(e,a))
        return ConsistencyReport(d<=self.tolerance,e,a,d)
