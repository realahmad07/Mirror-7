from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class GateReport:
    accepted: bool
    reason: str
    baseline: float
    candidate: float
    held_out: float
    regression_ok: bool
    resource_ok: bool

class SelfImprovementGate:
    """Adopts a change only when candidate evidence improves without regressions or budget violations."""
    def __init__(self,min_gain=.0,min_held_out=.0,max_cost=1e9):
        self.min_gain=min_gain; self.min_held_out=min_held_out; self.max_cost=max_cost

    def evaluate(self,baseline:float,candidate:float,held_out:float,regression:Sequence[bool],cost:float)->GateReport:
        ok=(candidate-baseline>=self.min_gain and held_out>=self.min_held_out and all(regression) and cost<=self.max_cost)
        if not regression: ok=False
        if candidate<0 or held_out<0: ok=False
        reason="accepted" if ok else "rejected: evidence/regression/resource gate failed"
        return GateReport(ok,reason,baseline,candidate,held_out,all(regression),cost<=self.max_cost)
