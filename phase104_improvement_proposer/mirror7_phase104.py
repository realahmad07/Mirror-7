from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class Proposal:
    target: str
    change: str
    expected_gain: float
    risk: float
    cost: float
    rationale: str

class ImprovementProposer:
    """Converts recurring verified failures into bounded improvement proposals; does not self-modify."""
    def propose(self,target:str, failures:Sequence[str], baseline:float, expected_gain:float, risk:float, cost:float)->Proposal|None:
        if not target or not failures: return None
        if not (0<=risk<=1 and cost>=0 and 0<=baseline<=1): raise ValueError("invalid metrics")
        gain=max(0.0,float(expected_gain))
        if gain <= 0 or gain <= risk: return None
        return Proposal(target, f"revise {target} to address: " + "; ".join(failures), gain,risk,cost,
                        f"Repeated failures={len(failures)}; baseline={baseline:.3f}")
