from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True)
class GuardResult:
    accepted:bool
    primary_gain:float
    protected_regressions:tuple[str,...]

class RegressionGuard:
    """Requires primary improvement while preventing material regression in protected capabilities."""
    def __init__(self,tolerance:float=0.0,min_primary_gain:float=.01):
        if tolerance<0 or min_primary_gain<0: raise ValueError("guard parameters must be non-negative")
        self.tolerance=tolerance; self.min_primary_gain=min_primary_gain
    def evaluate(self,baseline:Mapping[str,float],candidate:Mapping[str,float],primary:str)->GuardResult:
        if primary not in baseline or primary not in candidate: raise KeyError(primary)
        gain=float(candidate[primary])-float(baseline[primary])
        regressions=tuple(sorted(k for k in baseline if k!=primary and k in candidate and candidate[k]<baseline[k]-self.tolerance))
        return GuardResult(gain>=self.min_primary_gain and not regressions,gain,regressions)
