from dataclasses import dataclass

@dataclass(frozen=True)
class GeneralizationReport:
    passed: bool
    scores: tuple[float,...]
    mean: float
    worst: float
    reason: str

class GeneralizationGate:
    """Promotes a skill only when performance is acceptable across unseen contexts."""
    def __init__(self, min_mean=.8, min_worst=.6): self.min_mean=min_mean; self.min_worst=min_worst
    def evaluate(self,scores):
        xs=tuple(float(x) for x in scores)
        if not xs: return GeneralizationReport(False,(),0.0,0.0,"no evaluation contexts")
        mean=sum(xs)/len(xs); worst=min(xs)
        ok=mean>=self.min_mean and worst>=self.min_worst
        return GeneralizationReport(ok,xs,mean,worst,"generalized" if ok else "overfit or weak transfer")