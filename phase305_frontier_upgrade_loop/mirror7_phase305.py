from dataclasses import dataclass
from typing import Callable, Dict, Mapping

from phase299_capability_frontier import CapabilityFrontier
from phase302_improvement_history import ImprovementHistory
from phase303_cross_capability_guard import RegressionGuard
from phase304_frontier_scheduler import FrontierScheduler

@dataclass(frozen=True)
class UpgradeOutcome:
    score:float
    changed:bool
    variant:str
    protected_scores:Mapping[str,float]

class FrontierUpgradeLoop:
    """Continuously attacks the current frontier gap while preserving protected capabilities."""
    def __init__(self,frontier:CapabilityFrontier,adapters:Mapping[str,object],guard:RegressionGuard|None=None):
        self.frontier=frontier
        self.adapters=dict(adapters)
        self.guard=guard or RegressionGuard()
        self.history=ImprovementHistory()
        self.scheduler=FrontierScheduler(frontier)
        self.scores={t.name:t.baseline for t in frontier.targets}
        missing=[name for name in (t.name for t in frontier.targets) if name not in self.adapters]
        if missing: raise ValueError("missing adapters: "+",".join(missing))

    def step(self,max_rounds:int=4,max_candidates:int=8):
        scheduled=self.scheduler.next(max_rounds)
        if scheduled is None: return None
        adapter=self.adapters[scheduled.capability]
        outcome=adapter.improve(scheduled.seed,scheduled.campaign_rounds,max_candidates)
        baseline=dict(self.scores)
        candidate=dict(baseline)
        candidate[scheduled.capability]=float(outcome.score)
        candidate.update({k:float(v) for k,v in outcome.protected_scores.items()})
        guard=self.guard.evaluate(baseline,candidate,scheduled.capability)
        accepted=bool(outcome.changed and guard.accepted)
        if accepted:
            self.scores.update(candidate)
            self.frontier.update(scheduled.capability,outcome.score)
        self.history.record(scheduled.capability,baseline[scheduled.capability],float(outcome.score),accepted,outcome.variant,scheduled.campaign_rounds)
        return {
            "capability":scheduled.capability,
            "seed":scheduled.seed,
            "accepted":accepted,
            "score":float(outcome.score),
            "gap":self.frontier.top_gap().gap if self.frontier.top_gap() else 0.0,
            "variant":outcome.variant,
            "protected_regressions":guard.protected_regressions,
        }

    def run(self,steps:int=8,max_rounds:int=4,max_candidates:int=8):
        if steps<1: raise ValueError("steps must be positive")
        out=[]
        for _ in range(steps):
            step=self.step(max_rounds,max_candidates)
            if step is None: break
            out.append(step)
        return out
