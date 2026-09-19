from dataclasses import dataclass
from phase303_cross_capability_guard import RegressionGuard
from phase312_multicapability_frontier import build_default_frontier, build_default_adapters

@dataclass(frozen=True)
class CampaignEvent:
    capability:str
    accepted:bool
    score:float
    variant:str

class MultiCapabilityCampaign:
    """Runs concrete capability adapters through one shared frontier gate."""
    def __init__(self):
        self.frontier=build_default_frontier()
        self.adapters=build_default_adapters()
        self.scores={t.name:t.baseline for t in self.frontier.targets}
        self.guard=RegressionGuard()

    def step(self,seed:int=7,max_candidates:int=8)->CampaignEvent|None:
        target=self.frontier.top_gap()
        if target is None: return None
        result=self.adapters.get(target.name).improve(seed,4,max_candidates)
        baseline=dict(self.scores)
        candidate=dict(baseline)
        candidate[target.name]=result.score
        guard=self.guard.evaluate(baseline,candidate,target.name)
        accepted=bool(result.changed and guard.accepted)
        if accepted:
            self.scores[target.name]=result.score
            self.frontier.update(target.name,result.score)
        return CampaignEvent(target.name,accepted,result.score,result.variant)

    def run(self,steps:int=8)->list[CampaignEvent]:
        if steps<1: raise ValueError("steps must be positive")
        events=[]
        for i in range(steps):
            event=self.step(7+i)
            if event is None: break
            events.append(event)
        return events
