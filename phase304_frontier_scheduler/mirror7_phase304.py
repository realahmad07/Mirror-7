from dataclasses import dataclass
from phase299_capability_frontier import CapabilityFrontier

@dataclass(frozen=True)
class ScheduledUpgrade:
    capability:str
    gap:float
    priority:float
    seed:int
    campaign_rounds:int

class FrontierScheduler:
    """Chooses the next highest-pressure capability and rotates deterministic evaluation seeds."""
    def __init__(self,frontier:CapabilityFrontier,seeds=(7,19,31,997)):
        self.frontier=frontier; self.seeds=tuple(int(x) for x in seeds); self.index=0
        if not self.seeds: raise ValueError("at least one seed is required")
    def next(self,max_rounds:int=4)->ScheduledUpgrade|None:
        if max_rounds<1: raise ValueError("max_rounds must be positive")
        top=self.frontier.top_gap()
        if top is None: return None
        seed=self.seeds[self.index%len(self.seeds)]
        self.index+=1
        return ScheduledUpgrade(top.name,top.gap,top.priority,seed,max_rounds)
