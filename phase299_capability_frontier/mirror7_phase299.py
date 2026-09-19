from dataclasses import dataclass
from typing import Dict, Iterable, List

@dataclass(frozen=True)
class CapabilityTarget:
    name:str
    baseline:float
    target:float
    priority:float=1.0

    @property
    def gap(self)->float:
        return max(0.0,self.target-self.baseline)

    @property
    def pressure(self)->float:
        return self.gap*max(0.0,self.priority)

class CapabilityFrontier:
    """Persistent target profile for deciding which verified capability gap to attack next."""
    def __init__(self,targets:Iterable[CapabilityTarget]):
        self._targets={t.name:t for t in targets}
        if not self._targets: raise ValueError("at least one capability target is required")

    @property
    def targets(self)->tuple[CapabilityTarget,...]:
        return tuple(self._targets.values())

    def ranked_gaps(self)->List[CapabilityTarget]:
        return sorted((t for t in self._targets.values() if t.gap>0),
                      key=lambda t:(t.pressure,t.gap,t.priority,t.name),reverse=True)

    def top_gap(self)->CapabilityTarget|None:
        xs=self.ranked_gaps()
        return xs[0] if xs else None

    def update(self,name:str,score:float)->CapabilityTarget:
        if name not in self._targets: raise KeyError(name)
        score=max(0.0,min(1.0,float(score)))
        old=self._targets[name]
        self._targets[name]=CapabilityTarget(name,score,old.target,old.priority)
        return self._targets[name]
