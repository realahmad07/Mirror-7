from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

@dataclass(frozen=True)
class ImprovementRecord:
    capability:str
    before:float
    after:float
    accepted:bool
    fingerprint:str
    round:int

class ImprovementHistory:
    """Persistent bounded provenance for frontier upgrades and stagnation detection."""
    def __init__(self): self._records=[]
    @staticmethod
    def fingerprint(capability:str,variant:str)->str:
        return sha256((capability+"|"+variant).encode()).hexdigest()
    def record(self,capability:str,before:float,after:float,accepted:bool,variant:str,round:int):
        self._records.append(ImprovementRecord(capability,before,after,accepted,self.fingerprint(capability,variant),round))
    @property
    def records(self)->tuple[ImprovementRecord,...]: return tuple(self._records)
    def recent(self,capability:str,limit:int=5)->tuple[ImprovementRecord,...]:
        if limit<1: raise ValueError("limit must be positive")
        return tuple(r for r in self._records if r.capability==capability)[-limit:]
    def stagnant(self,capability:str,limit:int=3)->bool:
        xs=self.recent(capability,limit)
        return len(xs)>=limit and not any(r.accepted and r.after>r.before for r in xs)
