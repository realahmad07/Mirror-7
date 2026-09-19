from dataclasses import dataclass
from typing import Iterable, List

@dataclass(frozen=True)
class CandidateUtility:
    name:str
    gain:float
    held_out:float
    complexity:int
    operations:int
    utility:float

class ResourceAwareSelector:
    """Ranks verified candidates by evidence while applying explicit complexity penalties."""
    def __init__(self,complexity_weight:float=.01,operation_weight:float=.001):
        if complexity_weight<0 or operation_weight<0: raise ValueError("weights must be non-negative")
        self.complexity_weight=complexity_weight; self.operation_weight=operation_weight

    def make(self,name:str,gain:float,held_out:float,complexity:int,operations:int)->CandidateUtility:
        if gain<0 or not 0<=held_out<=1 or complexity<0 or operations<0:
            raise ValueError("invalid candidate metrics")
        utility=gain+held_out-self.complexity_weight*complexity-self.operation_weight*operations
        return CandidateUtility(name,gain,held_out,complexity,operations,utility)

    def rank(self,candidates:Iterable[CandidateUtility])->List[CandidateUtility]:
        return sorted(candidates,key=lambda c:(c.utility,c.held_out,c.gain,-c.complexity,-c.operations,c.name),reverse=True)
