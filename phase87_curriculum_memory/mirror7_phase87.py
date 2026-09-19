"""Phase 87: bounded autonomous curriculum from uncertainty and novelty."""

from dataclasses import dataclass
from math import isfinite
from typing import Dict, List, Optional, Sequence, Tuple

@dataclass(frozen=True)
class CurriculumItem:
    state: Tuple[float,...]
    action: str
    novelty: float
    uncertainty: float
    priority: float

class CurriculumMemory:
    def __init__(self,max_items=128,max_score=2.0):
        self.max_items=max_items
        self.max_score=max_score
        self.items: List[CurriculumItem]=[]
        self.seen: Dict[Tuple[Tuple[float,...],str],int]={}

    def propose(self,state: Sequence[float], action: str, novelty: float,
                uncertainty: float)->Optional[CurriculumItem]:
        try:
            s=tuple(float(x) for x in state); n=float(novelty); u=float(uncertainty)
        except (TypeError,ValueError):
            return None
        if not s or not action or not all(isfinite(x) for x in s) or not isfinite(n) or not isfinite(u):
            return None
        n=max(0.0,min(1.0,n)); u=max(0.0,min(1.0,u))
        key=(s,action)
        repeat=self.seen.get(key,0)
        priority=min(self.max_score, n+u)/(1+repeat)
        item=CurriculumItem(s,action,n,u,priority)
        self.seen[key]=repeat+1
        self.items.append(item)
        if len(self.items)>self.max_items: self.items.pop(0)
        return item

    def next(self)->Optional[CurriculumItem]:
        if not self.items: return None
        return max(self.items,key=lambda x:x.priority)

    def consume(self)->Optional[CurriculumItem]:
        item=self.next()
        if item is None: return None
        self.items.remove(item)
        return item
