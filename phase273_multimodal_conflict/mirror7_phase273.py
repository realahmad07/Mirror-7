
from dataclasses import dataclass
from math import exp
from typing import Any, Iterable, List, Optional

@dataclass(frozen=True)
class Evidence:
    modality: str
    value: Any
    confidence: float
    timestamp: float

class ConflictResolver:
    """Weighted, recency-aware fusion with explicit abstention."""

    def __init__(self, recency_half_life: float=10.0, margin: float=0.15):
        if recency_half_life<=0 or margin<0:
            raise ValueError("invalid resolver bounds")
        self.half_life=recency_half_life
        self.margin=margin

    def _weight(self, e:Evidence, now:float)->float:
        return max(0.0,float(e.confidence))*2.0**(-(max(0.0,now-e.timestamp)/self.half_life))

    def resolve(self, evidence:Iterable[Evidence], now:float) -> Optional[Any]:
        scores={}
        for e in evidence:
            if not 0.0<=e.confidence<=1.0:
                continue
            scores[e.value]=scores.get(e.value,0.0)+self._weight(e,now)
        if not scores:
            return None
        ranked=sorted(scores.items(), key=lambda x:(-x[1],repr(x[0])))
        if len(ranked)>1 and ranked[0][1]-ranked[1][1] < self.margin:
            return None
        return ranked[0][0]

    def contradiction(self, evidence:Iterable[Evidence], now:float)->bool:
        vals={e.value for e in evidence if self._weight(e,now)>0}
        return len(vals)>1
