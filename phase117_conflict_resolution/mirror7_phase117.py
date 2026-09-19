from dataclasses import dataclass
from collections import defaultdict
from typing import Any

@dataclass(frozen=True)
class Resolution:
    status: str
    value: Any
    confidence: float
    reason: str

class ConflictResolver:
    """Resolves evidence only when weighted support has a clear margin; otherwise abstains."""
    def resolve(self, evidence:list[tuple[Any,float]] , margin:float=0.2)->Resolution:
        if not evidence: return Resolution("unknown",None,0.0,"no evidence")
        scores=defaultdict(float)
        for value,weight in evidence: scores[repr(value)]+=max(0.0,float(weight))
        ranked=sorted(scores.items(),key=lambda x:x[1],reverse=True)
        total=sum(scores.values())
        best=ranked[0]
        second=ranked[1][1] if len(ranked)>1 else 0.0
        if total <= 0.0:
            return Resolution("unknown",None,0.0,"no positive support")
        confidence=best[1]/total
        if second and best[1]-second < margin:
            return Resolution("conflict",None,confidence,"support margin too small")
        original=next(v for v,w in evidence if repr(v)==best[0])
        return Resolution("resolved",original,confidence,"clear weighted support")
