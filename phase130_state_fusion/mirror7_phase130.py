from dataclasses import dataclass
from typing import Any
@dataclass(frozen=True)
class FusedValue:
    value:Any; confidence:float; supporters:int
class StateFusion:
    def fuse(self,observations:list[tuple[Any,float]]):
        if not observations: return None
        groups={}
        for v,c in observations: groups.setdefault(repr(v),[]).append(max(0.0,float(c)))
        ranked=sorted(groups.items(),key=lambda kv:(sum(kv[1]),len(kv[1])),reverse=True)
        total=sum(sum(v) for _,v in ranked)
        if total<=0: return None
        key,scores=ranked[0]; value=next(v for v,c in observations if repr(v)==key)
        return FusedValue(value,sum(scores)/total,len(scores))
