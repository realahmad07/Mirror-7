from __future__ import annotations
from dataclasses import dataclass
from math import isfinite

@dataclass(frozen=True)
class TransferMap:
    source_to_target:tuple[int,...]
    scale:tuple[float,...]
    confidence:float

class Phase69Transfer:
    """Environment transfer through invariant effect signatures and anchors."""
    def __init__(self,min_confidence=.8):
        self.min_confidence=min_confidence
    @staticmethod
    def _signature(v):
        vals=[abs(float(x)) for x in v]
        total=sum(vals)
        if total==0: return tuple(0.0 for _ in vals)
        return tuple(round(x/total,4) for x in vals)
    def learn_map(self,source_effect,target_effect):
        s=tuple(float(x) for x in source_effect); t=tuple(float(x) for x in target_effect)
        if len(s)!=len(t) or not s: return None
        used=set(); mapping=[]; scales=[]; errors=[]
        for x in s:
            candidates=sorted((abs(abs(x)-abs(y)),j,y) for j,y in enumerate(t) if j not in used)
            if not candidates: return None
            err,j,y=candidates[0]
            if abs(x)<1e-12 or abs(y)<1e-12: scale=1.0
            else: scale=abs(y/x)
            mapping.append(j); scales.append(scale); errors.append(err/(1+abs(x)+abs(y))); used.add(j)
        conf=1.0-sum(errors)/len(errors)
        if conf<self.min_confidence: return None
        return TransferMap(tuple(mapping),tuple(round(x,6) for x in scales),max(0.0,min(1.0,conf)))
    def transfer(self,effect,transfer_map):
        if len(effect)!=len(transfer_map.source_to_target): return None
        out=[0.0]*len(effect)
        for i,x in enumerate(effect):
            out[transfer_map.source_to_target[i]]=float(x)*transfer_map.scale[i]
        return tuple(out)
    def invariant(self,effect):
        return self._signature(effect)
