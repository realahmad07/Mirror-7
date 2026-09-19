from __future__ import annotations
from dataclasses import dataclass
from itertools import permutations

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
        if len(s)!=len(t) or not s or len(s)>8: return None
        s_sig=self._signature(s)
        t_sig=self._signature(t)
        def err_for(p):
            return sum(abs(s_sig[i]-t_sig[p[i]]) for i in range(len(s)))/len(s)
        p=min(permutations(range(len(t))),key=err_for)
        err=err_for(p)
        scales=tuple(1.0 if abs(s[i])<1e-12 or abs(t[p[i]])<1e-12 else abs(t[p[i]]/s[i]) for i in range(len(s)))
        conf=max(0.0,min(1.0,1.0-err))
        if conf<self.min_confidence: return None
        return TransferMap(tuple(p),tuple(round(x,6) for x in scales),conf)
    def transfer(self,effect,transfer_map):
        if transfer_map is None or len(effect)!=len(transfer_map.source_to_target): return None
        out=[0.0]*len(effect)
        for i,x in enumerate(effect):
            out[transfer_map.source_to_target[i]]=float(x)*transfer_map.scale[i]
        return tuple(out)
    def invariant(self,effect):
        return self._signature(effect)
