from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TransferResult:
    mapped: tuple|None
    confidence: float
    accepted: bool
    reason: str

class RawTransfer:
    """Small-state invariant transfer from source signatures to a new raw scale."""
    def __init__(self, tolerance=.25, min_pairs=2):
        self.tolerance=tolerance; self.min_pairs=min_pairs; self.source=[]; self.target=[]

    def _shape(self,x):
        x=tuple(float(v) for v in x)
        if not x: return None
        lo,hi=min(x),max(x); span=hi-lo
        if span==0: return tuple(0.0 for _ in x)
        return tuple(round((v-lo)/span,6) for v in x)

    def fit(self, source_cases, target_cases):
        if len(source_cases)!=len(target_cases) or len(source_cases)<self.min_pairs:
            return TransferResult(None,0.0,False,"insufficient_pairs")
        scores=[]
        for s,t in zip(source_cases,target_cases):
            a,b=self._shape(s),self._shape(t)
            if a is None or b is None or len(a)!=len(b): return TransferResult(None,0.0,False,"shape_mismatch")
            scores.append(sum(abs(x-y) for x,y in zip(a,b))/len(a))
        err=sum(scores)/len(scores); conf=max(0.0,1.0-err)
        self.source=list(source_cases); self.target=list(target_cases)
        return TransferResult(tuple(scores),conf,conf>=1-self.tolerance,"accepted" if conf>=1-self.tolerance else "rejected")

    def map(self, raw):
        if not self.target: return None
        shape=self._shape(raw)
        if shape is None: return None
        ref=self._shape(self.target[-1])
        if ref is None or len(shape)!=len(ref): return None
        err=sum(abs(x-y) for x,y in zip(shape,ref))/len(shape)
        return TransferResult(shape,max(0.0,1-err),err<=self.tolerance,"mapped" if err<=self.tolerance else "rejected")

    def fail_closed(self): return {"pairs":len(self.source),"ready":bool(self.target)}
