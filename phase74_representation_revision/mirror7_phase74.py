from __future__ import annotations
from dataclasses import dataclass
from collections import Counter

@dataclass(frozen=True)
class RevisionResult:
    representation: tuple
    changed: bool
    regime: int
    reason: str

class RepresentationRevision:
    """Bounded representation revision when raw signatures drift."""
    def __init__(self, max_features=16, drift_threshold=0.35, min_support=2):
        self.max_features=max_features; self.drift_threshold=drift_threshold; self.min_support=min_support
        self.features=Counter(); self.active=(); self.regime=0

    def _sig(self, raw):
        x=tuple(float(v) for v in raw)
        if not x: return ()
        mean=sum(x)/len(x); span=max(x)-min(x)
        return (round(mean,5), round(span,5), len(x))

    def observe(self, raw):
        sig=self._sig(raw)
        if not sig: return RevisionResult((),False,self.regime,"empty")
        self.features[sig]+=1
        if not self.active:
            self.active=sig
            return RevisionResult(self.active,False,self.regime,"initialized")
        old=self.active
        dist=sum(abs(float(a)-float(b))/(1+abs(float(a))+abs(float(b))) for a,b in zip(old[:2],sig[:2]))/2
        if dist >= self.drift_threshold and self.features[sig] >= self.min_support:
            self.active=sig; self.regime+=1
            if len(self.features)>self.max_features:
                keep=dict(self.features.most_common(self.max_features)); self.features=Counter(keep)
            return RevisionResult(self.active,True,self.regime,"drift")
        return RevisionResult(self.active,False,self.regime,"stable")

    def encode(self, raw): return self.observe(raw).representation
    def fail_closed(self): return {"features":len(self.features),"regime":self.regime,"active":self.active}
