from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict

@dataclass(frozen=True)
class Segment:
    start:int
    end:int
    signature:tuple
    support:int

class RawRepresentationLearner:
    """Deterministic, bounded segmentation and feature invention over raw numeric streams."""
    def __init__(self,window=3,change_threshold=2.0,min_support=2):
        if window<1: raise ValueError("window must be positive")
        self.window=window; self.change_threshold=change_threshold; self.min_support=min_support
        self.segments=[]; self.features=defaultdict(int)
    def _signature(self,chunk):
        vals=[float(x) for x in chunk]
        mean=sum(vals)/len(vals); span=max(vals)-min(vals)
        return (round(mean,6),round(span,6),len(vals))
    def discover(self,stream):
        x=tuple(float(v) for v in stream)
        if not x: return ()
        cuts=[0]
        for i in range(self.window,len(x),self.window):
            left=x[i-self.window:i]; right=x[i:i+self.window]
            if not right: break
            lm=sum(left)/len(left); rm=sum(right)/len(right)
            if abs(rm-lm)>=self.change_threshold: cuts.append(i)
        cuts.append(len(x))
        seg=[]
        for a,b in zip(cuts,cuts[1:]):
            if b>a:
                sig=self._signature(x[a:b]); self.features[sig]+=1
                if self.features[sig]>=self.min_support: seg.append(Segment(a,b,sig,self.features[sig]))
        self.segments=seg
        return tuple(seg)
    def encode(self,stream):
        return tuple(s.signature for s in self.discover(stream))
    def invented_features(self):
        return tuple(sorted(self.features))
    def fail_closed(self):
        return {"segments":len(self.segments),"features":len(self.features)}
