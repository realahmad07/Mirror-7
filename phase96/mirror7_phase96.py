from __future__ import annotations
from dataclasses import dataclass
from math import isfinite

@dataclass(frozen=True)
class Rule:
    key: tuple
    value: tuple[float,...]
    support: int
    confidence: float

class ContinualConsolidator:
    def __init__(self,max_rules=128,min_support=3,contradiction_margin=.25):
        if max_rules<1 or min_support<1 or contradiction_margin<0: raise ValueError("invalid bounds")
        self.max_rules=max_rules; self.min_support=min_support; self.contradiction_margin=float(contradiction_margin)
        self._counts={}; self.rules={}

    @staticmethod
    def _value(value):
        v=tuple(float(x) for x in value)
        if not v or not all(isfinite(x) for x in v): raise ValueError("invalid value")
        return v

    @staticmethod
    def _distance(a,b):
        return float("inf") if len(a)!=len(b) else max(abs(x-y) for x,y in zip(a,b))

    def observe(self,key,value):
        k=tuple(key); v=self._value(value); bucket=self._counts.setdefault(k,{})
        bucket[v]=bucket.get(v,0)+1
        winner,support=max(bucket.items(),key=lambda x:(x[1],x[0])); total=sum(bucket.values())
        confidence=support/total; existing=self.rules.get(k)
        if support<self.min_support: return existing
        if existing is not None and self._distance(existing.value,winner)>self.contradiction_margin and support*2<=total:
            return existing
        rule=Rule(k,winner,support,confidence); self.rules[k]=rule
        if len(self.rules)>self.max_rules:
            weakest=min(self.rules.values(),key=lambda r:(r.confidence,r.support)); del self.rules[weakest.key]
        return rule

    def recall(self,key): return self.rules.get(tuple(key))
    def consolidate(self): return tuple(sorted(self.rules.values(),key=lambda r:(r.key,-r.confidence)))
    def fail_closed(self): return {"rules":len(self.rules),"candidates":sum(len(v) for v in self._counts.values())}
