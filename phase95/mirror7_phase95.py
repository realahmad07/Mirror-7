from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Sequence

@dataclass(frozen=True)
class MemoryItem:
    signature: tuple[float, ...]
    state: tuple[float, ...]
    action: str
    result: tuple[float, ...]
    confidence: float
    visits: int

class AssociativeMemory:
    def __init__(self, max_items: int = 256, max_dimension: int = 32):
        if max_items < 1 or max_dimension < 1:
            raise ValueError("invalid bounds")
        self.max_items=max_items; self.max_dimension=max_dimension; self.items=[]

    @staticmethod
    def _vec(x):
        v=tuple(float(a) for a in x)
        if not v or not all(isfinite(a) for a in v): raise ValueError("invalid vector")
        return v

    def add(self, signature, state, action, result, confidence=1.0):
        s,st,r=self._vec(signature),self._vec(state),self._vec(result)
        c=float(confidence)
        if len(s)>self.max_dimension or len(st)!=len(r) or not 0.0<=c<=1.0: raise ValueError("invalid memory item")
        item=MemoryItem(s,st,str(action),r,c,1); self.items.append(item)
        if len(self.items)>self.max_items:
            self.items.sort(key=lambda x:(x.confidence,x.visits)); self.items.pop(0)
        return item

    @staticmethod
    def _distance(a,b):
        return float("inf") if len(a)!=len(b) else sum(abs(x-y) for x,y in zip(a,b))/len(a)

    def retrieve(self, signature, state, top_k=4, max_distance=float("inf")):
        if top_k<1: raise ValueError("top_k must be positive")
        qsig,qstate=self._vec(signature),self._vec(state); ranked=[]
        for item in self.items:
            ds=self._distance(qsig,item.signature); dt=self._distance(qstate,item.state)
            if ds==float("inf") or dt==float("inf"): continue
            d=.6*ds+.4*dt
            if d<=max_distance: ranked.append((d,-item.confidence,-item.visits,item))
        ranked.sort(key=lambda x:x[:3]); return tuple(x[3] for x in ranked[:top_k])

    def reinforce(self,item):
        for i,current in enumerate(self.items):
            if current==item:
                self.items[i]=MemoryItem(current.signature,current.state,current.action,current.result,min(1.0,current.confidence+.05),current.visits+1)
                return True
        return False

    def fail_closed(self): return {"items":len(self.items)}
