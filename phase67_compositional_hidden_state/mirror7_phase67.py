from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Hashable, Sequence

Action = Hashable
Vector = tuple[float | int | None, ...]

def vec(x: Sequence[float | int | None]) -> Vector:
    if not isinstance(x,(tuple,list)) or not x: raise ValueError("state must be non-empty")
    if any(v is not None and not isinstance(v,(int,float)) for v in x): raise ValueError("numeric state required")
    return tuple(x)

@dataclass
class Stat:
    n:int=0
    mean:float=0.0
    def add(self,x:float):
        self.n+=1; self.mean += (x-self.mean)/self.n

@dataclass(frozen=True)
class Component:
    action:Action
    dimension:int
    context:tuple[int,...]
    mean:float
    samples:int

class CompositionModel:
    """Learns baseline effects plus context-dependent pair interactions."""
    def __init__(self,min_samples=3):
        self.min_samples=min_samples
        self.base=defaultdict(Stat)
        self.interaction=defaultdict(Stat)
        self._dims=set()
    def observe(self,action,before,after):
        b,a=vec(before),vec(after)
        if len(b)!=len(a): raise ValueError("state dimensionality mismatch")
        active=tuple(i for i,x in enumerate(b) if x is not None and float(x)!=0.0)
        for d,(x,y) in enumerate(zip(b,a)):
            if x is None or y is None: continue
            delta=float(y)-float(x); self._dims.add(d)
            self.base[(action,d)].add(delta)
            for c in active:
                if c!=d: self.interaction[(action,d,c)].add(delta-self.base[(action,d)].mean)
    def _base(self,action,d):
        s=self.base.get((action,d)); return s.mean if s and s.n>=self.min_samples else None
    def predict_delta(self,action,state):
        s=vec(state); out=[0.0]*len(s); found=False
        active={i for i,x in enumerate(s) if x is not None and float(x)!=0.0}
        for d,x in enumerate(s):
            if x is None: continue
            b=self._base(action,d)
            if b is None: continue
            value=b
            for c in active:
                if c==d: continue
                q=self.interaction.get((action,d,c))
                if q and q.n>=self.min_samples: value+=q.mean
            out[d]=value; found=True
        return tuple(out) if found else None
    def components(self):
        out=[]
        for (a,d),s in self.base.items():
            if s.n>=self.min_samples: out.append(Component(a,d,(),s.mean,s.n))
        for (a,d,c),s in self.interaction.items():
            if s.n>=self.min_samples: out.append(Component(a,d,(c,),s.mean,s.n))
        return tuple(out)

@dataclass(frozen=True)
class Hypothesis:
    id:int
    signature:tuple
    support:int

class Phase67Agent:
    """Compositional latent-state learner with unlabeled competing hypotheses."""
    def __init__(self,min_samples=3,revision_window=2,revision_threshold=.25):
        self.model=CompositionModel(min_samples)
        self.min_samples=min_samples
        self.revision_window=revision_window
        self.revision_threshold=revision_threshold
        self.hypotheses:list[Hypothesis]=[]
        self.active=None
        self._mismatch=0
    def observe(self,action,before,after):
        self.model.observe(action,before,after)
        sig=tuple((repr(c.action),c.dimension,c.context,round(c.mean,4)) for c in self.model.components())
        if not sig: return
        if self.active is None:
            self.hypotheses.append(Hypothesis(0,sig,1)); self.active=0; return
        old=dict((k[:3],k[3]) for k in self.hypotheses[self.active].signature)
        new=dict((k[:3],k[3]) for k in sig)
        keys=set(old)|set(new)
        dist=sum(min(1.0,abs(old.get(k,0)-new.get(k,0))/max(1,abs(old.get(k,0)),abs(new.get(k,0)))) for k in keys)/max(1,len(keys))
        self._mismatch=self._mismatch+1 if dist>self.revision_threshold else 0
        if self._mismatch>=self.revision_window:
            self.hypotheses.append(Hypothesis(len(self.hypotheses),sig,1)); self.active=len(self.hypotheses)-1; self._mismatch=0
        else:
            h=self.hypotheses[self.active]; self.hypotheses[self.active]=Hypothesis(h.id,h.signature,h.support+1)
    def predict(self,action,state):
        return self.model.predict_delta(action,state)
    def infer(self):
        return self.active
    def fail_closed(self):
        return {"hypotheses":len(self.hypotheses),"active":self.active,"components":len(self.model.components())}
