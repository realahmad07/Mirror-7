from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from typing import Hashable

@dataclass(frozen=True)
class WorldPrediction:
    state:tuple
    uncertainty:float
    evidence:int

class UnifiedWorldModel:
    """Closed-loop bounded world model: learn, predict, counterfactually roll out, plan, revise."""
    def __init__(self,min_samples=3,max_depth=5,beam_width=8):
        self.min_samples=min_samples; self.max_depth=max_depth; self.beam_width=beam_width
        self.effects=defaultdict(list); self.hypotheses=[]; self.active=0; self._mismatch=0
    def observe(self,action,before,after):
        b,a=tuple(before),tuple(after)
        if len(b)!=len(a): raise ValueError("dimension mismatch")
        delta=tuple(None if x is None or y is None else float(y)-float(x) for x,y in zip(b,a))
        self.effects[action].append(delta)
        if len(self.effects[action])>=self.min_samples:
            mean=tuple(sum(v[d] for v in self.effects[action] if v[d] is not None)/max(1,sum(v[d] is not None for v in self.effects[action])) for d in range(len(b)))
            sig=(repr(action),tuple(round(x,4) for x in mean))
            if not self.hypotheses: self.hypotheses=[sig]
            else:
                old=self.hypotheses[self.active][1]
                dist=sum(abs(x-y)/(1+abs(x)+abs(y)) for x,y in zip(old,sig[1]))/len(old)
                self._mismatch=self._mismatch+1 if dist>.25 else 0
                if self._mismatch>=2: self.hypotheses.append(sig); self.active=len(self.hypotheses)-1; self._mismatch=0
    def predict(self,action,state):
        samples=self.effects.get(action,[])
        if len(samples)<self.min_samples: return None
        s=tuple(state); out=[]
        for d,x in enumerate(s):
            vals=[v[d] for v in samples if v[d] is not None]
            if x is None or not vals: out.append(None)
            else: out.append(float(x)+sum(vals)/len(vals))
        return WorldPrediction(tuple(out),1.0/(1+len(samples)),len(samples))
    def counterfactual(self,state,actions):
        s=tuple(state); uncertainty=0.0
        for action in actions:
            p=self.predict(action,s)
            if p is None: return None
            s=p.state; uncertainty+=p.uncertainty
        return WorldPrediction(s,uncertainty,sum(len(self.effects[a]) for a in actions))
    def plan(self,state,actions,target):
        frontier=[(0.0,(),tuple(state),0.0)]
        best=None
        for _ in range(self.max_depth):
            expanded=[]
            for _,seq,s,u in frontier:
                for a in actions:
                    p=self.predict(a,s)
                    if p is None: continue
                    dist=sum(abs(float(x)-float(y)) for x,y in zip(p.state,target) if x is not None)
                    item=(dist+u+p.uncertainty,seq+(a,),p.state,u+p.uncertainty)
                    expanded.append(item)
                    best=item if best is None or item[0]<best[0] else best
            if not expanded: break
            frontier=sorted(expanded,key=lambda x:(x[0],x[1]))[:self.beam_width]
        return None if best is None else WorldPrediction(best[2],best[3],len(best[1]))
    def select_action(self,state,actions,target):
        r=self.plan(state,actions,target)
        return None if r is None else r.state
    def fail_closed(self):
        return {"known_actions":len(self.effects),"hypotheses":len(self.hypotheses),"active":self.active}
