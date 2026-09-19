from __future__ import annotations
from dataclasses import dataclass
from typing import Hashable, Sequence

@dataclass(frozen=True)
class PlanResult:
    sequence:tuple
    predicted_state:tuple
    score:float
    depth:int

class Phase68Planner:
    """Uncertainty-aware bounded counterfactual planner."""
    def __init__(self,model,max_depth=6,beam_width=8,uncertainty_penalty=.25):
        self.model=model; self.max_depth=max_depth; self.beam_width=beam_width; self.uncertainty_penalty=uncertainty_penalty
    def _step(self,state,action):
        delta=self.model.predict(action,state)
        if delta is None: return None,1.0
        out=[]
        for x,d in zip(state,delta):
            out.append(None if x is None else float(x)+float(d))
        return tuple(out),0.0
    def plan(self,state,actions,target,goal_tolerance=.25):
        state=tuple(state); target=tuple(target); frontier=[(0.0,(),state,0.0)]
        best=None
        for depth in range(1,self.max_depth+1):
            expanded=[]
            for _,seq,s,unc in frontier:
                for action in actions:
                    ns,nu=self._step(s,action)
                    if ns is None: continue
                    dist=sum(abs(float(x)-float(y)) for x,y in zip(ns,target) if x is not None and y is not None)
                    score=dist+self.uncertainty_penalty*(unc+nu)
                    item=(score,seq+(action,),ns,unc+nu)
                    expanded.append(item)
                    if best is None or score<best[0]: best=item
            expanded.sort(key=lambda x:(x[0],x[1]))
            frontier=expanded[:self.beam_width]
            if best and best[0]<=goal_tolerance: break
        if best is None: return None
        return PlanResult(best[1],best[2],best[0],len(best[1]))

    def counterfactual(self,state,sequence):
        s=tuple(state); uncertainty=0.0
        for action in sequence:
            s,u=self._step(s,action)
            if s is None: return None
            uncertainty+=u
        return {"state":s,"uncertainty":uncertainty}
