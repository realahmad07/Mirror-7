from __future__ import annotations
from dataclasses import dataclass
from math import isfinite

@dataclass(frozen=True)
class Plan:
    actions: tuple[str,...]
    predicted: tuple[float,...]
    cost: float
    uncertainty: float
    failed_closed: bool

class UncertaintyAwarePlanner:
    def __init__(self,transitions,actions,max_depth=6,beam_width=8,node_budget=256,uncertainty_weight=.25):
        if min(max_depth,beam_width,node_budget)<1 or uncertainty_weight<0: raise ValueError("invalid bounds")
        self.transitions=transitions; self.actions=tuple(dict.fromkeys(actions))
        self.max_depth=max_depth; self.beam_width=beam_width; self.node_budget=node_budget
        self.uncertainty_weight=float(uncertainty_weight)

    @staticmethod
    def _state(x):
        s=tuple(float(v) for v in x)
        return s if s and all(isfinite(v) for v in s) else None

    def plan(self,start,goal):
        s,g=self._state(start),self._state(goal)
        if s is None or g is None or len(s)!=len(g): return Plan((),s or (),float("inf"),float("inf"),True)
        frontier=[(0.0,0.0,(),s)]; best=(float("inf"),float("inf"),(),s); expanded=0
        for _ in range(self.max_depth):
            nxt=[]
            for _,path_unc,path,state in frontier:
                for action in self.actions:
                    expanded+=1
                    if expanded>self.node_budget: return Plan(best[2],best[3],best[0],best[1],True)
                    edge=self.transitions.get((state,action))
                    if edge is None: continue
                    ns,unc=edge; ns=self._state(ns); u=float(unc)
                    if ns is None or not isfinite(u) or u<0: return Plan((),s,float("inf"),float("inf"),True)
                    dist=sum(abs(a-b) for a,b in zip(ns,g)); total_unc=path_unc+u
                    cost=dist+self.uncertainty_weight*total_unc
                    nxt.append((cost,total_unc,path+(action,),ns))
            if not nxt: break
            nxt.sort(key=lambda x:(x[0],x[2]))
            if nxt[0][0] < best[0]:
                best=nxt[0]
            if sum(abs(a-b) for a,b in zip(nxt[0][3],g)) == 0:
                x=nxt[0]; return Plan(x[2],x[3],x[0],x[1],False)
            frontier=nxt[:self.beam_width]
        return Plan(best[2],best[3],best[0],best[1],False)

    @staticmethod
    def replay(start,actions,transitions):
        state=UncertaintyAwarePlanner._state(start)
        if state is None: return None
        for action in actions:
            edge=transitions.get((state,action))
            if edge is None: return None
            state=tuple(edge[0])
        return state
