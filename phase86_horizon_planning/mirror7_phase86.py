"""Phase 86: bounded multi-step planning over learned transitions."""

from dataclasses import dataclass
from math import isfinite
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

State=Tuple[float,...]
Action=str

def norm(v):
    try: x=tuple(float(a) for a in v)
    except (TypeError,ValueError): return None
    return x if x and all(isfinite(a) for a in x) else None

@dataclass(frozen=True)
class PlanResult:
    actions: Tuple[Action,...]
    predicted: Optional[State]
    score: float
    failed_closed: bool

class HorizonPlanner:
    def __init__(self, transitions: Dict[Tuple[State,Action],State],
                 actions: Iterable[Action], max_depth=4, max_nodes=128):
        self.t=transitions
        self.actions=tuple(dict.fromkeys(actions))
        self.max_depth=max_depth
        self.max_nodes=max_nodes

    def plan(self, start: Sequence[float], goal: Sequence[float])->PlanResult:
        s,g=norm(start),norm(goal)
        if s is None or g is None or len(s)!=len(g) or self.max_depth<1 or self.max_nodes<1:
            return PlanResult((),None,float("-inf"),True)
        frontier=[(s,(),0.0)]
        visited=0
        best=((),s,float("-inf"))
        for _ in range(self.max_depth):
            nxt=[]
            for state,path,_ in frontier:
                for action in self.actions:
                    visited+=1
                    if visited>self.max_nodes: return PlanResult(best[0],best[1],best[2],True)
                    ns=self.t.get((state,action))
                    if ns is None: continue
                    score=-sum((a-b)**2 for a,b in zip(ns,g))
                    p=path+(action,)
                    if score>best[2]: best=(p,ns,score)
                    if score==0: return PlanResult(p,ns,score,False)
                    nxt.append((ns,p,score))
            frontier=nxt
            if not frontier: break
        return PlanResult(best[0],best[1],best[2],False)

def replay_plan(start, actions, transitions):
    s=norm(start)
    if s is None: return None
    for a in actions:
        s=transitions.get((s,a))
        if s is None: return None
    return s
