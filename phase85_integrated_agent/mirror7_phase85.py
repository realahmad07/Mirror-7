"""Phase 85: bounded integration of state, prediction, memory, graph, consistency and action feedback."""

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

State = Tuple[float, ...]
Action = str

def _state(v: Sequence[float]) -> Optional[State]:
    try:
        x=tuple(float(a) for a in v)
    except (TypeError, ValueError):
        return None
    return x if x and all(isfinite(a) for a in x) else None

@dataclass(frozen=True)
class StepResult:
    action: Optional[Action]
    predicted: Optional[State]
    observed: Optional[State]
    discrepancy: Optional[float]
    revised: bool
    failed_closed: bool

class IntegratedAgent:
    """Finite-action agent with bounded model, graph, memory and revision."""

    def __init__(self, actions: Iterable[Action], tolerance: float=0.25,
                 max_memory: int=128, max_revisions: int=8):
        self.actions=tuple(dict.fromkeys(actions))
        self.tolerance=float(tolerance)
        self.max_memory=max_memory
        self.max_revisions=max_revisions
        self.revisions=0
        self.transitions: Dict[Tuple[State,Action],State]={}
        self.memory: List[Tuple[State,Action,State]]=[]
        self.graph: Dict[State,List[Action]]={}

    def _valid(self, s: State, a: Action)->bool:
        return bool(s) and a in self.actions

    def learn(self, state: Sequence[float], action: Action, result: Sequence[float])->bool:
        s,r=_state(state),_state(result)
        if s is None or r is None or not self._valid(s,action) or len(s)!=len(r):
            return False
        self.transitions[(s,action)]=r
        self.graph.setdefault(s,[])
        if action not in self.graph[s]: self.graph[s].append(action)
        self.memory.append((s,action,r))
        if len(self.memory)>self.max_memory: self.memory.pop(0)
        return True

    def predict(self, state: Sequence[float], action: Action)->Optional[State]:
        s=_state(state)
        if s is None or not self._valid(s,action): return None
        return self.transitions.get((s,action))

    def step(self, state: Sequence[float], observe: Callable[[Action], Sequence[float]],
             score: Optional[Callable[[State],float]]=None)->StepResult:
        s=_state(state)
        if s is None or not self.actions: return StepResult(None,None,None,None,False,True)
        legal=list(self.graph.get(s,[]))
        unknown=[a for a in self.actions if a not in legal]
        candidates=legal+unknown
        if score is not None:
            try: candidates.sort(key=lambda a: float(score(self.predict(s,a) or s)), reverse=True)
            except Exception: return StepResult(None,None,None,None,False,True)
        action=candidates[0]
        predicted=self.predict(s,action)
        try: observed=_state(observe(action))
        except Exception: return StepResult(action,predicted,None,None,False,True)
        if observed is None or len(observed)!=len(s): return StepResult(action,predicted,observed,None,False,True)
        discrepancy=None if predicted is None else max(abs(a-b) for a,b in zip(predicted,observed))
        revised=False
        if predicted is not None and discrepancy is not None and discrepancy>self.tolerance:
            if self.revisions>=self.max_revisions:
                return StepResult(action,predicted,observed,discrepancy,False,True)
            self.revisions+=1; revised=True
        if not self.learn(s,action,observed):
            return StepResult(action,predicted,observed,discrepancy,revised,True)
        return StepResult(action,predicted,observed,discrepancy,revised,False)
