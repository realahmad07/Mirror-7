
from collections import deque
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

State = Tuple[int, ...]

class LongHorizonPlanner:
    """Breadth-first subgoal planner with explicit failure/backtracking memory."""

    def __init__(self, transition: Callable[[State,str],Optional[State]],
                 actions: Sequence[str], max_expansions: int = 10000):
        if max_expansions < 1: raise ValueError("max_expansions must be positive")
        self.transition=transition
        self.actions=tuple(actions)
        self.max_expansions=max_expansions
        self.failures=set()

    def ban(self, state: State, action: str):
        self.failures.add((tuple(state),action))

    def plan(self, start: State, goal: Callable[[State],bool], max_depth:int=20) -> Optional[List[str]]:
        q=deque([(tuple(start),[])])
        seen={(tuple(start),0)}
        expanded=0
        while q:
            state,path=q.popleft()
            if goal(state): return path
            if len(path)>=max_depth: continue
            expanded += 1
            if expanded>self.max_expansions: return None
            for action in self.actions:
                if (state,action) in self.failures: continue
                nxt=self.transition(state,action)
                if nxt is None: continue
                key=(tuple(nxt),len(path)+1)
                if key in seen: continue
                seen.add(key)
                q.append((tuple(nxt),path+[action]))
        return None

    def execute_with_backtracking(self,start:State,goal:Callable[[State],bool],
                                   executor:Callable[[State,str],Optional[State]],
                                   max_depth:int=20)->Optional[List[str]]:
        planned=self.plan(start,goal,max_depth)
        if planned is None: return None
        state=tuple(start)
        executed=[]
        for action in planned:
            nxt=executor(state,action)
            if nxt is None:
                self.ban(state,action)
                return self.execute_with_backtracking(start,goal,executor,max_depth)
            executed.append(action)
            state=tuple(nxt)
        return executed if goal(state) else None
