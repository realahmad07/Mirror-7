
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

State = Tuple[int, ...]
Action = Tuple[str, Tuple[int, ...]]

class AffordanceModel:
    """Infers observed preconditions and state deltas without a fixed action list."""

    def __init__(self):
        self._evidence=defaultdict(list)

    def observe(self, before: State, action: Action, after: State, success: bool=True):
        before=tuple(int(x) for x in before)
        after=tuple(int(x) for x in after)
        if len(before)!=len(after):
            raise ValueError("state width mismatch")
        self._evidence[action].append((before, after, bool(success)))

    def effect(self, state: State, action: Action) -> Optional[State]:
        rows=[r for r in self._evidence.get(action,[]) if r[2]]
        deltas={tuple(bi-ai for ai,bi in zip(before,after)) for before,after,_ in rows}
        matching=[]
        for d in deltas:
            if len([1 for before,after,_ in rows if tuple(y-x for x,y in zip(before,after))==d]) >= 1:
                matching.append(d)
        if len(set(matching))!=1:
            return None
        d=matching[0]
        return tuple(x+y for x,y in zip(state,d))

    def valid_actions(self, state: State) -> List[Action]:
        out=[]
        for action, rows in self._evidence.items():
            successes=sum(1 for before,after,ok in rows if ok and tuple(before)==tuple(state))
            failures=sum(1 for before,after,ok in rows if not ok and tuple(before)==tuple(state))
            if successes > failures and successes:
                out.append(action)
        return sorted(out)

    def actions(self) -> Tuple[Action,...]:
        return tuple(sorted(self._evidence))
