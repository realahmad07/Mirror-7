"""Phase 60: bounded long-horizon world modeling.

Learns action-conditioned state transitions with two evidence layers:
1. exact state/action memories for high-confidence replay;
2. factorized per-feature action rules that generalize to unseen states.

Rules require repeated consistent evidence. Conflicting evidence is marked
uncertain and predictions abstain rather than inventing a transition. Rollout
is therefore fail-closed over long horizons.
"""
from __future__ import annotations
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Hashable, Sequence, Tuple

State=Tuple[Hashable,...]
Action=Hashable

@dataclass(frozen=True)
class StepResult:
    next_state: State | None
    known: bool
    reason: str

class LongHorizonWorldModel:
    def __init__(self,min_support:int=2):
        if min_support<2:
            raise ValueError('min_support must be >=2')
        self.min_support=min_support
        self._exact=defaultdict(Counter)
        self._feature=defaultdict(lambda: defaultdict(Counter))
        self._uncertain=set()
        self._delta=defaultdict(Counter)
        self._fitted=False

    @staticmethod
    def _validate_state(state):
        if not isinstance(state,(tuple,list)) or not state:
            raise ValueError('state must be a non-empty tuple/list')
        return tuple(state)

    def fit(self,episodes:Sequence[Sequence[Tuple[State,Action,State]]]):
        if not episodes or any(not ep for ep in episodes):
            raise ValueError('episodes cannot be empty')
        self._exact=defaultdict(Counter)
        self._feature=defaultdict(lambda: defaultdict(Counter))
        self._uncertain=set()
        self._delta=defaultdict(Counter)
        for ep in episodes:
            for s,a,n in ep:
                s=self._validate_state(s)
                n=self._validate_state(n)
                if len(s)!=len(n):
                    raise ValueError('state dimensionality mismatch')
                self._exact[(s,a)][n]+=1
                for i,(x,y) in enumerate(zip(s,n)):
                    self._feature[(a,i)][x][y]+=1
                    if isinstance(x,(int,float)) and isinstance(y,(int,float)):
                        self._delta[(a,i)][y-x]+=1
        for key,by_value in self._feature.items():
            for old,counts in by_value.items():
                best=counts.most_common()
                if len(best)>1 and best[0][1] < sum(counts.values()):
                    if best[0][1] < self.min_support or best[0][1] <= best[1][1]:
                        self._uncertain.add((key,old))
        self._fitted=True
        return self

    def _exact_predict(self,state:State,action:Action):
        counts=self._exact.get((state,action))
        if not counts:
            return None
        best=counts.most_common()
        if best[0][1] < self.min_support:
            return None
        if len(best)>1 and best[0][1] <= best[1][1]:
            return None
        return best[0][0]

    def predict(self,state:State,action:Action)->StepResult:
        if not self._fitted:
            raise RuntimeError('model is not fitted')
        state=self._validate_state(state)
        exact=self._exact_predict(state,action)
        if exact is not None:
            return StepResult(exact,True,'exact')
        out=[]
        for i,x in enumerate(state):
            key=(action,i)
            counts=self._feature.get(key,{}).get(x)
            if counts and (key,x) not in self._uncertain:
                best=counts.most_common()
                if best[0][1] >= self.min_support and (len(best)==1 or best[0][1]>best[1][1]):
                    out.append(best[0][0])
                    continue
            if (key,x) in self._uncertain:
                return StepResult(None,False,'conflicting-observed-feature')
            deltas=self._delta.get(key)
            if deltas:
                dbest=deltas.most_common()
                if dbest[0][1] >= self.min_support and (len(dbest)==1 or dbest[0][1]>dbest[1][1]):
                    out.append(x + dbest[0][0])
                    continue
            return StepResult(None,False,'insufficient-or-conflicting-feature-evidence')
        return StepResult(tuple(out),True,'factorized')

    def update(self,state:State,action:Action,next_state:State)->None:
        state=self._validate_state(state)
        next_state=self._validate_state(next_state)
        if len(state)!=len(next_state):
            raise ValueError('state dimensionality mismatch')
        self._exact[(state,action)][next_state]+=1
        for i,(x,y) in enumerate(zip(state,next_state)):
            self._feature[(action,i)][x][y]+=1
            if isinstance(x,(int,float)) and isinstance(y,(int,float)):
                self._delta[(action,i)][y-x]+=1
        self._fitted=True

    def rollout(self,state:State,actions:Sequence[Action])->Tuple[State,...]:
        if not self._fitted:
            raise RuntimeError('model is not fitted')
        current=self._validate_state(state)
        out=[]
        for action in actions:
            step=self.predict(current,action)
            if not step.known:
                break
            current=step.next_state
            out.append(current)
        return tuple(out)

    def discrepancy(self,state:State,action:Action,observed:State)->dict:
        predicted=self.predict(state,action)
        observed=self._validate_state(observed)
        if not predicted.known:
            return {'known':False,'mismatch':None,'reason':predicted.reason}
        return {'known':True,'mismatch':predicted.next_state!=observed,'predicted':predicted.next_state,'observed':observed}

    def known_exact_pairs(self)->int:
        return len(self._exact)
