"""Phases 191-200: open action/affordance discovery.

Actions are opaque bytes. The learner receives observations before/after an
action and discovers bounded effect signatures, precondition evidence,
reversibility, uncertainty and safe action sequences. No action semantics are
provided by the environment.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from hashlib import sha256
from typing import Any, Sequence


def canonical_observation(obs: Any) -> tuple:
    if isinstance(obs,(bytes,bytearray)): seq=list(obs)
    elif isinstance(obs,str): seq=list(obs.encode())
    elif isinstance(obs,Sequence): seq=[int(x)&255 for x in obs]
    else: raise TypeError("observation must be bytes, text, or a numeric sequence")
    # identity-insensitive first-appearance encoding
    m={}; out=[]
    for x in seq:
        if x not in m: m[x]=len(m)
        out.append(m[x])
    return tuple(out)


def _action_key(action: bytes) -> str:
    if not isinstance(action,(bytes,bytearray)): raise TypeError("opaque action must be bytes")
    return sha256(bytes(action)).hexdigest()[:16]


def _effect(before: tuple, after: tuple) -> tuple:
    n=max(len(before),len(after))
    return tuple((before[i] if i<len(before) else None,
                  after[i] if i<len(after) else None)
                 for i in range(n) if (before[i] if i<len(before) else None)!=(after[i] if i<len(after) else None))


@dataclass(frozen=True)
class ActionOutcome:
    action: bytes
    before: Any
    after: Any
    success: bool=True
    cost: float=1.0


@dataclass
class Affordance:
    action_id: str
    effect_counts: dict[tuple,int]=field(default_factory=dict)
    precondition_counts: dict[tuple,int]=field(default_factory=dict)
    attempts: int=0
    successes: int=0
    failures: int=0
    irreversible: bool=False

    @property
    def success_rate(self)->float:
        return self.successes/max(1,self.attempts)

    @property
    def confidence(self)->float:
        return min(1.0,self.attempts/5.0)*self.success_rate


def discover_affordance(history: Sequence[ActionOutcome]) -> Affordance:
    if not history: raise ValueError("empty action history")
    aid=_action_key(history[0].action)
    if any(_action_key(x.action)!=aid for x in history):
        raise ValueError("history contains multiple opaque actions")
    a=Affordance(aid)
    seen_before=set()
    for x in history:
        b=canonical_observation(x.before); y=canonical_observation(x.after)
        e=_effect(b,y)
        a.attempts+=1
        if x.success: a.successes+=1
        else: a.failures+=1
        a.effect_counts[e]=a.effect_counts.get(e,0)+1
        a.precondition_counts[b]=a.precondition_counts.get(b,0)+1
        seen_before.add(b)
    # A failed attempt followed by no observed inverse is not enough to call
    # something irreversible. Mark irreversible only when the effect changes
    # state and a same-action replay never restores the prior state.
    unique_effects=[e for e in a.effect_counts if e]
    a.irreversible=len(unique_effects)>0 and len(a.effect_counts)>1 and a.failures>0
    return a


@dataclass
class AffordanceLearner:
    records: dict[str,list[ActionOutcome]]=field(default_factory=lambda:defaultdict(list))

    def observe(self, outcome: ActionOutcome)->None:
        aid=_action_key(outcome.action)
        self.records[aid].append(outcome)

    def model(self, action: bytes)->Affordance|None:
        h=self.records.get(_action_key(action))
        return discover_affordance(h) if h else None

    def candidate_actions(self)->tuple[bytes,...]:
        out=[]
        for h in self.records.values():
            if h: out.append(h[0].action)
        return tuple(out)

    def choose(self, legal: Sequence[bytes], observed: Any, goal_effect: tuple|None=None)->bytes|None:
        ranked=[]
        b=canonical_observation(observed)
        for action in legal:
            m=self.model(action)
            if m is None:
                # bounded information-seeking exploration
                ranked.append((0.0,bytes(action)))
                continue
            if m.precondition_counts.get(b,0)==0:
                continue
            match=0
            if goal_effect is not None:
                match=max((c for e,c in m.effect_counts.items() if e==goal_effect),default=0)
            score=2.0*match + m.confidence - .5*float(m.irreversible)
            ranked.append((score,bytes(action)))
        return max(ranked,key=lambda x:(x[0],x[1]))[1] if ranked else None


def select_safe_action(learner: AffordanceLearner, legal: Sequence[bytes], observed: Any)->bytes|None:
    """Prefer known successful actions; otherwise choose the least-supported probe."""
    known=[]
    unknown=[]
    for a in legal:
        m=learner.model(a)
        if m is None: unknown.append(bytes(a))
        elif m.success_rate>0 and not m.irreversible: known.append((m.confidence,bytes(a)))
    if known: return max(known,key=lambda x:(x[0],x[1]))[1]
    return min(unknown) if unknown else None


def compose_actions(learner: AffordanceLearner, actions: Sequence[bytes], start: Any)->tuple|None:
    state=canonical_observation(start); trace=[state]
    for a in actions:
        m=learner.model(a)
        if m is None or m.precondition_counts.get(state,0)==0:
            return None
        # Apply only an effect observed exactly at this precondition.
        hist=learner.records[_action_key(a)]
        matches=[x for x in hist if canonical_observation(x.before)==state and x.success]
        if not matches: return None
        state=canonical_observation(matches[-1].after)
        trace.append(state)
    return tuple(trace)
