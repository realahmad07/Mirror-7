from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist
from typing import Hashable, Sequence

Action = Hashable
Scalar = float | int | None
Observation = tuple[Scalar, ...]

def _obs(values: Sequence[Scalar]) -> Observation:
    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError("observation must be a non-empty tuple/list")
    out=[]
    for v in values:
        if v is not None and not isinstance(v,(int,float)):
            raise ValueError("observation values must be numeric or None")
        out.append(v)
    return tuple(out)

def _delta(a: Observation,b: Observation):
    if len(a)!=len(b):
        raise ValueError("observation dimensionality mismatch")
    for d,(x,y) in enumerate(zip(a,b)):
        if x is not None and y is not None:
            yield d,float(y)-float(x)

@dataclass
class RunningStat:
    n:int=0
    mean:float=0.0
    m2:float=0.0
    def add(self,x:float):
        self.n+=1
        q=x-self.mean
        self.mean+=q/self.n
        self.m2+=q*(x-self.mean)
    @property
    def variance(self): return self.m2/(self.n-1) if self.n>1 else 0.0
    @property
    def std(self): return sqrt(max(0.0,self.variance))

@dataclass(frozen=True)
class EffectDistribution:
    action:Action
    lag:int
    dimension:int
    mean:float
    std:float
    samples:int
    confidence:float

@dataclass(frozen=True)
class TransitionObservation:
    action:Action
    lag:int
    delta:tuple[float,...]
    visible:tuple[bool,...]

class EffectModel:
    """Multidimensional delayed stochastic transition evidence.

    Evidence is stored by (action, lag, dimension), while complete-vector
    observations are retained for cross-environment signatures.
    """
    def __init__(self,max_lag=8,min_samples=4):
        if max_lag<1 or min_samples<2: raise ValueError("invalid model bounds")
        self.max_lag=max_lag; self.min_samples=min_samples
        self.effects=defaultdict(RunningStat)
        self.vectors=defaultdict(list)
        self._recent=deque(maxlen=128)
    def add(self,action,before,after,lag):
        before,after=_obs(before),_obs(after)
        if lag<1 or lag>self.max_lag: raise ValueError("lag outside bounds")
        delta=[None]*len(before); vis=[False]*len(before)
        for d,x in _delta(before,after):
            self.effects[(action,lag,d)].add(x); delta[d]=x; vis[d]=True
        if any(vis):
            self.vectors[(action,lag)].append(tuple(delta))
            self._recent.append(TransitionObservation(action,lag,tuple(0.0 if x is None else x for x in delta),tuple(vis)))
    def estimate(self,action,lag,dimension):
        s=self.effects.get((action,lag,dimension))
        if s is None or s.n<self.min_samples: return None
        se=s.std/sqrt(s.n) if s.n>1 else float("inf")
        conf=2*NormalDist().cdf(abs(s.mean)/(se+1e-12))-1 if se!=float("inf") else 0.0
        return EffectDistribution(action,lag,dimension,s.mean,s.std,s.n,max(0.0,min(1.0,conf)))
    def best_lag(self,action,dimension):
        candidates=[]
        for lag in range(1,self.max_lag+1):
            e=self.estimate(action,lag,dimension)
            if e is None: continue
            se=e.std/sqrt(e.samples)
            z=abs(e.mean)/(se+1e-12)
            candidates.append((z,abs(e.mean),-lag,lag))
        return max(candidates)[-1] if candidates else None
    def signature(self):
        out={}
        for (a,l,d),s in self.effects.items():
            if s.n>=self.min_samples:
                out[(repr(a),l,d)]=round(s.mean,5)
        return tuple(sorted(out.items()))
    def mature(self):
        return any(s.n>=self.min_samples for s in self.effects.values())

@dataclass
class HiddenStateHypothesis:
    hypothesis_id:int
    signature:tuple
    support:int=1
    uncertainty:float=0.0

class Phase66Agent:
    """Phase 66 richer hidden-state learner.

    Adds multidimensional belief evidence, overlapping delayed effects,
    stochastic transition distributions, longer experiment planning, and
    transferable hypothesis snapshots. No hidden regime labels are accepted.
    """
    def __init__(self,*,max_lag=8,min_samples=4,merge_threshold=0.86,
                 mismatch_window=3,experiment_budget=24):
        if max_lag<2 or min_samples<2 or not 0<merge_threshold<=1 or mismatch_window<1 or experiment_budget<1:
            raise ValueError("invalid agent bounds")
        self.max_lag=max_lag; self.min_samples=min_samples
        self.merge_threshold=merge_threshold; self.mismatch_window=mismatch_window
        self.experiment_budget=experiment_budget
        self.model=EffectModel(max_lag,max(2,min_samples))
        self.hypotheses:list[HiddenStateHypothesis]=[]
        self.active_hypothesis=None
        self.revision_events=0; self.experiment_count=0
        self._mismatch=0; self._seen=set()
    def observe_effect(self,action,before,after,lag):
        self.model.add(action,before,after,lag)
        if self.model.mature(): self._revise()
    def observe_window(self,action,before,observations):
        """Attribute one action against several future observations.

        Every observation is retained, so effects from multiple actions may
        overlap in time rather than requiring a single completed action.
        """
        before=_obs(before)
        for lag,after in enumerate(observations,1):
            if lag>self.max_lag: break
            self.observe_effect(action,before,after,lag)
    def _distance(self,s1,s2):
        a=dict(s1); b=dict(s2); keys=set(a)|set(b)
        if not keys: return 1.0
        total=0.0
        for k in keys:
            x=a.get(k); y=b.get(k)
            if x is None or y is None: total+=1.0; continue
            scale=max(1.0,abs(x),abs(y))
            total+=min(1.0,abs(x-y)/scale)
        return total/len(keys)
    def _revise(self):
        sig=self.model.signature()
        if not sig: return
        if self.active_hypothesis is None:
            h=HiddenStateHypothesis(0,sig,1)
            self.hypotheses.append(h); self.active_hypothesis=0; self.revision_events+=1; return
        cur=self.hypotheses[self.active_hypothesis]
        dist=self._distance(cur.signature,sig)
        if dist>1-self.merge_threshold:
            self._mismatch+=1
        else:
            self._mismatch=0
        if self._mismatch>=self.mismatch_window:
            hid=len(self.hypotheses)
            self.hypotheses.append(HiddenStateHypothesis(hid,sig,1))
            self.active_hypothesis=hid; self.revision_events+=1; self._mismatch=0
        else:
            cur.support+=1
    def infer(self,model=None):
        model=model or self.model
        if not model.mature(): return None
        sig=model.signature()
        if not self.hypotheses: return None
        ranked=sorted((self._distance(h.signature,sig),h.hypothesis_id) for h in self.hypotheses)
        return ranked[0][1] if ranked[0][0] <= 1-self.merge_threshold else None
    def distribution(self,action,lag,dimension):
        return self.model.estimate(action,lag,dimension)
    def predict_mean(self,observation,action,horizon):
        obs=list(_obs(observation)); found=False
        for d,x in enumerate(obs):
            if x is None: continue
            lag=self.model.best_lag(action,d)
            if lag is None or lag>horizon: continue
            e=self.model.estimate(action,lag,d)
            if e: obs[d]=float(x)+e.mean; found=True
        return tuple(obs) if found else None
    def design_experiment(self,legal,*,slots=8):
        legal=tuple(dict.fromkeys(legal))
        if not legal or self.experiment_count>=self.experiment_budget: return None
        slots=max(2,min(self.max_lag,slots))
        candidates=[]
        for action in legal:
            if action=="__noop__": continue
            score=0.0
            for h1 in self.hypotheses:
                for h2 in self.hypotheses:
                    if h1.hypothesis_id>=h2.hypothesis_id: continue
                    a=dict(h1.signature); b=dict(h2.signature)
                    for k in set(a)&set(b):
                        if k[0]==repr(action): score+=abs(a[k]-b[k])
            # Prefer actions with evidence gaps as well as disagreement.
            known=sum(1 for k in dict(self.model.signature()) if k[0]==repr(action))
            score+=5.0 if known==0 else 0.0
            if "__noop__" in legal:
                for position in range(slots):
                    seq=(("__noop__",)*position)+(action,)+(("__noop__",)*(slots-position-1))
                    if seq in self._seen:
                        continue
                    candidates.append((score,repr(seq),seq))
            else:
                seq=(action,)
                if seq not in self._seen:
                    candidates.append((score,repr(seq),seq))
        if not candidates: return None
        candidates.sort(reverse=True)
        score,_,seq=candidates[0]; self._seen.add(seq); self.experiment_count+=1
        return {"sequence":seq,"score":score,"reason":"hidden-state-disagreement-plus-coverage"}
    def export_hypotheses(self):
        return tuple((h.hypothesis_id,h.signature,h.support) for h in self.hypotheses)
    def import_hypotheses(self,snapshot):
        self.hypotheses=[]
        for hid,sig,support in snapshot:
            self.hypotheses.append(HiddenStateHypothesis(int(hid),tuple(sig),int(support)))
        self.active_hypothesis=0 if self.hypotheses else None
    def fail_closed(self):
        return {"hypotheses":len(self.hypotheses),"active_hypothesis":self.active_hypothesis,
                "revisions":self.revision_events,"experiments":self.experiment_count}
