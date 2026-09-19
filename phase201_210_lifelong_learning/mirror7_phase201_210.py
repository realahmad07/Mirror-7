"""Phases 201-210: bounded lifelong persistent learning.

Evidence is versioned, confidence-weighted and provenance-tagged. Consolidation
requires repeated support; contradictions remain visible and can trigger
abstention. Resource bounds prevent unbounded memory growth.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Iterable


@dataclass(frozen=True)
class Evidence:
    key: str
    value: str
    source: str
    episode: int
    confidence: float=1.0

    @property
    def fingerprint(self)->str:
        return sha256(repr((self.key,self.value,self.source,self.episode)).encode()).hexdigest()


@dataclass
class PersistentLearner:
    capacity: int=256
    evidence: list[Evidence]=field(default_factory=list)
    counts: dict[tuple[str,str],float]=field(default_factory=dict)
    versions: int=0

    def observe(self,e: Evidence)->None:
        if not 0.0<=e.confidence<=1.0: raise ValueError("confidence outside [0,1]")
        self.evidence.append(e)
        self.evidence=self.evidence[-self.capacity:]
        k=(e.key,e.value)
        self.counts[k]=self.counts.get(k,0.0)+e.confidence
        self.versions+=1

    def support(self,key:str,value:str)->float:
        return self.counts.get((key,value),0.0)

    def candidates(self,key:str)->tuple[tuple[str,float],...]:
        vals=[(v,c) for (k,v),c in self.counts.items() if k==key]
        return tuple(sorted(vals,key=lambda x:(-x[1],x[0])))

    def recall(self,key:str,min_support:float=1.0)->str|None:
        vals=self.candidates(key)
        if not vals or vals[0][1]<min_support: return None
        if len(vals)>1 and vals[0][1]==vals[1][1]: return None
        return vals[0][0]

    def contradiction(self,key:str)->bool:
        vals=self.candidates(key)
        return len(vals)>1 and vals[0][1]>0 and vals[1][1]>0

    def consolidate(self)->dict[str,Any]:
        stable={}
        for key in sorted({k for k,_ in self.counts}):
            v=self.recall(key,min_support=2.0)
            if v is not None: stable[key]=v
        return {"stable":stable,"contradictions":tuple(k for k in sorted({k for k,_ in self.counts}) if self.contradiction(k)),
                "version":self.versions,"size":len(self.evidence)}

    def checkpoint(self)->dict[str,Any]:
        return {"capacity":self.capacity,"version":self.versions,
                "evidence":tuple((e.key,e.value,e.source,e.episode,round(e.confidence,6)) for e in self.evidence)}

    def restore(self,checkpoint:dict[str,Any])->None:
        self.capacity=int(checkpoint["capacity"])
        self.evidence=[Evidence(*x) for x in checkpoint["evidence"]]
        self.counts={}
        for e in self.evidence:
            self.counts[(e.key,e.value)]=self.counts.get((e.key,e.value),0)+e.confidence
        self.versions=int(checkpoint["version"])

    def forget_low_confidence(self,threshold:float)->int:
        before=len(self.evidence)
        self.evidence=[e for e in self.evidence if e.confidence>=threshold]
        self.evidence=self.evidence[-self.capacity:]
        self.counts={}
        for e in self.evidence:
            self.counts[(e.key,e.value)]=self.counts.get((e.key,e.value),0)+e.confidence
        return before-len(self.evidence)

    def replay(self,items:Iterable[Evidence])->None:
        for e in items: self.observe(e)
