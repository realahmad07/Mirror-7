from __future__ import annotations
from dataclasses import dataclass
from math import isfinite

@dataclass(frozen=True)
class GroundedConcept:
    concept_id:int
    fingerprint:tuple
    support:int

class MixedViewGrounder:
    """Bounded structural normalization for numeric, grid, and symbolic views."""
    def __init__(self,max_concepts=128):
        if max_concepts<1: raise ValueError("invalid max_concepts")
        self.max_concepts=max_concepts; self.concepts={}

    @staticmethod
    def fingerprint(view):
        if isinstance(view,str):
            tokens=tuple(view.split())
            if not tokens: raise ValueError("empty symbolic view")
            counts=tuple(sorted((t,tokens.count(t)) for t in set(tokens)))
            adjacency=tuple((tokens[i],tokens[i+1]) for i in range(len(tokens)-1))
            return ("symbolic",len(tokens),counts,adjacency)
        if isinstance(view,(list,tuple)) and view:
            if all(isinstance(x,(int,float)) for x in view):
                x=tuple(float(v) for v in view)
                if not all(isfinite(v) for v in x): raise ValueError("invalid numeric view")
                rises=sum(b>a for a,b in zip(x,x[1:])); falls=sum(b<a for a,b in zip(x,x[1:]))
                return ("numeric",len(x),round(sum(x)/len(x),5),round(max(x)-min(x),5),rises,falls)
            if all(isinstance(row,(list,tuple)) for row in view):
                rows=tuple(tuple(float(v) for v in row) for row in view)
                if any(not row for row in rows) or len({len(r) for r in rows})!=1: raise ValueError("invalid grid")
                flat=[v for row in rows for v in row]
                if not all(isfinite(v) for v in flat): raise ValueError("invalid grid")
                horizontal=sum(flat[i]<flat[i+1] for i in range(len(flat)-1))
                return ("grid",len(rows),len(rows[0]),round(sum(flat)/len(flat),5),round(max(flat)-min(flat),5),horizontal)
        raise ValueError("unsupported view")

    def observe(self,view):
        fp=self.fingerprint(view); c=self.concepts.get(fp)
        if c is not None:
            c=GroundedConcept(c.concept_id,c.fingerprint,c.support+1); self.concepts[fp]=c; return c
        if len(self.concepts)>=self.max_concepts: raise RuntimeError("concept budget exhausted")
        c=GroundedConcept(len(self.concepts),fp,1); self.concepts[fp]=c; return c

    def equivalent(self,a,b): return self.fingerprint(a)==self.fingerprint(b)
    def fail_closed(self): return {"concepts":len(self.concepts)}
