from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class CurriculumResult:
    selected: object|None
    score: float
    used: int
    status: str

class AutonomousCurriculum:
    """Bounded experiment selection driven by uncertainty and novelty."""
    def __init__(self,budget=8): self.budget=budget; self.used=0; self.history=[]
    def select(self,candidates,score_fn):
        if self.used>=self.budget or not candidates: return None
        ranked=[]
        for c in candidates:
            try: score=float(score_fn(c))
            except Exception: continue
            if score!=score or score<0: continue
            ranked.append((score,c))
        if not ranked: return None
        ranked.sort(key=lambda z:(-z[0],repr(z[1])))
        score,c=ranked[0]; self.used+=1; self.history.append((c,score)); return c
    def run(self,candidates,score_fn,stop_fn=lambda c:False):
        while self.used<self.budget:
            c=self.select(candidates,score_fn)
            if c is None: break
            if stop_fn(c): return CurriculumResult(c,float(score_fn(c)),self.used,"target_reached")
            candidates=[x for x in candidates if x!=c]
        return CurriculumResult(None,0.0,self.used,"budget_exhausted" if self.used>=self.budget else "no_candidate")
    def fail_closed(self): return {"budget":self.budget,"used":self.used,"history":len(self.history)}
