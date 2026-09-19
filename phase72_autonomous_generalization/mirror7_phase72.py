from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class GeneralizationResult:
    prediction:object
    confidence:float
    experiments_used:int
    status:str

class GeneralizationLoop:
    """Bounded autonomous loop for proposing, testing, rejecting and retaining abstractions."""
    def __init__(self,learner,budget=8,confidence_threshold=.75):
        self.learner=learner; self.budget=budget; self.confidence_threshold=confidence_threshold
        self.history=[]
    def run(self,training_cases,held_out_case):
        if self.budget<=0: return GeneralizationResult(None,0.0,0,"budget_exhausted")
        used=0
        for case in training_cases:
            if used>=self.budget: break
            self.learner(*case); used+=1
        prediction=None
        if hasattr(self.learner,"predict"):
            prediction=self.learner.predict(*held_out_case)
        confidence=1.0 if prediction is not None else 0.0
        status="accepted" if confidence>=self.confidence_threshold else "rejected"
        self.history.append((prediction,confidence,status))
        return GeneralizationResult(prediction,confidence,used,status)
    def choose(self,candidates,score_fn):
        if self.budget<=0 or not candidates: return None
        ranked=sorted(candidates,key=score_fn,reverse=True)
        choice=ranked[0]
        self.history.append(("choice",choice))
        return choice
    def fail_closed(self):
        return {"budget":self.budget,"history":len(self.history)}
