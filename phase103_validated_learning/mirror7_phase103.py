from dataclasses import dataclass
from typing import Any, Callable

@dataclass(frozen=True)
class Candidate:
    key: str
    value: Any
    support: int
    confirmations: int
    contradictions: int
    confidence: float

class ValidatedLearner:
    """Accepts new knowledge only after bounded corroboration and contradiction checks."""
    def __init__(self, min_support=2, min_confidence=0.75, max_rules=128):
        if min_support < 1 or not 0 <= min_confidence <= 1 or max_rules < 1: raise ValueError("invalid bounds")
        self.min_support=min_support; self.min_confidence=min_confidence; self.max_rules=max_rules
        self._rows={}

    def observe(self,key:str,value:Any,validator:Callable[[Any,Any],bool] | None=None):
        row=self._rows.setdefault(key,{"value":value,"support":0,"confirmations":0,"contradictions":0})
        if row["support"] and validator and not validator(row["value"],value):
            row["contradictions"]+=1
        elif row["support"] and row["value"] != value:
            row["contradictions"]+=1
        else:
            row["confirmations"]+=1
        row["support"] += 1
        conf=row["confirmations"]/max(1,row["confirmations"] + 0.5*row["contradictions"])
        if row["contradictions"] and conf < self.min_confidence:
            return None
        if row["support"] < self.min_support or conf < self.min_confidence:
            return None
        return Candidate(key,row["value"],row["support"],row["confirmations"],row["contradictions"],conf)

    def accepted(self):
        out=[]
        for key,row in self._rows.items():
            conf=row["confirmations"]/max(1,row["confirmations"] + 0.5*row["contradictions"])
            if row["support"]>=self.min_support and conf>=self.min_confidence:
                out.append(Candidate(key,row["value"],row["support"],row["confirmations"],row["contradictions"],conf))
        return out[:self.max_rules]
