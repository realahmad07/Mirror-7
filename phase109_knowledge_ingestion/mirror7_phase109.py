from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class KnowledgeRecord:
    claim: str
    value: Any
    source: str
    confidence: float
    accepted: bool
    conflict: bool

class ValidatedKnowledge:
    """Ingests external information only as evidence; conflicts remain explicit."""
    def __init__(self, max_items=256, min_confidence=.75):
        self.max_items=max_items; self.min_confidence=min_confidence; self.records=[]
    def ingest(self, claim, value, source, confidence, *, corroborates=True):
        c=max(0.0,min(1.0,float(confidence)))
        conflict=any(r.claim==claim and r.accepted and r.value!=value for r in self.records)
        accepted=bool(corroborates and c>=self.min_confidence and not conflict)
        r=KnowledgeRecord(claim,value,source,c,accepted,conflict); self.records.append(r)
        if len(self.records)>self.max_items: self.records=self.records[-self.max_items:]
        return r
    def query(self, claim):
        xs=[r for r in self.records if r.claim==claim and r.accepted]
        if not xs: return None
        return max(xs,key=lambda r:r.confidence)
