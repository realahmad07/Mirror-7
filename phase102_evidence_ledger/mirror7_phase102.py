from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Evidence:
    claim: str
    value: Any
    source: str
    confidence: float
    valid: bool=True

class EvidenceLedger:
    """Bounded, provenance-aware evidence with explicit conflict state."""
    def __init__(self, max_items: int=256):
        if max_items < 1: raise ValueError("max_items must be positive")
        self.max_items=max_items
        self.items: list[Evidence]=[]

    def add(self, evidence: Evidence) -> None:
        c=max(0.0,min(1.0,float(evidence.confidence)))
        self.items.append(Evidence(evidence.claim,evidence.value,evidence.source,c,bool(evidence.valid)))
        self._compact()

    def _compact(self):
        if len(self.items) <= self.max_items: return
        self.items=sorted(self.items,key=lambda e:(e.valid,e.confidence),reverse=True)[:self.max_items]

    def status(self, claim: str) -> str:
        xs=[e for e in self.items if e.claim==claim and e.valid]
        if not xs: return "unknown"
        vals={repr(e.value) for e in xs}
        if len(vals)>1: return "conflict"
        return "supported"

    def best(self, claim: str) -> Evidence | None:
        xs=[e for e in self.items if e.claim==claim and e.valid]
        return max(xs,key=lambda e:e.confidence,default=None)
