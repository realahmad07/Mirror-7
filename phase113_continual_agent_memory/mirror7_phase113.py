from dataclasses import dataclass
from collections import OrderedDict

@dataclass(frozen=True)
class MemoryItem:
    key: str
    value: object
    confidence: float
    uses: int=0

class AgentMemory:
    """Bounded persistent-style memory with confidence-aware reinforcement."""
    def __init__(self,max_items=64): self.max_items=max_items; self.items=OrderedDict()
    def remember(self,key,value,confidence):
        c=max(0.,min(1.,float(confidence)))
        old=self.items.get(key)
        uses=old.uses+1 if old else 0
        self.items[key]=MemoryItem(key,value,max(c,old.confidence) if old else c,uses)
        self.items.move_to_end(key)
        while len(self.items)>self.max_items:
            victim=min(self.items.values(),key=lambda x:(x.confidence,x.uses))
            del self.items[victim.key]
        return self.items[key]
    def recall(self,key,min_confidence=0.0):
        x=self.items.get(key); return x if x and x.confidence>=min_confidence else None
