from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class WorkspaceItem:
    channel:str; value:Any; confidence:float; source:str

class CognitiveWorkspace:
    """Bounded shared workspace for explicit cross-module state."""
    def __init__(self,max_items=64): self.max_items=max_items; self.items=[]
    def publish(self,channel,value,confidence=1.0,source="unknown"):
        c=max(0.0,min(1.0,float(confidence))); self.items.append(WorkspaceItem(channel,value,c,source))
        self.items=sorted(self.items,key=lambda x:x.confidence,reverse=True)[:self.max_items]
    def read(self,channel,min_confidence=0.0):
        return tuple(x for x in self.items if x.channel==channel and x.confidence>=min_confidence)
    def latest(self,channel,min_confidence=0.0):
        for item in reversed(self.items):
            if item.channel==channel and item.confidence>=min_confidence:
                return item
        return None
    def snapshot(self): return tuple(self.items)
