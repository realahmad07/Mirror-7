from dataclasses import dataclass
from collections import OrderedDict

@dataclass(frozen=True)
class Retrieved:
    key: str
    value: object
    score: float

class ContextMemory:
    """Bounded context-overlap retrieval with confidence and recency weighting."""
    def __init__(self,max_items=128): self.max_items=max_items; self.items=OrderedDict(); self.clock=0
    def remember(self,key,value,context:set[str],confidence=.5):
        self.clock+=1; self.items[key]=(set(context),value,max(0.,min(1.,confidence)),self.clock); self.items.move_to_end(key)
        while len(self.items)>self.max_items: self.items.popitem(last=False)
    def retrieve(self,context:set[str],limit=3,min_score=0.0):
        now=self.clock or 1; out=[]
        for k,(ctx,val,conf,t) in self.items.items():
            overlap=len(ctx & context)/max(1,len(ctx|context)); recency=1.0/(1.0+0.05*(now-t)); score=.7*overlap+.2*conf+.1*recency
            if score>=min_score: out.append(Retrieved(k,val,score))
        return tuple(sorted(out,key=lambda x:x.score,reverse=True)[:max(0,limit)])
