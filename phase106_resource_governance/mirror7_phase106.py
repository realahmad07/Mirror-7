from dataclasses import dataclass
from collections import OrderedDict
from typing import Any

@dataclass(frozen=True)
class ResourceReport:
    admitted: bool
    memory_used: int
    memory_limit: int
    steps_used: int
    step_limit: int
    reason: str

class ResourceGovernor:
    """Hard bounds for memory and compute-like step budgets with deterministic eviction."""
    def __init__(self,memory_limit=64,step_limit=128):
        if memory_limit<1 or step_limit<1: raise ValueError("limits must be positive")
        self.memory_limit=memory_limit; self.step_limit=step_limit; self.memory=OrderedDict(); self.steps=0

    def tick(self,n=1):
        if n<0 or self.steps+n>self.step_limit: return False
        self.steps+=n; return True

    def remember(self,key:str,value:Any,priority:float=0.0):
        if not self.tick(): return ResourceReport(False,len(self.memory),self.memory_limit,self.steps,self.step_limit,"step budget exhausted")
        self.memory[key]=(float(priority),value)
        self.memory.move_to_end(key)
        while len(self.memory)>self.memory_limit:
            worst=min(self.memory.items(),key=lambda kv:(kv[1][0],list(self.memory.keys()).index(kv[0])))
            del self.memory[worst[0]]
        return ResourceReport(True,len(self.memory),self.memory_limit,self.steps,self.step_limit,"admitted")

    def reset_steps(self): self.steps=0
