from dataclasses import dataclass
from typing import Any, Callable

@dataclass(frozen=True)
class ActionReport:
    action: str
    executed: bool
    verified: bool
    output: Any
    reason: str

class ActionExecutor:
    """Executes allowlisted actions and verifies postconditions before promotion."""
    def __init__(self): self.actions={}
    def register(self,name:str,fn:Callable[[dict],Any],precondition=None,postcondition=None):
        self.actions[name]=(fn,precondition,postcondition)
    def execute(self,name:str,args:dict):
        if name not in self.actions: return ActionReport(name,False,False,None,"action not registered")
        fn,pre,post=self.actions[name]
        if pre and not pre(args): return ActionReport(name,False,False,None,"precondition failed")
        try:
            out=fn(args)
        except Exception as e:
            return ActionReport(name,False,False,None,f"action error: {type(e).__name__}")
        ok=post(out) if post else out is not None
        return ActionReport(name,True,bool(ok),out,"verified" if ok else "postcondition failed")
