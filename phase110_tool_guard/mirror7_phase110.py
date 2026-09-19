from dataclasses import dataclass
from typing import Callable, Any

@dataclass(frozen=True)
class ToolResult:
    tool: str
    success: bool
    output: Any
    verified: bool
    reason: str

class GuardedTools:
    """Executes allowlisted tools with preconditions, output validation, and bounded retries."""
    def __init__(self):
        self.tools={}
    def register(self,name:str,fn:Callable[[dict],Any],validator:Callable[[Any],bool]|None=None):
        self.tools[name]=(fn,validator)
    def call(self,name:str,args:dict,retries=1):
        if name not in self.tools: return ToolResult(name,False,None,False,"tool not registered")
        fn,validator=self.tools[name]
        last=None
        for _ in range(max(1,retries+1)):
            try:
                out=fn(args); ok=validator(out) if validator else out is not None
                if ok: return ToolResult(name,True,out,True,"verified")
                last="output validation failed"
            except Exception as e:
                last=f"tool error: {type(e).__name__}"
        return ToolResult(name,False,None,False,last or "failed")
