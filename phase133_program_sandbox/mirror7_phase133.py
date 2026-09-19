from dataclasses import dataclass
from typing import Callable,Any
@dataclass(frozen=True)
class SandboxResult:
    ok:bool; value:Any; steps:int; reason:str
class ProgramSandbox:
    def run(self,program:Callable[[Any],Any],state,step_budget=1):
        if step_budget<1: return SandboxResult(False,None,0,"invalid budget")
        try: value=program(state)
        except Exception as e: return SandboxResult(False,None,1,f"error: {type(e).__name__}")
        return SandboxResult(True,value,1,"completed")
