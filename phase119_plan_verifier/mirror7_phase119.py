from dataclasses import dataclass
from typing import Callable, Any

@dataclass(frozen=True)
class PlanCheck:
    valid: bool
    checked: int
    reason: str

class PlanVerifier:
    """Simulates a bounded plan and rejects any step that violates its verifier."""
    def verify(self,steps:list[Callable[[Any],Any]],initial:Any,goal:Callable[[Any],bool],max_steps:int)->PlanCheck:
        if len(steps)>max_steps: return PlanCheck(False,0,"step budget exceeded")
        state=initial
        for i,step in enumerate(steps):
            try: state=step(state)
            except Exception as e: return PlanCheck(False,i,f"simulation error: {type(e).__name__}")
            if state is None: return PlanCheck(False,i+1,"unknown state")
        return PlanCheck(bool(goal(state)),len(steps),"goal reached" if goal(state) else "goal not reached")
